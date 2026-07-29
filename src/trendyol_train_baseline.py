#!/usr/bin/env python3
"""Train a baseline relevance classifier and produce a Kaggle submission."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

# src/ modüllerini import edebilmek için
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_features import (  # noqa: E402
    FEATURE_NAMES,
    attach_tfidf_similarity,
    build_lexical_features,
    make_item_text,
    normalize_category_path,
    normalize_text,
)
from experiment_utils import experiment_dir, stable_term_folds, validate_probability_frame, validate_submission, write_config  # noqa: E402

# --- Config ---
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "5000"))
THRESHOLD_GRID = np.arange(0.20, 0.81, 0.02)
SUBMISSION_CHUNK_SIZE = int(os.getenv("SUBMISSION_CHUNK_SIZE", "250000"))
USE_FULL_SUBMISSION = os.getenv("USE_FULL_SUBMISSION", "true").lower() == "true"
SUBMISSION_SAMPLE_SIZE = int(os.getenv("SUBMISSION_SAMPLE_SIZE", "200000"))

TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "40000"))
TFIDF_ITEM_POOL_SIZE = int(os.getenv("TFIDF_ITEM_POOL_SIZE", "120000"))

ROOT = SRC_DIR.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID", "baseline_v2")
EXPERIMENT_DIR = experiment_dir(ARTIFACTS_DIR, EXPERIMENT_ID)
TRAIN_PATH = Path(os.getenv("TRAIN_PATH", str(ARTIFACTS_DIR / "train_with_negatives.parquet")))
if not TRAIN_PATH.exists():
    TRAIN_PATH = Path(os.getenv("TRAIN_PATH_CSV", str(ARTIFACTS_DIR / "train_with_negatives.csv")))

MODEL_PATH = EXPERIMENT_DIR / "model.joblib"
THRESHOLD_PATH = EXPERIMENT_DIR / "threshold.txt"
CV_REPORT_PATH = EXPERIMENT_DIR / "cv_report.csv"
SUBMISSION_PATH = EXPERIMENT_DIR / "submission.csv"
PROBABILITY_PATH = EXPERIMENT_DIR / "probabilities.parquet"
# P0: hybrid/ratio submission'ları için sabit konumda, sade (id, prob) şeması.
BASELINE_PROBABILITIES_PATH = Path(
    os.getenv("BASELINE_PROBABILITIES_PATH", str(ARTIFACTS_DIR / "baseline_probabilities.parquet"))
)
FOLD_PATH = EXPERIMENT_DIR / "term_folds.parquet"


def locate_data_file(name: str) -> Path:
    candidates = [
        DATA_DIR / name,
        ROOT / "data" / name,
        Path.cwd() / "data" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{name} bulunamadı. DATA_DIR={DATA_DIR}")


def load_training_frame() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Eğitim verisi yok: {TRAIN_PATH}\n"
            "Önce negatif örnekleme çalıştırın:\n"
            "  USE_FULL_DATA=false POS_SAMPLE_SIZE=50000 python src/trendyol_negative_sampling_kaggle.py"
        )

    if TRAIN_PATH.suffix == ".parquet":
        frame = pd.read_parquet(TRAIN_PATH)
    else:
        frame = pd.read_csv(TRAIN_PATH)

    required = {
        "term_id", "item_id", "query", "title", "category",
        "brand", "gender", "age_group", "attributes", "label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Eğitim verisinde eksik kolonlar: {sorted(missing)}")

    if len(frame) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Eğitim verisi çok küçük ({len(frame):,} satır). "
            f"En az {MIN_TRAIN_ROWS:,} satır gerekli. Negatif örnekleme script'ini "
            "daha büyük POS_SAMPLE_SIZE ile çalıştırın."
        )

    if "sample_weight" not in frame.columns:
        frame["sample_weight"] = 1.0

    for column in ["query", "title", "brand", "gender", "age_group", "attributes"]:
        frame[column] = frame[column].astype("string").fillna("").map(normalize_text)
    frame["category"] = frame["category"].astype("string").fillna("").map(normalize_category_path)

    frame["label"] = frame["label"].astype(int)
    frame["sample_weight"] = frame["sample_weight"].astype(float)
    return frame.reset_index(drop=True)


def load_items_catalog() -> pd.DataFrame:
    items = pd.read_csv(
        locate_data_file("items.csv"),
        usecols=["item_id", "title", "category", "brand", "gender", "age_group", "attributes"],
        dtype="string",
    )
    items = items.drop_duplicates("item_id").reset_index(drop=True)
    for column in items.columns:
        if column not in {"item_id", "category"}:
            items[column] = items[column].fillna("").astype("string").map(normalize_text)
    items["category"] = items["category"].fillna("").astype("string").map(normalize_category_path)
    items["item_text"] = make_item_text(items)
    return items


def load_terms() -> pd.DataFrame:
    terms = pd.read_csv(
        locate_data_file("terms.csv"),
        usecols=["term_id", "query"],
        dtype={"term_id": "string", "query": "string"},
    )
    terms = terms.drop_duplicates("term_id")
    terms["query"] = terms["query"].fillna("").astype("string").map(normalize_text)
    return terms


class TfidfSimilarityIndex:
    """TF-IDF vektörleri; query/item benzerliği için önbellek."""

    def __init__(self, items: pd.DataFrame) -> None:
        self.items = items.set_index("item_id", drop=False)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=TFIDF_MAX_FEATURES,
            dtype=np.float32,
        )

    def fit(self, item_pool: pd.DataFrame) -> None:
        self.vectorizer.fit(item_pool["item_text"].tolist())
        self._item_matrix = normalize(
            self.vectorizer.transform(item_pool["item_text"].tolist())
        )
        self._item_ids = item_pool["item_id"].to_numpy()
        self._item_id_to_row = {
            item_id: idx for idx, item_id in enumerate(self._item_ids)
        }
        title_matrix = normalize(
            self.vectorizer.transform(item_pool["title"].tolist())
        )
        self._title_vectors = {
            item_id: title_matrix[row]
            for item_id, row in self._item_id_to_row.items()
        }
        self._item_vectors = {
            item_id: self._item_matrix[row]
            for item_id, row in self._item_id_to_row.items()
        }

    def transform_terms(self, terms: pd.DataFrame) -> dict[str, np.ndarray]:
        matrix = normalize(self.vectorizer.transform(terms["query"].tolist()))
        return {str(term_id): matrix[idx] for idx, term_id in enumerate(terms["term_id"].tolist())}

    def extend_items(self, extra_items: pd.DataFrame) -> None:
        """Eğitim/test'te havuzda olmayan item'lar için vektör ekle."""
        missing_ids = set(extra_items["item_id"].astype(str)) - set(self._item_id_to_row)
        if not missing_ids:
            return
        missing = extra_items[extra_items["item_id"].astype(str).isin(missing_ids)].drop_duplicates("item_id")
        if "item_text" not in missing.columns:
            missing = missing.copy()
            missing["item_text"] = make_item_text(missing)
        title_matrix = normalize(self.vectorizer.transform(missing["title"].tolist()))
        item_matrix = normalize(self.vectorizer.transform(missing["item_text"].tolist()))
        for idx, row in enumerate(missing.itertuples(index=False)):
            item_id = str(row.item_id)
            self._title_vectors[item_id] = title_matrix[idx]
            self._item_vectors[item_id] = item_matrix[idx]

    @property
    def title_vectors(self) -> dict[str, np.ndarray]:
        return self._title_vectors

    @property
    def item_vectors(self) -> dict[str, np.ndarray]:
        return self._item_vectors


def build_feature_matrix(
    frame: pd.DataFrame,
    tfidf_index: TfidfSimilarityIndex,
    query_vectors: dict[str, np.ndarray],
) -> pd.DataFrame:
    tfidf_index.extend_items(
        frame[["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]]
        .drop_duplicates("item_id")
    )
    lexical = build_lexical_features(frame)
    enriched = attach_tfidf_similarity(
        frame.assign(term_id=frame["term_id"].astype(str), item_id=frame["item_id"].astype(str)),
        query_vectors,
        tfidf_index.title_vectors,
        tfidf_index.item_vectors,
    )
    features = lexical.join(enriched[["tfidf_title_sim", "tfidf_item_sim"]])
    return features[FEATURE_NAMES].astype(np.float32)


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_threshold, best_score = 0.50, -1.0
    for threshold in THRESHOLD_GRID:
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_true, pred, average="macro")
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def cross_validate(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> tuple[HistGradientBoostingClassifier, float, pd.DataFrame]:
    splitter = GroupKFold(n_splits=N_SPLITS)
    oof_proba = np.zeros(len(labels), dtype=np.float32)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(features, labels, groups)):
        model = HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.08,
            max_iter=250,
            min_samples_leaf=40,
            l2_regularization=0.1,
            random_state=RANDOM_STATE + fold,
        )
        model.fit(
            features.iloc[train_idx],
            labels[train_idx],
            sample_weight=weights[train_idx],
        )
        valid_proba = model.predict_proba(features.iloc[valid_idx])[:, 1]
        oof_proba[valid_idx] = valid_proba

        threshold, macro_f1 = tune_threshold(labels[valid_idx], valid_proba)
        fold_rows.append(
            {
                "fold": fold,
                "valid_rows": len(valid_idx),
                "valid_terms": len(np.unique(groups[valid_idx])),
                "macro_f1": macro_f1,
                "threshold": threshold,
            }
        )
        print(
            f"Fold {fold}: Macro F1={macro_f1:.4f} @ threshold={threshold:.2f} "
            f"({len(valid_idx):,} satır, {len(np.unique(groups[valid_idx])):,} term)"
        )

    global_threshold, global_score = tune_threshold(labels, oof_proba)
    print(f"\nOOF Macro F1: {global_score:.4f} @ threshold={global_threshold:.2f}")

    final_model = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.08,
        max_iter=250,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    final_model.fit(features, labels, sample_weight=weights)
    report = pd.DataFrame(fold_rows)
    report.loc[len(report)] = {
        "fold": "oof",
        "valid_rows": len(labels),
        "valid_terms": len(np.unique(groups)),
        "macro_f1": global_score,
        "threshold": global_threshold,
    }
    return final_model, global_threshold, report


def enrich_pairs(pairs: pd.DataFrame, terms: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    enriched = (
        pairs.merge(terms, on="term_id", how="left", validate="many_to_one")
        .merge(items, on="item_id", how="left", validate="many_to_one")
    )
    for column in ["query", "title", "brand", "gender", "age_group", "attributes"]:
        enriched[column] = enriched[column].fillna("").astype("string").map(normalize_text)
    enriched["category"] = enriched["category"].fillna("").astype("string").map(normalize_category_path)
    return enriched


def predict_submission(
    model: HistGradientBoostingClassifier,
    threshold: float,
    tfidf_index: TfidfSimilarityIndex,
    query_vectors: dict[str, np.ndarray],
    terms: pd.DataFrame,
    items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    submission_pairs = pd.read_csv(
        locate_data_file("submission_pairs.csv"),
        usecols=["id", "term_id", "item_id"],
        dtype={"id": "string", "term_id": "string", "item_id": "string"},
    )
    if not USE_FULL_SUBMISSION and len(submission_pairs) > SUBMISSION_SAMPLE_SIZE:
        submission_pairs = submission_pairs.sample(
            n=SUBMISSION_SAMPLE_SIZE, random_state=RANDOM_STATE
        ).reset_index(drop=True)
        print(f"Submission örneklemi: {len(submission_pairs):,} satır (USE_FULL_SUBMISSION=false)")

    missing_terms = set(submission_pairs["term_id"]) - set(query_vectors)
    if missing_terms:
        extra_terms = terms[terms["term_id"].isin(missing_terms)].copy()
        query_vectors.update(tfidf_index.transform_terms(extra_terms))

    predictions = []
    probabilities = []
    for start in range(0, len(submission_pairs), SUBMISSION_CHUNK_SIZE):
        chunk = submission_pairs.iloc[start: start + SUBMISSION_CHUNK_SIZE].copy()
        enriched = enrich_pairs(chunk, terms, items)
        features = build_feature_matrix(enriched, tfidf_index, query_vectors)
        proba = model.predict_proba(features)[:, 1]
        pred = (proba >= threshold).astype(int)
        predictions.append(pd.DataFrame({"id": chunk["id"], "prediction": pred}))
        probabilities.append(pd.DataFrame({
            "id": chunk["id"].astype("string"),
            "term_id": chunk["term_id"].astype("string"),
            "item_id": chunk["item_id"].astype("string"),
            "probability": proba.astype(np.float32),
        }))
        print(f"  Tahmin: {start + len(chunk):,}/{len(submission_pairs):,}")

    return pd.concat(predictions, ignore_index=True), pd.concat(probabilities, ignore_index=True)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    write_config(EXPERIMENT_DIR / "config.json", {
        "experiment_id": EXPERIMENT_ID, "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS, "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_item_pool_size": TFIDF_ITEM_POOL_SIZE, "train_path": TRAIN_PATH,
    })

    print("=== Eğitim verisi ===")
    train = load_training_frame()
    print(f"Satır: {len(train):,} | Pozitif: {train['label'].sum():,} | Term: {train['term_id'].nunique():,}")

    print("\n=== Katalog ===")
    items = load_items_catalog()
    terms = load_terms()
    print(f"items: {len(items):,} | terms: {len(terms):,}")

    print("\n=== TF-IDF index ===")
    rng = np.random.default_rng(RANDOM_STATE)
    if len(items) <= TFIDF_ITEM_POOL_SIZE:
        item_pool = items
    else:
        positions = rng.choice(len(items), size=TFIDF_ITEM_POOL_SIZE, replace=False)
        item_pool = items.iloc[positions].reset_index(drop=True)
    print(f"TF-IDF havuzu: {len(item_pool):,} ürün")

    tfidf_index = TfidfSimilarityIndex(items)
    tfidf_index.fit(item_pool)

    train_terms = terms[terms["term_id"].isin(train["term_id"].unique())]
    query_vectors = tfidf_index.transform_terms(train_terms)

    print("\n=== Feature matrix ===")
    features = build_feature_matrix(train, tfidf_index, query_vectors)
    labels = train["label"].to_numpy()
    groups = train["term_id"].to_numpy()
    weights = train["sample_weight"].to_numpy()
    stable_term_folds(train["term_id"], N_SPLITS, RANDOM_STATE).to_parquet(FOLD_PATH, index=False)
    print(f"Feature shape: {features.shape}")

    print("\n=== Cross-validation ===")
    model, threshold, cv_report = cross_validate(features, labels, groups, weights)
    cv_report.to_csv(CV_REPORT_PATH, index=False)
    print(f"CV raporu: {CV_REPORT_PATH}")

    print("\n=== Kayıt ===")
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "feature_names": FEATURE_NAMES,
            "tfidf_vectorizer": tfidf_index.vectorizer,
            "item_pool_ids": item_pool["item_id"].tolist(),
        },
        MODEL_PATH,
    )
    THRESHOLD_PATH.write_text(f"{threshold:.4f}\n")
    print(f"Model: {MODEL_PATH}")
    print(f"Threshold: {threshold:.4f}")

    print("\n=== Submission ===")
    submission, probabilities = predict_submission(model, threshold, tfidf_index, query_vectors, terms, items)
    validate_probability_frame(probabilities, expected_rows=len(submission))
    probabilities.to_parquet(PROBABILITY_PATH, index=False)
    # P0: predict_proba çıktısını sabit konuma sade (id, prob) şemasıyla yaz.
    # Submission üretimini bozmaz; make_submission_from_probs.py bunu tüketir.
    baseline_probabilities = (
        probabilities[["id", "probability"]]
        .rename(columns={"probability": "prob"})
        .reset_index(drop=True)
    )
    BASELINE_PROBABILITIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline_probabilities.to_parquet(BASELINE_PROBABILITIES_PATH, index=False)
    print(f"Baseline olasılıkları: {BASELINE_PROBABILITIES_PATH} ({len(baseline_probabilities):,} satır, kolonlar: id, prob)")
    if USE_FULL_SUBMISSION:
        sample_submission = pd.read_csv(locate_data_file("sample_submission.csv"), dtype={"id": "string", "prediction": "int8"})
        validate_submission(submission, sample_submission)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission: {SUBMISSION_PATH} ({len(submission):,} satır)")
    print(submission["prediction"].value_counts())
    print(f"Probabilities: {PROBABILITY_PATH}")
    print("\nTamamlandı.")


if __name__ == "__main__":
    main()
