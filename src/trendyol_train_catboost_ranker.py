#!/usr/bin/env python3
"""Query-grouped CatBoost ranker for test-like candidate lists.

This is a separate ranking experiment. It does not overwrite the cross-encoder
or classical artifacts. The model is trained on known positives plus
confidence-weighted retrieval candidates and writes raw/probability scores;
the user chooses the final Macro-F1 threshold or ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_utils import experiment_dir, stable_term_folds, validate_probability_frame, validate_submission, write_config  # noqa: E402
from text_features import (  # noqa: E402
    build_lexical_features, category_parts, make_item_text,
    normalize_category_path, normalize_text,
)

ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]


def require_catboost():
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as exc:
        raise RuntimeError("CatBoost gerekli: pip install -r requirements-gpu.txt") from exc
    return CatBoostRanker, Pool


def normalize_items(items: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    items["category"] = items["category"].fillna("").map(normalize_category_path)
    for column in ["title", "brand", "gender", "age_group", "attributes"]:
        items[column] = items[column].fillna("").map(normalize_text)
    parts = items["category"].map(category_parts)
    items["top_category"] = parts.map(lambda p: p["top_category"])
    items["leaf_category"] = parts.map(lambda p: p["leaf_category"])
    return items


def build_pair_features(pairs: pd.DataFrame, items: pd.DataFrame, embedding_path: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    item_fields = ["title", "category", "brand", "gender", "age_group", "attributes", "top_category", "leaf_category"]
    missing_fields = [field for field in item_fields if field not in pairs.columns]
    frame = pairs.merge(
        items[["item_id", *missing_fields]],
        on="item_id", how="left", validate="many_to_one",
    ) if missing_fields else pairs.copy()
    lexical = build_lexical_features(frame)
    result = lexical.reset_index(drop=True)
    meta = pd.DataFrame(index=result.index)
    rank_raw = frame["retrieval_rank"] if "retrieval_rank" in frame else pd.Series(0, index=frame.index)
    score_raw = frame["retrieval_score"] if "retrieval_score" in frame else pd.Series(0, index=frame.index)
    rank = pd.to_numeric(rank_raw, errors="coerce").fillna(0).astype(float)
    retrieval_score = pd.to_numeric(score_raw, errors="coerce").fillna(0).astype(float)
    meta["retrieval_rank"] = rank
    meta["retrieval_rank_pct"] = np.where(rank.gt(0), 1.0 / rank, 0.0)
    meta["retrieval_score"] = retrieval_score
    meta["query_char_count"] = frame["query"].fillna("").str.len().astype(float)
    meta["title_char_count"] = frame["title"].fillna("").str.len().astype(float)
    meta["attribute_char_count"] = frame["attributes"].fillna("").str.len().astype(float)
    meta["top_category_candidate_count"] = frame.groupby(["term_id", "top_category"], sort=False)["item_id"].transform("size").astype(float)
    meta["leaf_category_candidate_count"] = frame.groupby(["term_id", "leaf_category"], sort=False)["item_id"].transform("size").astype(float)
    for source in ["title", "item_text", "forced_positive", "char", "word"]:
        if "retrieval_source" in frame:
            meta[f"source_{source}"] = frame["retrieval_source"].eq(source).astype(np.float32)
    result = pd.concat([result, meta.reset_index(drop=True)], axis=1)
    if embedding_path and embedding_path.exists():
        embedding = pd.read_parquet(embedding_path)
        result = result.join(
            frame[["term_id", "item_id"]].astype(str).merge(
                embedding, on=["term_id", "item_id"], how="left", validate="one_to_one"
            ).drop(columns=["term_id", "item_id"]).reset_index(drop=True)
        )
    numeric = [column for column in result.columns if pd.api.types.is_numeric_dtype(result[column])]
    result[numeric] = result[numeric].replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
    return result[numeric], numeric


def group_codes(frame: pd.DataFrame) -> np.ndarray:
    return pd.factorize(frame["term_id"].astype(str), sort=True)[0].astype(np.int64)


def best_macro_f1_threshold(labels: np.ndarray, raw_score: np.ndarray) -> tuple[float, float]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(raw_score, -30, 30)))
    best = (0.5, -1.0)
    for threshold in np.arange(0.05, 0.951, 0.005):
        score = f1_score(labels, probability >= threshold, average="macro")
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def load_pairs(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype="string")


def train(args, out: Path):
    CatBoostRanker, Pool = require_catboost()
    train = load_pairs(args.train)
    if "sample_weight" not in train:
        rank_raw = train["retrieval_rank"] if "retrieval_rank" in train else pd.Series(0, index=train.index)
        rank = pd.to_numeric(rank_raw, errors="coerce").fillna(0)
        train["sample_weight"] = np.where(
            train["label"].eq(1), 1.0,
            np.where(rank.le(10), 0.15, np.where(rank.le(50), 0.50, 0.85)),
        ).astype(np.float32)
    items = normalize_items(pd.read_csv(args.data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype="string"))
    train["query"] = train["query"].fillna("").map(normalize_text)
    features, feature_names = build_pair_features(train, items, args.embedding_features)
    folds = stable_term_folds(train["term_id"], args.n_splits, args.seed).set_index("term_id")["fold"]
    fold_id = train["term_id"].map(folds).to_numpy()
    reports = []
    oof_raw = np.full(len(train), np.nan, dtype=np.float32)
    final_model = None
    for fold in range(args.n_splits):
        train_idx = np.flatnonzero(fold_id != fold)
        valid_idx = np.flatnonzero(fold_id == fold)
        # CatBoost ranking requires rows of each group to be contiguous.
        train_idx = train_idx[np.argsort(train.iloc[train_idx]["term_id"].astype(str).to_numpy(), kind="stable")]
        valid_idx = valid_idx[np.argsort(train.iloc[valid_idx]["term_id"].astype(str).to_numpy(), kind="stable")]
        train_groups = group_codes(train.iloc[train_idx])
        valid_groups = group_codes(train.iloc[valid_idx])
        train_pool = Pool(features.iloc[train_idx], label=train.iloc[train_idx]["label"],
                          group_id=train_groups, weight=train.iloc[train_idx].get("sample_weight"))
        valid_pool = Pool(features.iloc[valid_idx], label=train.iloc[valid_idx]["label"],
                          group_id=valid_groups, weight=train.iloc[valid_idx].get("sample_weight"))
        model = CatBoostRanker(
            loss_function=args.loss_function, eval_metric="NDCG:top=20",
            iterations=args.iterations, depth=args.depth,
            learning_rate=args.learning_rate, l2_leaf_reg=args.l2_leaf_reg,
            random_seed=args.seed + fold, verbose=args.verbose,
            task_type=args.task_type, devices=args.devices,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        valid_score = model.predict(features.iloc[valid_idx])
        oof_raw[valid_idx] = valid_score
        threshold, macro_f1 = best_macro_f1_threshold(train.iloc[valid_idx]["label"].to_numpy(), valid_score)
        reports.append({
            "fold": fold, "rows": len(valid_idx),
            "valid_terms": int(train.iloc[valid_idx]["term_id"].nunique()),
            "mean_rank_score": float(valid_score.mean()),
            "macro_f1": macro_f1, "threshold": threshold,
        })

    final_model = CatBoostRanker(
        loss_function=args.loss_function, eval_metric="NDCG:top=20",
        iterations=args.iterations, depth=args.depth,
        learning_rate=args.learning_rate, l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.seed, verbose=args.verbose,
        task_type=args.task_type, devices=args.devices,
    )
    full_groups = group_codes(train.sort_values("term_id", kind="stable"))
    ordered = train.sort_values("term_id", kind="stable").index.to_numpy()
    full_pool = Pool(features.iloc[ordered], label=train.iloc[ordered]["label"], group_id=full_groups, weight=train.iloc[ordered].get("sample_weight"))
    final_model.fit(full_pool)
    threshold, oof_macro_f1 = best_macro_f1_threshold(train["label"].to_numpy(), oof_raw)
    reports.append({"fold": "oof", "rows": len(train), "valid_terms": int(train["term_id"].nunique()), "mean_rank_score": float(np.nanmean(oof_raw)), "macro_f1": oof_macro_f1, "threshold": threshold})
    final_model.save_model(str(out / "catboost_ranker.cbm"))
    joblib.dump({"feature_names": feature_names, "args": vars(args), "threshold": threshold}, out / "ranker_meta.joblib")
    pd.DataFrame(reports).to_csv(out / "cv_report.csv", index=False)
    pd.DataFrame({
        "term_id": train["term_id"].astype(str), "item_id": train["item_id"].astype(str),
        "label": train["label"].astype(np.int8), "raw_score": oof_raw,
        "probability": 1.0 / (1.0 + np.exp(-np.clip(oof_raw, -30, 30))),
    }).to_parquet(out / "oof_probabilities.parquet", index=False)
    return final_model, feature_names, items


def infer(args, out: Path, model=None, feature_names=None, items=None):
    CatBoostRanker, _ = require_catboost()
    if model is None:
        model = CatBoostRanker()
        model.load_model(str(args.model_path or out / "catboost_ranker.cbm"))
        meta = joblib.load(args.meta_path or out / "ranker_meta.joblib")
        feature_names = meta["feature_names"]
    if items is None:
        items = normalize_items(pd.read_csv(args.data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype="string"))
    terms = pd.read_csv(args.data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string")
    pairs = pd.read_csv(args.data_dir / "submission_pairs.csv", dtype="string")
    if args.test_retrieval:
        retrieval = pd.read_parquet(args.test_retrieval)
        pairs = pairs.merge(
            retrieval[["term_id", "item_id", "retrieval_rank", "retrieval_score", "retrieval_source"]],
            on=["term_id", "item_id"], how="left", validate="one_to_one",
        )
    pairs = pairs.merge(terms, on="term_id", validate="many_to_one")
    features, _ = build_pair_features(pairs, items, args.embedding_features_test)
    for name in feature_names:
        if name not in features:
            features[name] = 0.0
    raw = np.asarray(model.predict(features[feature_names]), dtype=np.float32)
    probability = (1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))).astype(np.float32)
    scores = pairs[["id", "term_id", "item_id"]].copy()
    scores["raw_score"] = raw
    scores["probability"] = probability
    validate_probability_frame(scores, len(pairs))
    scores.to_parquet(out / "probabilities.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "train-infer"], default="train")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--train", type=Path, required=False)
    parser.add_argument("--experiment-id", default="catboost_ranker_v1")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--meta-path", type=Path)
    parser.add_argument("--embedding-features", type=Path)
    parser.add_argument("--embedding-features-test", type=Path)
    parser.add_argument("--test-retrieval", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--loss-function", choices=["YetiRankPairwise", "PairLogit", "YetiRank"], default="YetiRankPairwise")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode in {"train", "train-infer"} and not args.train:
        parser.error("--train eğitim modunda zorunlu")
    out = experiment_dir(args.artifacts_dir, args.experiment_id)
    write_config(out / "config.json", vars(args))
    trained = None
    if args.mode in {"train", "train-infer"}:
        trained = train(args, out)
    if args.mode in {"infer", "train-infer"}:
        infer(args, out, *(trained or (None, None, None)))


if __name__ == "__main__":
    main()
