#!/usr/bin/env python3
"""Leakage-resistant split manifests, adversarial validation and OOF calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from trendyol_v3_core import normalize_text, set_global_seed


SplitKind = Literal["group", "semantic", "test_like"]
CalibrationKind = Literal["none", "temperature", "platt", "isotonic", "beta"]


@dataclass(frozen=True)
class SplitConfig:
    """Immutable query-level split configuration."""

    kind: SplitKind = "group"
    n_splits: int = 5
    seed: int = 42
    semantic_clusters: int = 192
    test_like_fraction: float = 0.15
    item_policy: Literal["audit", "purge_item", "purge_family"] = "audit"

    def validate(self) -> None:
        """Validate configuration bounds before any split is created."""

        if self.n_splits < 2:
            raise ValueError("n_splits en az 2 olmalıdır")
        if self.semantic_clusters < self.n_splits:
            raise ValueError("semantic_clusters en az n_splits olmalıdır")
        if not 0.01 <= self.test_like_fraction <= 0.5:
            raise ValueError("test_like_fraction [0.01, 0.5] aralığında olmalıdır")


@dataclass(frozen=True)
class CandidateProfile:
    """Empirical test candidate-count distribution."""

    counts: tuple[int, ...]
    median: float
    p95: float
    maximum: int
    source_hash: str

    @classmethod
    def fit(cls, submission_pairs: pd.DataFrame) -> "CandidateProfile":
        """Fit the profile using only observable submission candidate groups."""

        if "term_id" not in submission_pairs:
            raise ValueError("Candidate profile için term_id gerekli")
        counts = submission_pairs.groupby("term_id", sort=False).size().astype(int)
        if counts.empty or counts.le(0).any():
            raise ValueError("Aday sayısı dağılımı boş veya geçersiz")
        ordered = tuple(sorted(int(value) for value in counts))
        digest = hashlib.sha256(np.asarray(ordered, dtype=np.int32).tobytes()).hexdigest()
        return cls(
            counts=ordered,
            median=float(counts.median()),
            p95=float(counts.quantile(0.95)),
            maximum=int(counts.max()),
            source_hash=digest,
        )

    def deterministic_count(self, term_id: object, seed: int) -> int:
        """Draw one reproducible empirical count without consuming global RNG state."""

        if not self.counts:
            raise ValueError("CandidateProfile.counts boş")
        digest = hashlib.blake2b(f"{seed}:{term_id}".encode(), digest_size=8).digest()
        return self.counts[int.from_bytes(digest, "little") % len(self.counts)]


@dataclass
class SplitManifest:
    """A validated term-level fold manifest and its immutable metadata."""

    frame: pd.DataFrame
    config: SplitConfig
    source_hash: str

    def validate(self) -> None:
        """Assert term uniqueness, fold bounds and group integrity."""

        required = {"term_id", "query", "canonical_query_group", "fold"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Split manifest kolonları eksik: {sorted(missing)}")
        if self.frame["term_id"].isna().any() or self.frame["term_id"].duplicated().any():
            raise ValueError("Manifest term_id boş veya duplicate")
        if self.config.kind == "test_like":
            valid_folds = {-1, 0}
        else:
            valid_folds = set(range(self.config.n_splits))
        actual = set(pd.to_numeric(self.frame["fold"], errors="raise").astype(int))
        if not actual.issubset(valid_folds):
            raise ValueError(f"Manifest fold değerleri geçersiz: {sorted(actual - valid_folds)}")
        if self.config.kind != "test_like" and actual != valid_folds:
            raise ValueError(f"Bütün fold'lar dolu olmalı; bulunan={sorted(actual)}")
        group_columns = ["canonical_query_group"]
        if self.config.kind == "semantic":
            if "semantic_cluster" not in self.frame:
                raise ValueError("Semantic manifest semantic_cluster içermeli")
            group_columns.append("semantic_cluster")
        for column in group_columns:
            if self.frame.groupby(column, dropna=False)["fold"].nunique().max() != 1:
                raise ValueError(f"{column} birden fazla fold'a sızdı")

    def scope(self, fold: int) -> tuple[frozenset[str], frozenset[str]]:
        """Return training and validation term IDs for one outer fold."""

        self.validate()
        if self.config.kind == "test_like" and fold != 0:
            raise ValueError("test_like manifest yalnız fold=0 içerir")
        valid = frozenset(self.frame.loc[self.frame["fold"].eq(fold), "term_id"].astype(str))
        train = frozenset(self.frame.loc[self.frame["fold"].ne(fold), "term_id"].astype(str))
        if not valid or not train or valid.intersection(train):
            raise RuntimeError("Boş veya kesişen train/validation scope")
        return train, valid

    def save(self, output_dir: Path) -> None:
        """Persist the manifest and metadata atomically enough for Kaggle resumes."""

        self.validate()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(output_dir / f"{self.config.kind}_manifest.parquet", index=False)
        payload = {"config": asdict(self.config), "source_hash": self.source_hash, "rows": len(self.frame)}
        (output_dir / f"{self.config.kind}_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


@dataclass
class AdversarialValidationResult:
    """OOF domain-classifier scores and interpretable diagnostics."""

    term_scores: pd.DataFrame
    fold_metrics: pd.DataFrame
    top_features: pd.DataFrame
    overall_auc: float

    def save(self, output_dir: Path) -> None:
        """Persist all adversarial-validation artifacts."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.term_scores.to_parquet(output_dir / "adversarial_term_scores.parquet", index=False)
        self.fold_metrics.to_csv(output_dir / "adversarial_fold_metrics.csv", index=False)
        self.top_features.to_csv(output_dir / "adversarial_top_features.csv", index=False)
        (output_dir / "adversarial_summary.json").write_text(
            json.dumps({"overall_auc": self.overall_auc}, indent=2) + "\n", encoding="utf-8"
        )


class Calibrator(Protocol):
    """Common API implemented by every OOF calibrator."""

    def fit(self, raw_logit: np.ndarray, label: np.ndarray) -> "Calibrator":
        """Fit the calibrator."""

    def predict_proba(self, raw_logit: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities."""


def _frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash ordered canonical columns for lineage checks."""

    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Hash kolonları eksik: {sorted(missing)}")
    values = pd.util.hash_pandas_object(frame[list(columns)].astype("string"), index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def _canonical_query(value: object) -> str:
    """Canonicalize a query for exact/paraphrase leakage grouping."""

    return normalize_text(value, ascii_fold=True)


def _balanced_group_assignment(
    frame: pd.DataFrame,
    *,
    group_col: str,
    weight_col: str,
    n_splits: int,
    seed: int,
) -> pd.Series:
    """Greedily balance whole groups by positive-pair load."""

    if frame[group_col].isna().any():
        raise ValueError(f"{group_col} boş değer içeriyor")
    loads = frame.groupby(group_col, sort=False)[weight_col].sum().astype(float)
    if len(loads) < n_splits:
        raise ValueError(f"Grup sayısı={len(loads)}, n_splits={n_splits}")
    tie = {
        group: int.from_bytes(hashlib.blake2b(f"{seed}:{group}".encode(), digest_size=8).digest(), "little")
        for group in loads.index
    }
    ordered = sorted(loads.index, key=lambda group: (-loads.loc[group], tie[group]))
    fold_loads = np.zeros(n_splits, dtype=np.float64)
    fold_groups = np.zeros(n_splits, dtype=np.int64)
    assignment: dict[Any, int] = {}
    for group in ordered:
        candidates = np.lexsort((np.arange(n_splits), fold_groups, fold_loads))
        fold = int(candidates[0])
        assignment[group] = fold
        fold_loads[fold] += float(loads.loc[group])
        fold_groups[fold] += 1
    return frame[group_col].map(assignment).astype("int16")


def _term_frame(positives: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
    """Build one row per positive training term with pair weights."""

    required_pos = {"term_id", "item_id"}
    required_terms = {"term_id", "query"}
    if missing := required_pos - set(positives.columns):
        raise ValueError(f"Pozitif kolonları eksik: {sorted(missing)}")
    if missing := required_terms - set(terms.columns):
        raise ValueError(f"Term kolonları eksik: {sorted(missing)}")
    if positives.duplicated(["term_id", "item_id"]).any():
        raise ValueError("Ham pozitiflerde duplicate pair var")
    counts = positives.groupby("term_id", sort=False).size().rename("positive_count")
    frame = terms[["term_id", "query"]].drop_duplicates("term_id").merge(
        counts, left_on="term_id", right_index=True, how="inner", validate="one_to_one"
    )
    if len(frame) != positives["term_id"].nunique():
        raise ValueError("Bazı pozitif term_id değerleri terms tablosunda bulunamadı")
    frame["canonical_query_group"] = frame["query"].map(_canonical_query)
    if frame["canonical_query_group"].eq("").any():
        raise ValueError("Boş normalize query bulundu")
    return frame.reset_index(drop=True)


def build_group_manifest(
    positives: pd.DataFrame,
    terms: pd.DataFrame,
    config: SplitConfig,
) -> SplitManifest:
    """Build Validation A with canonical-query-strict balanced groups."""

    config.validate()
    if config.kind != "group":
        raise ValueError("build_group_manifest için config.kind='group' olmalıdır")
    set_global_seed(config.seed)
    frame = _term_frame(positives, terms)
    frame["fold"] = _balanced_group_assignment(
        frame, group_col="canonical_query_group", weight_col="positive_count",
        n_splits=config.n_splits, seed=config.seed,
    )
    manifest = SplitManifest(frame, config, _frame_hash(frame, ["term_id", "query", "positive_count"]))
    manifest.validate()
    return manifest


def encode_query_embeddings(
    term_frame: pd.DataFrame,
    cache_dir: Path,
    *,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 42,
    instruction: str = (
        "Given a Turkish e-commerce search query, retrieve products that satisfy "
        "the shopper's intended product type and requested attributes"
    ),
) -> np.ndarray:
    """Create or reuse frozen query embeddings with an ordered-input fingerprint."""

    required = {"term_id", "query"}
    if missing := required - set(term_frame.columns):
        raise ValueError(f"Embedding kolonları eksik: {sorted(missing)}")
    if batch_size < 1:
        raise ValueError("batch_size pozitif olmalıdır")
    set_global_seed(seed)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_hash = _frame_hash(term_frame, ["term_id", "query"])
    model_key = hashlib.blake2b(model_name.encode(), digest_size=8).hexdigest()
    array_path = cache_dir / f"query_embeddings_{model_key}.npy"
    meta_path = cache_dir / f"query_embeddings_{model_key}.json"
    if array_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("data_hash") == data_hash and meta.get("model_name") == model_name:
            cached = np.load(array_path, mmap_mode="r")
            if cached.shape[0] != len(term_frame):
                raise RuntimeError("Embedding cache satır sayısı metadata ile uyuşmuyor")
            return np.asarray(cached, dtype=np.float32)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Embedding için sentence-transformers>=2.7 gereklidir") from exc
    kwargs: dict[str, Any] = {"device": device}
    model = SentenceTransformer(model_name, **kwargs)
    prompt = f"Instruct: {instruction}\nQuery:"
    texts = term_frame["query"].fillna("").astype(str).tolist()
    embeddings = model.encode(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    if embeddings.shape[0] != len(term_frame) or not np.isfinite(embeddings).all():
        raise RuntimeError("Üretilen query embeddingleri geçersiz")
    np.save(array_path, embeddings.astype(np.float16))
    meta_path.write_text(
        json.dumps(
            {"data_hash": data_hash, "model_name": model_name, "instruction": instruction,
             "rows": len(term_frame), "dimension": embeddings.shape[1]},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return embeddings


def build_semantic_manifest(
    positives: pd.DataFrame,
    terms: pd.DataFrame,
    embeddings: np.ndarray,
    config: SplitConfig,
) -> SplitManifest:
    """Build Validation B by holding complete frozen semantic clusters together."""

    config.validate()
    if config.kind != "semantic":
        raise ValueError("build_semantic_manifest için config.kind='semantic' olmalıdır")
    set_global_seed(config.seed)
    frame = _term_frame(positives, terms)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(frame):
        raise ValueError(f"Embedding shape={embeddings.shape}; beklenen satır={len(frame)}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding NaN/Inf içeriyor")
    cluster_count = min(config.semantic_clusters, len(frame))
    clusterer = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=config.seed,
        batch_size=min(4096, max(256, len(frame))),
        n_init=10,
        reassignment_ratio=0.0,
    )
    frame["semantic_cluster"] = clusterer.fit_predict(embeddings).astype("int32")
    # Canonical duplicates are forced into the same semantic group before balancing.
    canonical_min_cluster = frame.groupby("canonical_query_group")["semantic_cluster"].transform("min")
    frame["semantic_cluster"] = canonical_min_cluster.astype("int32")
    frame["fold"] = _balanced_group_assignment(
        frame, group_col="semantic_cluster", weight_col="positive_count",
        n_splits=config.n_splits, seed=config.seed,
    )
    manifest = SplitManifest(frame, config, _frame_hash(frame, ["term_id", "query", "positive_count"]))
    manifest.validate()
    return manifest


def _numeric_query_features(queries: pd.Series) -> tuple[sparse.csr_matrix, list[str]]:
    """Create symmetric, observable query-only domain features."""

    normalized = queries.fillna("").map(normalize_text)
    values = np.column_stack([
        normalized.str.len().to_numpy(),
        normalized.str.split().str.len().to_numpy(),
        normalized.str.count(r"\d").to_numpy(),
        normalized.str.count(r"[çğıöşü]").to_numpy(),
        normalized.str.count(r"\b(?:gb|tb|ml|kg|cm|mm|mah)\b").to_numpy(),
        normalized.str.count(r"[^a-zçğıöşü0-9 ]").to_numpy(),
    ]).astype(np.float32)
    scale = np.maximum(values.std(axis=0, keepdims=True), 1.0)
    values = (values - values.mean(axis=0, keepdims=True)) / scale
    names = ["length_chars", "length_tokens", "digit_count", "turkish_char_count", "unit_count", "punctuation_count"]
    return sparse.csr_matrix(values), names


def run_adversarial_validation(
    train_terms: pd.DataFrame,
    test_terms: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    max_word_features: int = 40_000,
    max_char_features: int = 40_000,
) -> AdversarialValidationResult:
    """Fit a query-only train-vs-test classifier and return fully OOF domain scores."""

    for name, frame in (("train", train_terms), ("test", test_terms)):
        if not {"term_id", "query"}.issubset(frame.columns):
            raise ValueError(f"{name} terms term_id,query içermeli")
        if frame["term_id"].duplicated().any():
            raise ValueError(f"{name} terms duplicate term_id içeriyor")
    if set(train_terms["term_id"]).intersection(set(test_terms["term_id"])):
        raise ValueError("Adversarial validation train/test term_id kesişiyor")
    if n_splits < 2:
        raise ValueError("n_splits en az 2 olmalıdır")
    set_global_seed(seed)
    combined = pd.concat([
        train_terms[["term_id", "query"]].assign(domain_label=np.int8(0), source="train"),
        test_terms[["term_id", "query"]].assign(domain_label=np.int8(1), source="test"),
    ], ignore_index=True)
    text = combined["query"].fillna("").map(normalize_text)
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=max_word_features,
        sublinear_tf=True, dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=max_char_features,
        sublinear_tf=True, dtype=np.float32,
    )
    word_matrix = word.fit_transform(text)
    char_matrix = char.fit_transform(text)
    numeric_matrix, numeric_names = _numeric_query_features(text)
    matrix = sparse.hstack([word_matrix, char_matrix, numeric_matrix], format="csr", dtype=np.float32)
    names = (
        [f"word:{name}" for name in word.get_feature_names_out()]
        + [f"char:{name}" for name in char.get_feature_names_out()]
        + [f"numeric:{name}" for name in numeric_names]
    )
    labels = combined["domain_label"].to_numpy(dtype=np.int8)
    oof = np.full(len(combined), np.nan, dtype=np.float32)
    folds = np.full(len(combined), -1, dtype=np.int16)
    metrics: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(matrix, labels)):
        model = LogisticRegression(
            C=1.0, max_iter=300, class_weight="balanced", solver="liblinear", random_state=seed + fold,
        )
        model.fit(matrix[train_idx], labels[train_idx])
        probability = model.predict_proba(matrix[valid_idx])[:, 1]
        oof[valid_idx] = probability.astype(np.float32)
        folds[valid_idx] = fold
        metrics.append({
            "fold": fold,
            "auc": float(roc_auc_score(labels[valid_idx], probability)),
            "rows": int(len(valid_idx)),
        })
    if np.isnan(oof).any() or (folds < 0).any():
        raise RuntimeError("Adversarial OOF coverage eksik")
    final_model = LogisticRegression(
        C=1.0, max_iter=500, class_weight="balanced", solver="liblinear", random_state=seed,
    ).fit(matrix, labels)
    coefficients = final_model.coef_.ravel()
    order = np.argsort(np.abs(coefficients))[::-1][:200]
    top = pd.DataFrame({
        "feature": np.asarray(names, dtype=object)[order],
        "coefficient": coefficients[order].astype(np.float32),
        "direction": np.where(coefficients[order] > 0, "test", "train"),
    })
    scores = combined[["term_id", "query", "source", "domain_label"]].copy()
    scores["adversarial_oof_score"] = oof
    scores["adversarial_fold"] = folds
    return AdversarialValidationResult(
        term_scores=scores,
        fold_metrics=pd.DataFrame(metrics),
        top_features=top,
        overall_auc=float(roc_auc_score(labels, oof)),
    )


def build_test_like_manifest(
    positives: pd.DataFrame,
    terms: pd.DataFrame,
    adversarial_scores: pd.DataFrame,
    config: SplitConfig,
    *,
    semantic_clusters: pd.Series | None = None,
) -> SplitManifest:
    """Build Validation C from whole high-domain-score semantic/canonical groups."""

    config.validate()
    if config.kind != "test_like":
        raise ValueError("build_test_like_manifest için config.kind='test_like' olmalıdır")
    frame = _term_frame(positives, terms)
    required = {"term_id", "adversarial_oof_score"}
    if missing := required - set(adversarial_scores.columns):
        raise ValueError(f"Adversarial score kolonları eksik: {sorted(missing)}")
    train_scores = adversarial_scores.copy()
    if "source" in train_scores:
        train_scores = train_scores[train_scores["source"].eq("train")]
    frame = frame.merge(
        train_scores[["term_id", "adversarial_oof_score"]],
        on="term_id", how="left", validate="one_to_one",
    )
    if frame["adversarial_oof_score"].isna().any():
        raise ValueError("Bazı training term'leri için OOF adversarial score eksik")
    if semantic_clusters is not None:
        if len(semantic_clusters) != len(frame):
            raise ValueError("semantic_clusters uzunluğu term frame ile aynı olmalı")
        frame["semantic_cluster"] = np.asarray(semantic_clusters, dtype=np.int32)
        group_col = "semantic_cluster"
    else:
        group_col = "canonical_query_group"
    groups = frame.groupby(group_col, sort=False).agg(
        score=("adversarial_oof_score", "mean"),
        positive_count=("positive_count", "sum"),
        term_count=("term_id", "size"),
    ).reset_index()
    groups = groups.sort_values(["score", "positive_count", group_col], ascending=[False, False, True])
    target = max(1, int(math.ceil(frame["positive_count"].sum() * config.test_like_fraction)))
    groups["cumulative"] = groups["positive_count"].cumsum()
    selected = groups.loc[groups["cumulative"].shift(fill_value=0).lt(target), group_col]
    frame["is_test_like"] = frame[group_col].isin(set(selected))
    frame["fold"] = np.where(frame["is_test_like"], 0, -1).astype("int16")
    manifest = SplitManifest(frame, config, _frame_hash(frame, ["term_id", "query", "positive_count"]))
    manifest.validate()
    return manifest


def audit_item_overlap(
    manifest: SplitManifest,
    positives: pd.DataFrame,
    *,
    family_map: pd.Series | None = None,
) -> pd.DataFrame:
    """Measure positive-item and optional family overlap in every outer fold."""

    manifest.validate()
    if not {"term_id", "item_id"}.issubset(positives.columns):
        raise ValueError("positives term_id,item_id içermeli")
    reports: list[dict[str, Any]] = []
    folds = [0] if manifest.config.kind == "test_like" else list(range(manifest.config.n_splits))
    term_fold = manifest.frame.set_index("term_id")["fold"]
    pair_frame = positives[["term_id", "item_id"]].copy()
    pair_frame["fold"] = pair_frame["term_id"].map(term_fold)
    if pair_frame["fold"].isna().any():
        raise ValueError("Manifestte bulunmayan positive term_id var")
    if family_map is not None:
        pair_frame["family_id"] = pair_frame["item_id"].map(family_map)
    for fold in folds:
        valid = pair_frame[pair_frame["fold"].eq(fold)]
        train = pair_frame[pair_frame["fold"].ne(fold)]
        item_overlap = set(valid["item_id"]).intersection(train["item_id"])
        row: dict[str, Any] = {
            "fold": fold,
            "train_terms": int(train["term_id"].nunique()),
            "valid_terms": int(valid["term_id"].nunique()),
            "train_pairs": int(len(train)),
            "valid_pairs": int(len(valid)),
            "overlap_items": int(len(item_overlap)),
        }
        if family_map is not None:
            family_overlap = set(valid["family_id"].dropna()).intersection(train["family_id"].dropna())
            row["overlap_families"] = int(len(family_overlap))
        reports.append(row)
    return pd.DataFrame(reports)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid returning float64 probabilities."""

    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class IdentityCalibrator:
    """Map raw logits through sigmoid without learned calibration."""

    def fit(self, raw_logit: np.ndarray, label: np.ndarray) -> "IdentityCalibrator":
        """Validate arrays and retain no parameters."""

        _validate_calibration_arrays(raw_logit, label)
        return self

    def predict_proba(self, raw_logit: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities."""

        return sigmoid(raw_logit)


class TemperatureCalibrator:
    """Positive scalar temperature fitted by held-out log loss."""

    def __init__(self) -> None:
        """Initialize with neutral temperature one."""

        self.temperature = 1.0

    def fit(self, raw_logit: np.ndarray, label: np.ndarray) -> "TemperatureCalibrator":
        """Fit log-temperature with a bounded scalar optimizer."""

        x, y = _validate_calibration_arrays(raw_logit, label)

        def objective(log_temperature: float) -> float:
            """Return calibration log loss at one log-temperature."""

            probability = sigmoid(x / math.exp(log_temperature))
            return float(log_loss(y, probability, labels=[0, 1]))

        result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
        if not result.success:
            raise RuntimeError(f"Temperature scaling başarısız: {result.message}")
        self.temperature = float(math.exp(result.x))
        return self

    def predict_proba(self, raw_logit: np.ndarray) -> np.ndarray:
        """Apply the learned temperature."""

        return sigmoid(np.asarray(raw_logit) / self.temperature)


class PlattCalibrator:
    """Logistic affine mapping from raw score to probability."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize a nearly unregularized deterministic logistic model."""

        self.seed = seed
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500, random_state=seed)

    def fit(self, raw_logit: np.ndarray, label: np.ndarray) -> "PlattCalibrator":
        """Fit affine logistic calibration."""

        x, y = _validate_calibration_arrays(raw_logit, label)
        self.model.fit(x.reshape(-1, 1), y)
        return self

    def predict_proba(self, raw_logit: np.ndarray) -> np.ndarray:
        """Return positive probabilities from the fitted logistic model."""

        return self.model.predict_proba(np.asarray(raw_logit).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Monotone non-parametric OOF score calibrator."""

    def __init__(self) -> None:
        """Initialize clipped out-of-range isotonic regression."""

        self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)

    def fit(self, raw_logit: np.ndarray, label: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic regression on raw logits."""

        x, y = _validate_calibration_arrays(raw_logit, label)
        self.model.fit(x, y)
        return self

    def predict_proba(self, raw_logit: np.ndarray) -> np.ndarray:
        """Return clipped monotone probabilities."""

        return np.asarray(self.model.predict(np.asarray(raw_logit)), dtype=np.float64)


class BetaCalibrator:
    """Beta calibration using log(p) and log(1-p) features."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the deterministic beta-calibration logistic layer."""

        self.seed = seed
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500, random_state=seed)

    @staticmethod
    def _features(raw_logit: np.ndarray) -> np.ndarray:
        """Transform raw logits into stable beta-calibration features."""

        probability = np.clip(sigmoid(raw_logit), 1e-6, 1 - 1e-6)
        return np.column_stack([np.log(probability), np.log1p(-probability)])

    def fit(self, raw_logit: np.ndarray, label: np.ndarray) -> "BetaCalibrator":
        """Fit beta calibration."""

        x, y = _validate_calibration_arrays(raw_logit, label)
        self.model.fit(self._features(x), y)
        return self

    def predict_proba(self, raw_logit: np.ndarray) -> np.ndarray:
        """Apply beta calibration."""

        return self.model.predict_proba(self._features(np.asarray(raw_logit)))[:, 1]


def _validate_calibration_arrays(
    raw_logit: np.ndarray,
    label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary calibration arrays and return canonical dtypes."""

    x = np.asarray(raw_logit, dtype=np.float64).reshape(-1)
    y = np.asarray(label, dtype=np.int8).reshape(-1)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Calibration score/label uzunlukları boş veya eşit değil")
    if not np.isfinite(x).all() or not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) < 2:
        raise ValueError("Calibration finite raw score ve iki binary sınıf gerektirir")
    return x, y


def make_calibrator(kind: CalibrationKind, seed: int = 42) -> Calibrator:
    """Construct a named calibrator."""

    factories: dict[str, Calibrator] = {
        "none": IdentityCalibrator(),
        "temperature": TemperatureCalibrator(),
        "platt": PlattCalibrator(seed),
        "isotonic": IsotonicCalibrator(),
        "beta": BetaCalibrator(seed),
    }
    if kind not in factories:
        raise ValueError(f"Bilinmeyen calibration method: {kind}")
    return factories[kind]


def classification_report_dict(
    label: np.ndarray,
    prediction: np.ndarray,
    *,
    probability: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict[str, float | int | None]:
    """Return all competition-relevant binary classification metrics."""

    y = np.asarray(label, dtype=np.int8)
    pred = np.asarray(prediction, dtype=np.int8)
    if len(y) != len(pred) or not set(np.unique(y)).issubset({0, 1}) or not set(np.unique(pred)).issubset({0, 1}):
        raise ValueError("Metric arrays eşit uzunlukta binary değerler olmalıdır")
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    report: dict[str, float | int | None] = {
        "rows": int(len(y)), "threshold": threshold,
        "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
        "class_0_precision": float(precision[0]), "class_0_recall": float(recall[0]),
        "class_0_f1": float(f1[0]), "class_0_support": int(support[0]),
        "class_1_precision": float(precision[1]), "class_1_recall": float(recall[1]),
        "class_1_f1": float(f1[1]), "class_1_support": int(support[1]),
        "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)), "accuracy": float(accuracy_score(y, pred)),
        "predicted_positive_rate": float(pred.mean()),
    }
    if probability is not None:
        prob = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
        if len(prob) != len(y) or not np.isfinite(prob).all():
            raise ValueError("Probability metric array geçersiz")
        report.update({
            "log_loss": float(log_loss(y, prob, labels=[0, 1])),
            "brier": float(brier_score_loss(y, prob)),
            "ece": float(expected_calibration_error(y, prob)),
        })
    return report


def expected_calibration_error(
    label: np.ndarray,
    probability: np.ndarray,
    bins: int = 15,
) -> float:
    """Compute equal-width expected calibration error."""

    if bins < 2:
        raise ValueError("ECE bins en az 2 olmalıdır")
    y = np.asarray(label, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            error += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(error)


def threshold_sweep(
    label: np.ndarray,
    probability: np.ndarray,
    *,
    grid_size: int = 1001,
) -> pd.DataFrame:
    """Evaluate Macro-F1 and both class F1 scores over an exact bounded grid."""

    y = np.asarray(label, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if len(y) != len(p) or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Threshold sweep arrays geçersiz")
    if grid_size < 3:
        raise ValueError("grid_size en az 3 olmalıdır")
    candidates = np.unique(np.concatenate([
        np.linspace(0.001, 0.999, grid_size),
        np.quantile(p, np.linspace(0.0, 1.0, min(grid_size, len(p))))
    ]))
    order = np.argsort(-p, kind="mergesort")
    sorted_probability = p[order]
    sorted_label = y[order].astype(np.int64)
    positive_prefix = np.concatenate([[0], np.cumsum(sorted_label, dtype=np.int64)])
    predicted_positive = np.searchsorted(-sorted_probability, -candidates, side="right")
    true_positive = positive_prefix[predicted_positive].astype(np.float64)
    false_positive = predicted_positive.astype(np.float64) - true_positive
    total_positive = float(sorted_label.sum())
    total_negative = float(len(sorted_label) - total_positive)
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive
    class_1_denominator = 2 * true_positive + false_positive + false_negative
    class_0_denominator = 2 * true_negative + false_negative + false_positive
    class_1_f1 = np.divide(
        2 * true_positive, class_1_denominator,
        out=np.zeros_like(true_positive), where=class_1_denominator > 0,
    )
    class_0_f1 = np.divide(
        2 * true_negative, class_0_denominator,
        out=np.zeros_like(true_negative), where=class_0_denominator > 0,
    )
    report = pd.DataFrame({
        "threshold": candidates.astype(np.float64),
        "class_0_f1": class_0_f1,
        "class_1_f1": class_1_f1,
        "macro_f1": (class_0_f1 + class_1_f1) / 2,
        "positive_rate": predicted_positive / len(y),
    })
    return report.sort_values(
        ["macro_f1", "threshold"], ascending=[False, True]
    ).reset_index(drop=True)


def select_threshold(label: np.ndarray, probability: np.ndarray, grid_size: int = 1001) -> float:
    """Select the lowest threshold among tied maximum Macro-F1 values."""

    report = threshold_sweep(label, probability, grid_size=grid_size)
    return float(report.iloc[0]["threshold"])


def validate_oof_frame(frame: pd.DataFrame) -> None:
    """Validate exact-once OOF coverage and mandatory raw-score lineage."""

    required = {"pair_uid", "term_id", "item_id", "label", "fold", "raw_logit"}
    if missing := required - set(frame.columns):
        raise ValueError(f"OOF kolonları eksik: {sorted(missing)}")
    if frame["pair_uid"].isna().any() or frame["pair_uid"].duplicated().any():
        raise ValueError("OOF pair_uid boş veya duplicate")
    if frame.duplicated(["term_id", "item_id"]).any():
        raise ValueError("OOF duplicate term-item çifti içeriyor")
    if frame[["term_id", "item_id", "fold", "raw_logit"]].isna().any().any():
        raise ValueError("OOF ana kolonlarında null var")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("OOF label yalnızca 0/1 olmalıdır")
    raw = pd.to_numeric(frame["raw_logit"], errors="coerce")
    if not np.isfinite(raw).all():
        raise ValueError("OOF raw_logit NaN/Inf içeriyor")


def cross_fit_calibration_and_threshold(
    oof: pd.DataFrame,
    *,
    methods: Sequence[CalibrationKind] = ("none", "temperature", "platt", "isotonic", "beta"),
    seed: int = 42,
    threshold_grid_size: int = 501,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit calibrators and thresholds without using the target fold labels."""

    validate_oof_frame(oof)
    working = oof.copy()
    working["fold"] = pd.to_numeric(working["fold"], errors="raise").astype(int)
    folds = sorted(working["fold"].unique())
    if len(folds) < 2:
        raise ValueError("Cross-fit calibration en az iki OOF fold gerektirir")
    outputs: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for method in methods:
        method_parts: list[pd.DataFrame] = []
        for fold in folds:
            fit_mask = working["fold"].ne(fold)
            apply_mask = working["fold"].eq(fold)
            fit = working.loc[fit_mask]
            apply = working.loc[apply_mask]
            calibrator = make_calibrator(method, seed + fold).fit(
                fit["raw_logit"].to_numpy(), fit["label"].to_numpy()
            )
            fit_probability = calibrator.predict_proba(fit["raw_logit"].to_numpy())
            threshold = select_threshold(fit["label"].to_numpy(), fit_probability, threshold_grid_size)
            probability = calibrator.predict_proba(apply["raw_logit"].to_numpy())
            part = apply[["pair_uid", "fold", "label"]].copy()
            part["calibration_method"] = method
            part["calibrated_probability"] = probability.astype(np.float32)
            part["selected_threshold"] = np.float32(threshold)
            part["prediction"] = (probability >= threshold).astype(np.int8)
            part["fit_folds"] = ",".join(str(value) for value in folds if value != fold)
            method_parts.append(part)
        crossfit = pd.concat(method_parts, ignore_index=True)
        if len(crossfit) != len(working) or crossfit["pair_uid"].duplicated().any():
            raise RuntimeError(f"{method} cross-fit OOF coverage bozuk")
        metrics = classification_report_dict(
            crossfit["label"].to_numpy(), crossfit["prediction"].to_numpy(),
            probability=crossfit["calibrated_probability"].to_numpy(), threshold=None,
        )
        metrics["calibration_method"] = method
        reports.append(metrics)
        outputs.append(crossfit)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(reports).sort_values(
        ["macro_f1", "log_loss"], ascending=[False, True]
    ).reset_index(drop=True)


def candidate_count_bucket(values: pd.Series) -> pd.Series:
    """Map observable query candidate counts to stable inference-time buckets."""

    numeric = pd.to_numeric(values, errors="raise")
    bins = [-np.inf, 50, 100, 150, 300, np.inf]
    labels = ["le50", "51_100", "101_150", "151_300", "gt300"]
    return pd.cut(numeric, bins=bins, labels=labels).astype("string")


def learn_bucket_thresholds(
    frame: pd.DataFrame,
    *,
    bucket_col: str,
    probability_col: str = "calibrated_probability",
    min_rows: int = 2_000,
    min_terms: int = 50,
    shrinkage: float = 5_000.0,
) -> tuple[float, dict[str, float]]:
    """Learn supported bucket thresholds with global shrinkage and fallback."""

    required = {bucket_col, probability_col, "label", "term_id"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Bucket threshold kolonları eksik: {sorted(missing)}")
    if min_rows < 1 or min_terms < 1 or shrinkage < 0:
        raise ValueError("Bucket support/shrinkage değerleri geçersiz")
    global_threshold = select_threshold(frame["label"].to_numpy(), frame[probability_col].to_numpy())
    thresholds: dict[str, float] = {}
    for bucket, group in frame.groupby(bucket_col, dropna=False):
        if len(group) < min_rows or group["term_id"].nunique() < min_terms:
            continue
        local = select_threshold(group["label"].to_numpy(), group[probability_col].to_numpy())
        weight = len(group) / (len(group) + shrinkage)
        thresholds[str(bucket)] = float(weight * local + (1 - weight) * global_threshold)
    return global_threshold, thresholds


def apply_bucket_thresholds(
    frame: pd.DataFrame,
    *,
    bucket_col: str,
    global_threshold: float,
    thresholds: dict[str, float],
    probability_col: str = "calibrated_probability",
) -> pd.DataFrame:
    """Apply bucket thresholds, falling back globally for unseen buckets."""

    required = {bucket_col, probability_col}
    if missing := required - set(frame.columns):
        raise ValueError(f"Bucket apply kolonları eksik: {sorted(missing)}")
    out = frame.copy()
    out["applied_threshold"] = out[bucket_col].astype(str).map(thresholds).fillna(global_threshold).astype("float32")
    out["prediction"] = (out[probability_col] >= out["applied_threshold"]).astype("int8")
    return out


def query_postprocess_predictions(
    frame: pd.DataFrame,
    *,
    probability_col: str,
    threshold: float,
    ensure_one: bool = False,
    top_k: int | None = None,
    contradiction_penalty: float = 0.0,
) -> np.ndarray:
    """Apply deterministic query-aware rules for OOF-only ablation."""

    frame = frame.reset_index(drop=True)
    required = {"term_id", probability_col}
    if missing := required - set(frame.columns):
        raise ValueError(f"Postprocess kolonları eksik: {sorted(missing)}")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k pozitif olmalıdır")
    score = pd.to_numeric(frame[probability_col], errors="raise").to_numpy(dtype=np.float64)
    if contradiction_penalty:
        if "contradiction_count" not in frame:
            raise ValueError("contradiction_penalty için contradiction_count gerekli")
        score = score - contradiction_penalty * frame["contradiction_count"].to_numpy(dtype=np.float64)
    prediction = (score >= threshold).astype(np.int8)
    working = pd.DataFrame({"term_id": frame["term_id"].astype(str), "score": score, "row": np.arange(len(frame))})
    if top_k is not None:
        prediction[:] = 0
        ranked = working.sort_values(["term_id", "score", "row"], ascending=[True, False, True])
        chosen = ranked.groupby("term_id", sort=False).head(top_k)["row"].to_numpy()
        prediction[chosen] = 1
    if ensure_one:
        positive_counts = pd.Series(prediction).groupby(working["term_id"], sort=False).transform("sum")
        missing_terms = set(working.loc[positive_counts.eq(0), "term_id"])
        if missing_terms:
            top = (
                working[working["term_id"].isin(missing_terms)]
                .sort_values(["term_id", "score", "row"], ascending=[True, False, True])
                .groupby("term_id", sort=False).head(1)
            )
            prediction[top["row"].to_numpy()] = 1
    return prediction


def error_analysis(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    prediction_col: str = "prediction",
    probability_col: str = "calibrated_probability",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Categorize FP/FN rows and summarize error-group metrics."""

    required = {label_col, prediction_col, probability_col, "term_id", "item_id", "query"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Error analysis kolonları eksik: {sorted(missing)}")
    out = frame.copy()
    label = out[label_col].astype(int)
    prediction = out[prediction_col].astype(int)
    out["error_type"] = np.select(
        [(label.eq(0) & prediction.eq(1)), (label.eq(1) & prediction.eq(0))],
        ["false_positive", "false_negative"], default="correct",
    )
    normalized_query = out["query"].fillna("").map(normalize_text)
    categories = pd.Series("other", index=out.index, dtype="string")
    mappings = [
        ("wrong_product_type", "product_type_error"), ("wrong_category", "category_error"),
        ("wrong_brand", "brand_error"), ("wrong_gender", "gender_error"),
        ("wrong_age_group", "age_error"), ("wrong_color", "color_error"),
        ("wrong_size_measure", "size_measure_error"),
        ("wrong_model_capacity", "model_capacity_error"),
        ("accessory_main_mismatch", "accessory_main_confusion"),
    ]
    for column, name in mappings:
        if column in out:
            categories = categories.mask(categories.eq("other") & out[column].fillna(0).astype(bool), name)
    categories = categories.mask(categories.eq("other") & normalized_query.str.split().str.len().le(1), "very_short_query")
    categories = categories.mask(
        categories.eq("other") & ~normalized_query.str.contains(r"[çğıöşü]", regex=True), "turkish_ascii_query"
    )
    if "same_family" in out:
        categories = categories.mask(categories.eq("other") & out["same_family"].astype(bool), "similar_variant")
    if "false_negative_reason" in out:
        categories = categories.mask(
            categories.eq("other") & out["false_negative_reason"].fillna("").ne(""), "possible_false_negative"
        )
    out["error_group"] = categories
    error_rows = out[out["error_type"].ne("correct")].copy()
    summaries: list[dict[str, Any]] = []
    for group_name, group in out.groupby("error_group", dropna=False):
        metrics = classification_report_dict(
            group[label_col].to_numpy(), group[prediction_col].to_numpy(),
            probability=group[probability_col].to_numpy(), threshold=None,
        )
        metrics["error_group"] = str(group_name)
        metrics["false_positive_count"] = int(((group[label_col] == 0) & (group[prediction_col] == 1)).sum())
        metrics["false_negative_count"] = int(((group[label_col] == 1) & (group[prediction_col] == 0)).sum())
        summaries.append(metrics)
    return error_rows, pd.DataFrame(summaries).sort_values("rows", ascending=False)


def parse_args() -> argparse.Namespace:
    """Parse the manifest-building CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3/validation"))
    parser.add_argument("--kind", choices=["group", "semantic", "test_like", "all"], default="group")
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--semantic-clusters", type=int, default=192)
    parser.add_argument("--test-like-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Build requested split manifests and adversarial-validation reports."""

    from trendyol_v3_core import AuditConfig, load_data_bundle

    args = parse_args()
    bundle = load_data_bundle(args.data_dir, AuditConfig(debug=False, seed=args.seed))
    train_term_ids = pd.Index(bundle.training_pairs["term_id"].unique())
    test_term_ids = pd.Index(bundle.submission_pairs["term_id"].unique())
    train_terms = bundle.terms[bundle.terms["term_id"].isin(train_term_ids)].reset_index(drop=True)
    test_terms = bundle.terms[bundle.terms["term_id"].isin(test_term_ids)].reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.kind in {"group", "all"}:
        config = SplitConfig(kind="group", n_splits=args.n_splits, seed=args.seed)
        build_group_manifest(bundle.training_pairs, bundle.terms, config).save(args.output_dir)
    embeddings: np.ndarray | None = None
    if args.kind in {"semantic", "test_like", "all"}:
        term_frame = _term_frame(bundle.training_pairs, bundle.terms)
        embeddings = (
            np.load(args.embeddings) if args.embeddings
            else encode_query_embeddings(term_frame, args.output_dir / "embedding_cache", seed=args.seed)
        )
    semantic_manifest: SplitManifest | None = None
    if args.kind in {"semantic", "all"}:
        config = SplitConfig(
            kind="semantic", n_splits=args.n_splits, seed=args.seed,
            semantic_clusters=args.semantic_clusters,
        )
        semantic_manifest = build_semantic_manifest(bundle.training_pairs, bundle.terms, embeddings, config)
        semantic_manifest.save(args.output_dir)
    if args.kind in {"test_like", "all"}:
        adversarial = run_adversarial_validation(train_terms, test_terms, n_splits=args.n_splits, seed=args.seed)
        adversarial.save(args.output_dir)
        if semantic_manifest is None:
            semantic_config = SplitConfig(
                kind="semantic", n_splits=args.n_splits, seed=args.seed,
                semantic_clusters=args.semantic_clusters,
            )
            semantic_manifest = build_semantic_manifest(
                bundle.training_pairs, bundle.terms, embeddings, semantic_config
            )
        config = SplitConfig(
            kind="test_like", n_splits=args.n_splits, seed=args.seed,
            semantic_clusters=args.semantic_clusters, test_like_fraction=args.test_like_fraction,
        )
        test_like = build_test_like_manifest(
            bundle.training_pairs,
            bundle.terms,
            adversarial.term_scores,
            config,
            semantic_clusters=semantic_manifest.frame["semantic_cluster"],
        )
        test_like.save(args.output_dir)


if __name__ == "__main__":
    main()
