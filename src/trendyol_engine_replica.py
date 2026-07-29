#!/usr/bin/env python3
"""Learn observable retrieval membership and recover structural residuals.

Queries with exactly 100 supplied candidates act as positive examples of the
hidden candidate engine.  Semantically-near queries provide hard decoys.  The
replica is evaluated query-cold on a 100-positive/100-decoy pool and is then
used to separate the likely base top-100 from appended rows in larger groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_trendyol_domain_features import pair_text_features  # noqa: E402


FEATURE_NAMES = [
    "ty_cosine", "token_recall", "token_jaccard", "query_in_title",
    "taxonomy_overlap", "title_char_count",
]


def stable_fold(value: str, n_splits: int = 5, seed: int = 42) -> int:
    raw = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(raw, "little") % n_splits


def require_catboost():
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("catboost is required for the engine replica") from exc
    return CatBoostClassifier


def load_cache(cache_dir: Path):
    query_catalog = pd.read_parquet(cache_dir / "query_catalog.parquet")
    item_catalog = pd.read_parquet(cache_dir / "item_catalog.parquet")
    query_embedding = np.load(cache_dir / "query_embeddings.f16.npy", mmap_mode="r")
    item_embedding = np.load(cache_dir / "item_embeddings.f16.npy", mmap_mode="r")
    return query_catalog, item_catalog, query_embedding, item_embedding


def semantic_cosine(
    query_indices: np.ndarray,
    item_indices: np.ndarray,
    query_embedding: np.ndarray,
    item_embedding: np.ndarray,
    chunk_size: int = 200_000,
) -> np.ndarray:
    result = np.empty(len(query_indices), dtype=np.float32)
    for start in range(0, len(result), chunk_size):
        end = min(start + chunk_size, len(result))
        q = np.asarray(query_embedding[query_indices[start:end]], dtype=np.float32)
        i = np.asarray(item_embedding[item_indices[start:end]], dtype=np.float32)
        result[start:end] = np.einsum("ij,ij->i", q, i)
    return result


def calculate_pair_features(
    query_indices: np.ndarray,
    item_indices: np.ndarray,
    query_catalog: pd.DataFrame,
    item_catalog: pd.DataFrame,
    query_embedding: np.ndarray,
    item_embedding: np.ndarray,
) -> pd.DataFrame:
    lexical = pair_text_features(
        query_indices, item_indices,
        query_catalog["query"].fillna("").astype(str).tolist(),
        item_catalog["title"].fillna("").astype(str).tolist(),
        item_catalog["category"].fillna("").astype(str).tolist(),
        item_catalog["brand"].fillna("").astype(str).tolist(),
    )
    query_lengths = query_catalog["query"].fillna("").str.len().to_numpy(dtype=np.float32)
    title_lengths = item_catalog["title"].fillna("").str.len().to_numpy(dtype=np.float32)
    return pd.DataFrame({
        "ty_cosine": semantic_cosine(query_indices, item_indices, query_embedding, item_embedding),
        "token_recall": lexical[0],
        "token_jaccard": lexical[1],
        "query_in_title": lexical[2],
        "taxonomy_overlap": lexical[3],
        "query_char_count": query_lengths[query_indices],
        "title_char_count": title_lengths[item_indices],
    })


def nearest_query_indices(query_embedding: np.ndarray, query_indices: np.ndarray, neighbors: int = 12) -> np.ndarray:
    """Return nearest cache indices for the selected queries."""

    vectors = np.asarray(query_embedding[query_indices], dtype=np.float32)
    try:
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        _, local = index.search(vectors, min(neighbors + 1, len(vectors)))
    except ImportError:
        similarity = vectors @ vectors.T
        local = np.argsort(-similarity, axis=1)[:, : min(neighbors + 1, len(vectors))]
    return query_indices[local]


def sample_hard_decoys(
    positive: pd.DataFrame,
    selected_terms: list[str],
    query_embedding: np.ndarray,
    *,
    negatives_per_query: int,
    seed: int,
) -> pd.DataFrame:
    """Borrow candidates from nearby queries while excluding observed pairs."""

    rng = np.random.default_rng(seed)
    by_term = {
        str(term): group["item_embedding_index"].to_numpy(dtype=np.int32)
        for term, group in positive.groupby("term_id", sort=False)
    }
    q_index = positive.groupby("term_id", sort=False)["query_embedding_index"].first().reindex(selected_terms)
    cache_indices = q_index.to_numpy(dtype=np.int32)
    neighbors = nearest_query_indices(query_embedding, cache_indices, neighbors=16)
    cache_to_term = {int(idx): term for term, idx in zip(selected_terms, cache_indices)}
    global_items = positive["item_embedding_index"].drop_duplicates().to_numpy(dtype=np.int32)
    records: list[tuple[str, int, int]] = []
    for row, term_id in enumerate(selected_terms):
        query_idx = int(cache_indices[row])
        observed = set(int(value) for value in by_term[term_id])
        candidates: list[int] = []
        for neighbor_idx in neighbors[row]:
            neighbor_term = cache_to_term.get(int(neighbor_idx))
            if not neighbor_term or neighbor_term == term_id:
                continue
            borrowed = by_term[neighbor_term].copy()
            rng.shuffle(borrowed)
            candidates.extend(int(value) for value in borrowed if int(value) not in observed)
            if len(set(candidates)) >= negatives_per_query * 2:
                break
        if len(set(candidates)) < negatives_per_query:
            fallback = rng.choice(global_items, size=min(len(global_items), negatives_per_query * 3), replace=False)
            candidates.extend(int(value) for value in fallback if int(value) not in observed)
        unique = list(dict.fromkeys(candidates))
        if len(unique) < negatives_per_query:
            raise RuntimeError(f"could not sample {negatives_per_query} decoys for {term_id}")
        chosen = unique[:negatives_per_query]
        records.extend((term_id, query_idx, item_idx) for item_idx in chosen)
    return pd.DataFrame(records, columns=["term_id", "query_embedding_index", "item_embedding_index"])


def recall_at_group_cutoff(frame: pd.DataFrame, score_col: str, cutoff: int = 100) -> pd.Series:
    ordered = frame.sort_values(["term_id", score_col], ascending=[True, False])
    top = ordered.groupby("term_id", sort=False).head(cutoff)
    positives = top.groupby("term_id")["membership_label"].sum()
    denominators = frame.groupby("term_id")["membership_label"].sum().clip(upper=cutoff)
    return (positives / denominators).fillna(0.0)


def train_replica(args: argparse.Namespace) -> dict[str, object]:
    feature_frame = pd.read_parquet(args.features)
    required = {
        "term_id", "item_id", "candidate_count", "query_embedding_index",
        "item_embedding_index", "ty_cosine", "token_recall", "token_jaccard",
        "query_in_title", "taxonomy_overlap",
    }
    if missing := required - set(feature_frame.columns):
        raise ValueError(f"domain feature columns missing: {sorted(missing)}")
    exact = feature_frame[feature_frame["candidate_count"].eq(100)].copy()
    exact_terms = exact["term_id"].astype(str).drop_duplicates().to_numpy()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(exact_terms)
    if args.max_exact_terms:
        exact_terms = exact_terms[: args.max_exact_terms]
    selected_terms = [str(value) for value in exact_terms]
    exact = exact[exact["term_id"].astype(str).isin(selected_terms)].copy()
    query_catalog, item_catalog, query_embedding, item_embedding = load_cache(args.cache_dir)
    decoy = sample_hard_decoys(
        exact, selected_terms, query_embedding,
        negatives_per_query=args.negatives_per_query, seed=args.seed,
    )
    decoy_features = calculate_pair_features(
        decoy["query_embedding_index"].to_numpy(dtype=np.int32),
        decoy["item_embedding_index"].to_numpy(dtype=np.int32),
        query_catalog, item_catalog, query_embedding, item_embedding,
    )
    decoy = pd.concat([decoy.reset_index(drop=True), decoy_features], axis=1)
    positive = exact[["term_id", *FEATURE_NAMES]].copy()
    positive["membership_label"] = np.int8(1)
    decoy["membership_label"] = np.int8(0)
    pool = pd.concat([positive, decoy[["term_id", *FEATURE_NAMES, "membership_label"]]], ignore_index=True)
    pool["fold"] = pool["term_id"].astype(str).map(lambda value: stable_fold(value, 5, args.seed)).astype(np.int8)
    train = pool[pool["fold"].ne(0)]
    valid = pool[pool["fold"].eq(0)].copy()
    CatBoostClassifier = require_catboost()
    model = CatBoostClassifier(
        iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate,
        l2_leaf_reg=5.0, loss_function="Logloss", eval_metric="AUC",
        random_seed=args.seed, verbose=args.verbose, thread_count=-1,
    )
    model.fit(
        train[FEATURE_NAMES], train["membership_label"],
        eval_set=(valid[FEATURE_NAMES], valid["membership_label"]),
        use_best_model=True,
    )
    valid["replica_score"] = model.predict_proba(valid[FEATURE_NAMES])[:, 1]
    valid["lexical_score"] = (
        valid["token_recall"] + 0.35 * valid["token_jaccard"]
        + 0.50 * valid["query_in_title"] + 0.20 * valid["taxonomy_overlap"]
    )
    replica_recall = recall_at_group_cutoff(valid, "replica_score", 100)
    dense_recall = recall_at_group_cutoff(valid, "ty_cosine", 100)
    lexical_recall = recall_at_group_cutoff(valid, "lexical_score", 100)
    recall_by_method = {
        "replica": float(replica_recall.mean()),
        "dense": float(dense_recall.mean()),
        "lexical": float(lexical_recall.mean()),
    }
    primary_method = max(recall_by_method, key=recall_by_method.get)
    report: dict[str, object] = {
        "selected_exact_terms": len(selected_terms),
        "train_terms": int(train["term_id"].nunique()),
        "valid_terms": int(valid["term_id"].nunique()),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "valid_auc": float(roc_auc_score(valid["membership_label"], valid["replica_score"])),
        "valid_average_precision": float(average_precision_score(valid["membership_label"], valid["replica_score"])),
        "replica_recall_at_100_mean": float(replica_recall.mean()),
        "replica_recall_at_100_p10": float(replica_recall.quantile(0.10)),
        "dense_recall_at_100_mean": float(dense_recall.mean()),
        "lexical_recall_at_100_mean": float(lexical_recall.mean()),
        "primary_method": primary_method,
        "best_iteration": int(model.get_best_iteration()),
        "feature_importance": dict(zip(FEATURE_NAMES, map(float, model.get_feature_importance()))),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_dir / "engine_replica.cbm"))
    joblib.dump(
        {"feature_names": FEATURE_NAMES, "primary_method": primary_method},
        args.output_dir / "engine_replica_meta.joblib",
    )
    (args.output_dir / "engine_replica_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def add_structural_residual_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    group = out.groupby("term_id", sort=False)
    out["engine_rank"] = group["engine_score"].rank(method="first", ascending=False).astype(np.int32)
    replica_score = "replica_score" if "replica_score" in out else "engine_score"
    out["replica_rank"] = group[replica_score].rank(method="first", ascending=False).astype(np.int32)
    out["lexical_rank"] = group["lexical_score"].rank(method="first", ascending=False).astype(np.int32)
    if "ty_rank" not in out:
        out["ty_rank"] = group["ty_cosine"].rank(method="first", ascending=False).astype(np.int32)
    out["is_structural_residual"] = out["candidate_count"].gt(100) & out["engine_rank"].gt(100)
    out["outside_agreement"] = (
        out["replica_rank"].gt(100).astype(np.int8)
        + out["ty_rank"].gt(100).astype(np.int8)
        + out["lexical_rank"].gt(100).astype(np.int8)
    )
    boundary_rows = (
        out.sort_values(["term_id", "engine_score"], ascending=[True, False])
        .groupby("term_id", sort=False, as_index=False).nth(99)
    )
    boundary = boundary_rows.set_index("term_id")["engine_score"]
    out["engine_boundary"] = out["term_id"].map(boundary).astype(np.float32)
    std = group["engine_score"].transform("std").fillna(0.0).clip(lower=1e-6)
    out["forced_margin"] = ((out["engine_boundary"] - out["engine_score"]) / std).clip(lower=0).astype(np.float32)
    out["forced_confidence"] = (
        out["forced_margin"] + 0.30 * (out["outside_agreement"] - 1).clip(lower=0)
    ).astype(np.float32)
    return out


def apply_replica(args: argparse.Namespace) -> None:
    CatBoostClassifier = require_catboost()
    model = CatBoostClassifier()
    model.load_model(str(args.model_path or (args.output_dir / "engine_replica.cbm")))
    frame = pd.read_parquet(args.features)
    frame["query_char_count"] = frame["term_id"].map(
        pd.read_parquet(args.cache_dir / "query_catalog.parquet").set_index("term_id")["query"].fillna("").str.len()
    ).astype(np.float32)
    frame["title_char_count"] = frame["item_id"].map(
        pd.read_parquet(args.cache_dir / "item_catalog.parquet").set_index("item_id")["title"].fillna("").str.len()
    ).astype(np.float32)
    frame["replica_score"] = model.predict_proba(frame[FEATURE_NAMES])[:, 1].astype(np.float32)
    frame["lexical_score"] = (
        frame["token_recall"] + 0.35 * frame["token_jaccard"]
        + 0.50 * frame["query_in_title"] + 0.20 * frame["taxonomy_overlap"]
    ).astype(np.float32)
    metadata_path = args.output_dir / "engine_replica_meta.joblib"
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {"primary_method": "replica"}
    primary_method = metadata.get("primary_method", "replica")
    primary_column = {
        "replica": "replica_score", "dense": "ty_cosine", "lexical": "lexical_score",
    }[primary_method]
    frame["engine_score"] = frame[primary_column].astype(np.float32)
    scored = add_structural_residual_columns(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.output_dir / "submission_engine_scores.parquet", index=False)
    residual = scored[scored["is_structural_residual"]].sort_values(
        ["forced_confidence", "term_id"], ascending=[False, True]
    )
    residual.to_parquet(args.output_dir / "structural_residual_candidates.parquet", index=False)
    expected = int((scored.groupby("term_id")["candidate_count"].first() - 100).clip(lower=0).sum())
    summary = {
        "rows": len(scored),
        "groups": int(scored["term_id"].nunique()),
        "groups_above_100": int(scored.loc[scored["candidate_count"].gt(100), "term_id"].nunique()),
        "expected_structural_residual": expected,
        "selected_structural_residual": len(residual),
        "primary_method": primary_method,
        "agreement_3": int(residual["outside_agreement"].eq(3).sum()),
        "agreement_at_least_2": int(residual["outside_agreement"].ge(2).sum()),
    }
    if len(residual) != expected:
        raise RuntimeError(f"residual count mismatch: {len(residual):,} != {expected:,}")
    (args.output_dir / "structural_residual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "apply", "all"], default="all")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/trendyol_domain"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/engine_replica"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--max-exact-terms", type=int, default=6_000)
    parser.add_argument("--negatives-per-query", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--verbose", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"train", "all"}:
        train_replica(args)
    if args.mode in {"apply", "all"}:
        apply_replica(args)


if __name__ == "__main__":
    main()
