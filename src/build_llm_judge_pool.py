#!/usr/bin/env python3
"""Build a compact, high-leverage test-pair pool for self-hosted LLM judging.

The pool is the union of two independent disagreements with the proven v6
anchor:

1. A global v2/v4 percentile-rank blend at an explicit positive-rate target.
2. A per-query, anchor-count-preserving v2/intent semantic rerank.

No labels are inferred here and no Kaggle API is called.  The resulting pool
is intentionally auditable: every selected row carries its source flags,
scores, original row position, and the binary alternative to the anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_V2 = Path(
    "/Users/seyma/2Kaggle_Trendyol/2lady-recall/data/val_probs/"
    "v2true_submission_probs.parquet"
)
DEFAULT_V4 = Path(
    "/Users/seyma/2Kaggle_Trendyol/2lady-recall/data/val_probs/"
    "submission_probs_v4randneg.parquet"
)
DEFAULT_INTENT = Path(
    "/Users/seyma/2Kaggle_Trendyol/2lady-recall/data/val_probs/"
    "submission_probs_catboost_intent.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/submission_pairs.csv"))
    parser.add_argument("--terms", type=Path, default=Path("data/terms.csv"))
    parser.add_argument("--items", type=Path, default=Path("data/items.csv"))
    parser.add_argument(
        "--anchor",
        type=Path,
        default=Path("artifacts/final_candidates/00_proven_anchor_v6_lb0874.csv"),
    )
    parser.add_argument("--v2-probs", type=Path, default=DEFAULT_V2)
    parser.add_argument("--v4-probs", type=Path, default=DEFAULT_V4)
    parser.add_argument("--intent-probs", type=Path, default=DEFAULT_INTENT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/llm_judge_v1/input")
    )
    parser.add_argument("--target-positive-rate", type=float, default=0.24)
    parser.add_argument("--global-v2-weight", type=float, default=0.55)
    parser.add_argument("--global-v4-weight", type=float, default=0.45)
    parser.add_argument("--count-v2-weight", type=float, default=0.45)
    parser.add_argument("--count-intent-weight", type=float, default=0.55)
    parser.add_argument(
        "--count-anchor-regularization",
        type=float,
        default=0.10,
        help=(
            "Positive bonus for the proven anchor inside the per-query rerank. "
            "This keeps the pool broad while avoiding an unconstrained semantic reset."
        ),
    )
    parser.add_argument(
        "--max-pool-rows",
        type=int,
        default=400_000,
        help="Cap inference cost after taking the union; 0 disables the cap.",
    )
    parser.add_argument(
        "--attribute-chars",
        type=int,
        default=360,
        help="Maximum compact attribute characters retained per item.",
    )
    return parser.parse_args()


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def align_values(
    target_ids: pd.Series,
    source: pd.DataFrame,
    value_col: str,
    source_name: str,
) -> np.ndarray:
    if source.columns.tolist().count("id") != 1 or value_col not in source.columns:
        raise ValueError(f"{source_name}: expected id and {value_col!r} columns")
    if source["id"].duplicated().any():
        raise ValueError(f"{source_name}: duplicate ids")
    if len(source) != len(target_ids):
        raise ValueError(
            f"{source_name}: row mismatch ({len(source):,} != {len(target_ids):,})"
        )

    source_ids = source["id"].astype("string")
    if np.array_equal(source_ids.to_numpy(), target_ids.to_numpy()):
        values = source[value_col].to_numpy()
    else:
        positions = pd.Index(source_ids).get_indexer(target_ids)
        if (positions < 0).any():
            raise ValueError(f"{source_name}: ids do not match submission pairs")
        values = source[value_col].to_numpy()[positions]
    return values


def global_percentile(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values, copy=False)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32, copy=False)
    )


def term_percentile(values: np.ndarray, term_codes: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values, copy=False)
        .groupby(term_codes, sort=False)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32, copy=False)
    )


def exact_top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    n = len(scores)
    if not 0 < k < n:
        raise ValueError(f"top-k must be between 1 and n-1, got {k:,}/{n:,}")
    chosen = np.argpartition(scores, n - k)[n - k :]
    prediction = np.zeros(n, dtype=np.uint8)
    prediction[chosen] = 1
    return prediction, float(np.min(scores[chosen]))


def count_preserving_top_k(
    scores: np.ndarray, term_codes: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select each query's top k, where k is its anchor-positive count."""
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, -scores, term_codes))
    sorted_codes = term_codes[order]

    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    rank_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)

    positive_counts = (
        pd.Series(anchor, copy=False)
        .groupby(term_codes, sort=False)
        .sum()
        .to_numpy(dtype=np.int64, copy=False)
    )
    k_sorted = positive_counts[sorted_codes]
    selected_sorted = rank_sorted < k_sorted

    prediction = np.zeros(n, dtype=np.uint8)
    prediction[order] = selected_sorted.astype(np.uint8)

    # A per-query decision boundary supports audit/prioritisation.  For edge
    # groups (k=0 or k=n), place it just outside the observed score range.
    boundaries = np.empty(len(positive_counts), dtype=np.float32)
    sorted_scores = scores[order]
    for start, end, code in zip(starts, ends, sorted_codes[starts]):
        k = int(positive_counts[code])
        if k <= 0:
            boundaries[code] = np.float32(sorted_scores[start] + 1e-6)
        elif k >= end - start:
            boundaries[code] = np.float32(sorted_scores[end - 1] - 1e-6)
        else:
            boundaries[code] = np.float32(
                (float(sorted_scores[start + k - 1]) + float(sorted_scores[start + k]))
                / 2.0
            )
    return prediction, boundaries[term_codes]


def compact_attributes(values: pd.Series, max_chars: int) -> pd.Series:
    if max_chars < 80:
        raise ValueError("--attribute-chars must be at least 80")
    values = values.fillna("").astype("string").str.replace(r"\s+", " ", regex=True)
    head_chars = int(math.ceil(max_chars * 0.68))
    tail_chars = max_chars - head_chars - 3
    long_mask = values.str.len().fillna(0) > max_chars
    compact = values.copy()
    compact.loc[long_mask] = (
        values.loc[long_mask].str.slice(0, head_chars)
        + " … "
        + values.loc[long_mask].str.slice(-tail_chars)
    )
    return compact


def main() -> None:
    args = parse_args()
    require_paths(
        [
            args.pairs,
            args.terms,
            args.items,
            args.anchor,
            args.v2_probs,
            args.v4_probs,
            args.intent_probs,
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] IDs, anchor and independent score families are loading...")
    pairs = pd.read_csv(
        args.pairs,
        usecols=["id", "term_id", "item_id"],
        dtype={"id": "string", "term_id": "string", "item_id": "string"},
    )
    if pairs["id"].duplicated().any():
        raise ValueError("submission_pairs contains duplicate ids")
    target_ids = pairs["id"]

    anchor_frame = pd.read_csv(
        args.anchor, dtype={"id": "string", "prediction": "uint8"}
    )
    anchor = align_values(target_ids, anchor_frame, "prediction", "anchor").astype(
        np.uint8, copy=False
    )
    if set(np.unique(anchor).tolist()) != {0, 1}:
        raise ValueError("anchor predictions must contain exactly binary 0/1")

    v2_frame = pd.read_parquet(args.v2_probs, columns=["id", "cross_encoder_prob"])
    v4_frame = pd.read_parquet(args.v4_probs, columns=["id", "prob"])
    intent_frame = pd.read_parquet(args.intent_probs, columns=["id", "prob"])
    v2 = align_values(target_ids, v2_frame, "cross_encoder_prob", "v2").astype(
        np.float32, copy=False
    )
    v4 = align_values(target_ids, v4_frame, "prob", "v4").astype(
        np.float32, copy=False
    )
    intent = align_values(target_ids, intent_frame, "prob", "intent").astype(
        np.float32, copy=False
    )
    del anchor_frame, v2_frame, v4_frame, intent_frame

    if not all(np.isfinite(values).all() for values in (v2, v4, intent)):
        raise ValueError("model scores contain NaN or infinite values")

    print("[2/6] Global v2/v4 rank-blend alternative is being built...")
    global_weight_sum = args.global_v2_weight + args.global_v4_weight
    if global_weight_sum <= 0:
        raise ValueError("global blend weights must sum to a positive value")
    v2_rank_global = global_percentile(v2)
    v4_rank_global = global_percentile(v4)
    global_score = (
        args.global_v2_weight * v2_rank_global
        + args.global_v4_weight * v4_rank_global
    ) / global_weight_sum
    global_score = global_score.astype(np.float32, copy=False)
    global_k = int(round(len(pairs) * args.target_positive_rate))
    global_prediction, global_boundary = exact_top_k(global_score, global_k)

    print("[3/6] Per-query anchor-count-preserving semantic alternative is being built...")
    term_codes, unique_terms = pd.factorize(pairs["term_id"], sort=False)
    term_codes = term_codes.astype(np.int32, copy=False)
    count_weight_sum = args.count_v2_weight + args.count_intent_weight
    if count_weight_sum <= 0:
        raise ValueError("count-preserving weights must sum to a positive value")
    v2_rank_term = term_percentile(v2, term_codes)
    intent_rank_term = term_percentile(intent, term_codes)
    count_score = (
        args.count_v2_weight * v2_rank_term
        + args.count_intent_weight * intent_rank_term
    ) / count_weight_sum
    count_score = count_score + args.count_anchor_regularization * anchor
    count_score = count_score.astype(np.float32, copy=False)
    count_prediction, count_boundary = count_preserving_top_k(
        count_score, term_codes, anchor
    )

    source_global = global_prediction != anchor
    source_count = count_prediction != anchor
    union_mask = source_global | source_count
    source_votes = source_global.astype(np.uint8) + source_count.astype(np.uint8)

    global_strength = np.abs(global_score - global_boundary).astype(np.float32)
    count_strength = np.abs(count_score - count_boundary).astype(np.float32)
    priority = (
        source_votes.astype(np.float32) * 10.0
        + np.where(source_global, global_strength, 0.0)
        + np.where(source_count, count_strength, 0.0)
    )

    union_count = int(union_mask.sum())
    if args.max_pool_rows > 0 and union_count > args.max_pool_rows:
        candidate_rows = np.flatnonzero(union_mask)
        keep_local = np.argpartition(
            priority[candidate_rows], len(candidate_rows) - args.max_pool_rows
        )[-args.max_pool_rows :]
        pool_rows = np.sort(candidate_rows[keep_local])
        capped = True
    else:
        pool_rows = np.flatnonzero(union_mask)
        capped = False

    print(
        "    global disputes: "
        f"{int(source_global.sum()):,} | count-preserving disputes: "
        f"{int(source_count.sum()):,} | union: {union_count:,} | kept: "
        f"{len(pool_rows):,}"
    )

    print("[4/6] Query and compact product text are being joined...")
    selected = pairs.iloc[pool_rows].copy()
    selected["row_position"] = pool_rows.astype(np.int32)
    selected["anchor_prediction"] = anchor[pool_rows]
    selected["alternative_prediction"] = 1 - anchor[pool_rows]
    selected["global_prediction"] = global_prediction[pool_rows]
    selected["count_prediction"] = count_prediction[pool_rows]
    selected["source_global"] = source_global[pool_rows]
    selected["source_count"] = source_count[pool_rows]
    selected["source_votes"] = source_votes[pool_rows]
    selected["v2_rank_global"] = v2_rank_global[pool_rows]
    selected["v4_rank_global"] = v4_rank_global[pool_rows]
    selected["v2_rank_term"] = v2_rank_term[pool_rows]
    selected["intent_rank_term"] = intent_rank_term[pool_rows]
    selected["global_score"] = global_score[pool_rows]
    selected["count_score"] = count_score[pool_rows]
    selected["priority"] = priority[pool_rows]

    terms = pd.read_csv(
        args.terms,
        usecols=["term_id", "query"],
        dtype={"term_id": "string", "query": "string"},
    )
    selected = selected.merge(terms, on="term_id", how="left", validate="many_to_one")
    if selected["query"].isna().any():
        raise ValueError("pool contains term ids missing from terms.csv")

    item_columns = [
        "item_id",
        "title",
        "category",
        "brand",
        "gender",
        "age_group",
        "attributes",
    ]
    items = pd.read_csv(
        args.items,
        usecols=item_columns,
        dtype={column: "string" for column in item_columns},
    )
    needed_item_ids = pd.Index(selected["item_id"].unique())
    items = items[items["item_id"].isin(needed_item_ids)].copy()
    items["attributes_compact"] = compact_attributes(
        items.pop("attributes"), args.attribute_chars
    )
    selected = selected.merge(items, on="item_id", how="left", validate="many_to_one")
    if selected["title"].isna().any():
        raise ValueError("pool contains item ids missing from items.csv")

    text_columns = [
        "query",
        "title",
        "category",
        "brand",
        "gender",
        "age_group",
        "attributes_compact",
    ]
    for column in text_columns:
        selected[column] = selected[column].fillna("").astype("string")

    ordered_columns = [
        "id",
        "term_id",
        "item_id",
        "row_position",
        "query",
        "title",
        "category",
        "brand",
        "gender",
        "age_group",
        "attributes_compact",
        "anchor_prediction",
        "alternative_prediction",
        "global_prediction",
        "count_prediction",
        "source_global",
        "source_count",
        "source_votes",
        "v2_rank_global",
        "v4_rank_global",
        "v2_rank_term",
        "intent_rank_term",
        "global_score",
        "count_score",
        "priority",
    ]
    # Group identical queries together.  The runner places the query before the
    # product text, so vLLM prefix caching can reuse both the system prompt and
    # the repeated query prefix across a term's candidates.
    selected = selected[ordered_columns].sort_values(
        ["term_id", "row_position"], kind="stable"
    )
    if selected["id"].duplicated().any() or len(selected) != len(pool_rows):
        raise ValueError("pool join changed row cardinality or introduced duplicate ids")

    print("[5/6] Compact Parquet inputs are being written...")
    pool_path = args.output_dir / "llm_judge_pool.parquet"
    anchor_path = args.output_dir / "anchor_v6.parquet"
    selected.to_parquet(pool_path, index=False, compression="zstd")
    pd.DataFrame(
        {"id": target_ids, "prediction": anchor.astype(np.uint8, copy=False)}
    ).to_parquet(anchor_path, index=False, compression="zstd")

    print("[6/6] Manifest and integrity checks are being written...")
    kept_global = int(source_global[pool_rows].sum())
    kept_count = int(source_count[pool_rows].sum())
    both_sources = int((source_votes[pool_rows] == 2).sum())
    manifest = {
        "schema_version": 1,
        "rows_total": int(len(pairs)),
        "terms_total": int(len(unique_terms)),
        "pool_rows": int(len(selected)),
        "pool_union_before_cap": union_count,
        "pool_was_capped": capped,
        "source_global_rows_kept": kept_global,
        "source_count_rows_kept": kept_count,
        "source_both_rows_kept": both_sources,
        "anchor_positive_count": int(anchor.sum()),
        "anchor_positive_rate": float(anchor.mean()),
        "global_positive_count": int(global_prediction.sum()),
        "global_positive_rate": float(global_prediction.mean()),
        "count_positive_count": int(count_prediction.sum()),
        "count_positive_rate": float(count_prediction.mean()),
        "pool_anchor_0": int((selected["anchor_prediction"] == 0).sum()),
        "pool_anchor_1": int((selected["anchor_prediction"] == 1).sum()),
        "parameters": {
            "target_positive_rate": args.target_positive_rate,
            "global_v2_weight": args.global_v2_weight,
            "global_v4_weight": args.global_v4_weight,
            "count_v2_weight": args.count_v2_weight,
            "count_intent_weight": args.count_intent_weight,
            "count_anchor_regularization": args.count_anchor_regularization,
            "max_pool_rows": args.max_pool_rows,
            "attribute_chars": args.attribute_chars,
        },
        "files": {
            pool_path.name: {
                "bytes": pool_path.stat().st_size,
                "sha256": sha256_file(pool_path),
            },
            anchor_path.name: {
                "bytes": anchor_path.stat().st_size,
                "sha256": sha256_file(anchor_path),
            },
        },
        "source_files": {
            "pairs": str(args.pairs.resolve()),
            "terms": str(args.terms.resolve()),
            "items": str(args.items.resolve()),
            "anchor": str(args.anchor.resolve()),
            "v2_probs": str(args.v2_probs.resolve()),
            "v4_probs": str(args.v4_probs.resolve()),
            "intent_probs": str(args.intent_probs.resolve()),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
