#!/usr/bin/env python3
"""Recover likely appended test positives from existing classical shards.

Most submission groups contain exactly 100 candidates.  For groups above 100,
the bottom ``count - 100`` rows under a retrieval-like lexical score are a
structural positive proxy.  Existing relevance predictions only set confidence
tiers; they never determine which rows satisfy the structural constraint.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "id", "term_id", "item_id", "overlap_title", "jaccard_title",
    "overlap_category", "brand_in_query",
]


def load_feature_shards(feature_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(feature_dir / "part-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"feature shards not found under {feature_dir}")
    parts = [pd.read_parquet(path, columns=FEATURE_COLUMNS) for path in paths]
    frame = pd.concat(parts, ignore_index=True)
    if frame["id"].duplicated().any():
        raise ValueError("feature shard IDs are not unique")
    return frame


def add_fast_retrieval_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    taxonomy = out["overlap_category"].gt(0) | out["brand_in_query"].gt(0)
    out["fast_retrieval_score"] = (
        out["overlap_title"]
        + 0.35 * out["jaccard_title"]
        + 0.50 * out["overlap_title"].eq(1.0)
        + 0.20 * taxonomy
    ).astype(np.float32)
    group = out.groupby("term_id", sort=False)
    out["candidate_position"] = group.cumcount().add(1).astype(np.int32)
    out["candidate_count"] = group["item_id"].transform("size").astype(np.int32)
    out["fast_retrieval_rank"] = group["fast_retrieval_score"].rank(
        method="first", ascending=False,
    ).astype(np.int32)
    out["is_structural_residual"] = (
        out["candidate_count"].gt(100) & out["fast_retrieval_rank"].gt(100)
    )
    out["is_order_residual"] = out["candidate_position"].gt(100)
    boundary_rows = (
        out.sort_values(["term_id", "fast_retrieval_score"], ascending=[True, False])
        .groupby("term_id", sort=False, as_index=False).nth(99)
    )
    boundary = boundary_rows.set_index("term_id")["fast_retrieval_score"]
    out["retrieval_boundary"] = out["term_id"].map(boundary).astype(np.float32)
    out["retrieval_margin"] = (
        out["retrieval_boundary"] - out["fast_retrieval_score"]
    ).clip(lower=0).astype(np.float32)
    return out


def load_aligned_prediction(path: Path, ids: pd.Series, column: str) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"id": "string", "prediction": "int8"})
    if len(frame) != len(ids) or not frame["id"].equals(ids.astype("string")):
        raise ValueError(f"prediction IDs/order do not match feature shards: {path}")
    return frame["prediction"].to_numpy(dtype=np.int8)


def make_pseudolabels(residual: pd.DataFrame) -> pd.DataFrame:
    out = residual[[
        "id", "term_id", "item_id", "candidate_count", "candidate_position",
        "fast_retrieval_rank", "is_structural_residual", "is_order_residual",
        "fast_retrieval_score", "retrieval_margin", "anchor_prediction",
        "secondary_prediction",
    ]].copy()
    both_structure = out["is_structural_residual"] & out["is_order_residual"]
    fast_only = out["is_structural_residual"] & ~out["is_order_residual"]
    order_only = out["is_order_residual"] & ~out["is_structural_residual"]
    model_consensus = out["anchor_prediction"].eq(1) & out["secondary_prediction"].eq(1)
    out["label"] = np.int8(1)
    out["sample_weight"] = np.select(
        [both_structure & model_consensus, both_structure, fast_only, order_only],
        [1.0, 0.90, 0.80, 0.70], default=0.55,
    ).astype(np.float32)
    out["confidence_tier"] = np.select(
        [both_structure & model_consensus, both_structure, fast_only, order_only],
        ["both_structure+two_models", "both_structure", "fast_only", "order_only"],
        default="structural_fallback",
    )
    out["source"] = "test_structural_residual"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir", type=Path,
        default=Path("artifacts/experiments/classical_v2/feature_shards"),
    )
    parser.add_argument(
        "--anchor", type=Path,
        default=Path("artifacts/experiments/classical_v2/submission.csv"),
    )
    parser.add_argument(
        "--secondary", type=Path,
        default=Path("artifacts/experiments/classical_v3/submission.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/structural_pseudolabels"),
    )
    args = parser.parse_args()

    scored = add_fast_retrieval_score(load_feature_shards(args.feature_dir))
    scored["anchor_prediction"] = load_aligned_prediction(args.anchor, scored["id"], "anchor")
    scored["secondary_prediction"] = load_aligned_prediction(args.secondary, scored["id"], "secondary")
    residual = scored[scored["is_structural_residual"]].copy()
    order_residual = scored[scored["is_order_residual"]].copy()
    expected = int((
        scored.groupby("term_id", sort=False)["candidate_count"].first() - 100
    ).clip(lower=0).sum())
    if len(residual) != expected:
        raise RuntimeError(f"structural residual mismatch: {len(residual):,} != {expected:,}")

    structural_union = scored[
        scored["is_structural_residual"] | scored["is_order_residual"]
    ].copy()
    pseudolabels = make_pseudolabels(structural_union)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    residual.to_parquet(args.output_dir / "fast_structural_residual.parquet", index=False)
    order_residual.to_parquet(args.output_dir / "order_structural_residual.parquet", index=False)
    pseudolabels.to_parquet(args.output_dir / "test_structural_pseudopositives.parquet", index=False)
    novel = pseudolabels[pseudolabels["anchor_prediction"].eq(0)].copy()
    novel.to_parquet(args.output_dir / "anchor_disagreement_candidates.parquet", index=False)
    counts = pseudolabels["confidence_tier"].value_counts()
    report = {
        "rows": len(scored),
        "groups": int(scored["term_id"].nunique()),
        "groups_above_100": int(scored.loc[scored["candidate_count"].gt(100), "term_id"].nunique()),
        "expected_structural_residual": expected,
        "selected_structural_residual": len(residual),
        "selected_order_residual": len(order_residual),
        "structural_intersection": int((
            scored["is_structural_residual"] & scored["is_order_residual"]
        ).sum()),
        "structural_union": len(structural_union),
        "anchor_positive": int(residual["anchor_prediction"].eq(1).sum()),
        "anchor_disagreement": len(novel),
        "confidence_tiers": {str(key): int(value) for key, value in counts.items()},
        "mean_sample_weight": float(pseudolabels["sample_weight"].mean()),
    }
    (args.output_dir / "structural_pseudolabel_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
