#!/usr/bin/env python3
"""Convert high-scoring unknown pairs into low-weight second-stage negatives."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-query", type=int, default=6)
    args = parser.parse_args()
    scores = pd.read_parquet(args.probabilities).sort_values(["term_id", "probability"], ascending=[True, False])
    known = pd.read_csv(args.data_dir / "training_pairs.csv", usecols=["term_id", "item_id"], dtype="string")
    known_pairs = pd.MultiIndex.from_frame(known)
    candidate_index = pd.MultiIndex.from_frame(scores[["term_id", "item_id"]])
    unknown = scores[~candidate_index.isin(known_pairs)].groupby("term_id", sort=False).head(args.per_query).copy()
    terms = pd.read_csv(args.data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string")
    items = pd.read_csv(args.data_dir / "items.csv", dtype="string")
    unknown = unknown.merge(terms, on="term_id", validate="many_to_one").merge(items, on="item_id", validate="many_to_one")
    unknown["label"] = np.int8(0)
    unknown["negative_type"] = "model_hard"
    unknown["negative_confidence"] = "ambiguous_model_hard"
    unknown["retrieval_source"] = "cross_encoder"
    unknown["retrieval_rank"] = unknown.groupby("term_id")["probability"].rank(method="first", ascending=False).astype(int)
    unknown["retrieval_score"] = unknown["probability"]
    unknown["sample_weight"] = np.clip(0.35 - 0.20 * unknown["probability"], 0.15, 0.35).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    unknown.to_parquet(args.output, index=False)
    print(f"{args.output}: {len(unknown):,}")


if __name__ == "__main__":
    main()
