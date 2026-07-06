#!/usr/bin/env python3
"""Create stable term folds and evaluate saved retrieval candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiment_utils import stable_term_folds
from retrieval_utils import retrieval_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-pairs", type=Path, default=Path("data/training_pairs.csv"))
    parser.add_argument("--retrieval-candidates", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positives = pd.read_csv(args.training_pairs, usecols=["term_id", "item_id"], dtype="string")
    folds = stable_term_folds(positives["term_id"], args.n_splits, args.seed)
    folds.to_parquet(args.output_dir / "term_folds.parquet", index=False)
    if folds.groupby("term_id")["fold"].nunique().max() != 1:
        raise RuntimeError("Bir term birden fazla fold'a atandı")
    if args.retrieval_candidates:
        candidates = pd.read_parquet(args.retrieval_candidates)
        report = retrieval_metrics(candidates, positives)
        report.to_csv(args.output_dir / "retrieval_metrics.csv", index=False)
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
