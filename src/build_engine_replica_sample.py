#!/usr/bin/env python3
"""Create a deterministic real-data slice for end-to-end engine-replica QA."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def stable_key(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(), "little")


def choose_terms(counts: pd.Series, *, size: int, seed: int) -> list[str]:
    values = [str(value) for value in counts.index]
    return sorted(values, key=lambda value: stable_key(value, seed))[:size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/engine_replica_real_sample/data"))
    parser.add_argument("--exact-terms", type=int, default=300)
    parser.add_argument("--excess-terms", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = pd.read_csv(args.data_dir / "submission_pairs.csv", dtype="string")
    counts = pairs.groupby("term_id").size()
    exact = choose_terms(counts[counts.eq(100)], size=args.exact_terms, seed=args.seed)
    excess = choose_terms(counts[counts.gt(100)], size=args.excess_terms, seed=args.seed + 1)
    selected = set(exact + excess)
    pairs = pairs[pairs["term_id"].astype(str).isin(selected)].copy()
    terms = pd.read_csv(args.data_dir / "terms.csv", dtype="string")
    terms = terms[terms["term_id"].astype(str).isin(selected)].copy()
    item_ids = set(pairs["item_id"].astype(str))
    items = pd.read_csv(args.data_dir / "items.csv", dtype="string")
    items = items[items["item_id"].astype(str).isin(item_ids)].copy()
    sample = pd.DataFrame({"id": pairs["id"], "prediction": 0})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "submission_pairs.csv", index=False)
    terms.to_csv(args.output_dir / "terms.csv", index=False)
    items.to_csv(args.output_dir / "items.csv", index=False)
    sample.to_csv(args.output_dir / "sample_submission.csv", index=False)
    print({
        "exact_terms": len(exact), "excess_terms": len(excess),
        "pairs": len(pairs), "items": len(items), "excess_rows": int((counts.reindex(excess) - 100).sum()),
    })


if __name__ == "__main__":
    main()
