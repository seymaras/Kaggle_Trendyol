#!/usr/bin/env python3
"""Safely combine base and second-stage model-hard training rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, extra = pd.read_parquet(args.base), pd.read_parquet(args.extra)
    columns = list(base.columns)
    missing = set(columns) - set(extra.columns)
    if missing:
        raise ValueError(f"Extra training kolonları eksik: {sorted(missing)}")
    combined = pd.concat([base, extra[columns]], ignore_index=True)
    combined["_priority"] = combined["label"].eq(1).astype(int)
    combined = combined.sort_values("_priority", ascending=False).drop_duplicates(["term_id", "item_id"]).drop(columns="_priority")
    if combined.duplicated(["term_id", "item_id"]).any():
        raise RuntimeError("Duplicate pair kaldı")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    print(f"{args.output}: base={len(base):,}, extra={len(extra):,}, combined={len(combined):,}")


if __name__ == "__main__":
    main()
