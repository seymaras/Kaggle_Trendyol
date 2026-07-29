#!/usr/bin/env python3
"""Precompute exact seven-model global-rank relevance for 100+ test groups."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--source", nargs=2, action="append", required=True,
        metavar=("PARQUET", "COLUMN"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs, usecols=["id", "term_id"], dtype="string")
    candidate_count = pairs.groupby("term_id", sort=False)["id"].transform("size")
    eligible = candidate_count.gt(100).to_numpy()
    total = np.zeros(len(pairs), dtype=np.float32)

    source_reports = []
    for raw_path, column in args.source:
        path = Path(raw_path)
        frame = pd.read_parquet(path, columns=["id", column])
        ids = frame["id"].astype("string")
        if len(frame) != len(pairs) or not ids.equals(pairs["id"]):
            raise ValueError(f"source IDs/order do not match pairs: {path}")
        if frame[column].isna().any():
            raise ValueError(f"source contains missing scores: {path}")
        ranked = frame[column].rank(pct=True).to_numpy(dtype=np.float32)
        total += ranked
        source_reports.append({"file": path.name, "column": column, "rows": len(frame)})
        del frame, ids, ranked
        gc.collect()

    output = pairs.loc[eligible, ["id", "term_id"]].copy()
    output["relevance_score"] = (total[eligible] / len(args.source)).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")

    counts = candidate_count[eligible].groupby(pairs.loc[eligible, "term_id"]).first()
    report = {
        "all_rows": len(pairs),
        "eligible_rows": len(output),
        "eligible_terms": int(output["term_id"].nunique()),
        "structural_excess": int((counts - 100).sum()),
        "models": len(args.source),
        "score_min": float(output["relevance_score"].min()),
        "score_max": float(output["relevance_score"].max()),
        "sources": source_reports,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
