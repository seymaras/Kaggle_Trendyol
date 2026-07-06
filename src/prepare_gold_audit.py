#!/usr/bin/env python3
"""Prepare a deterministic 2,000-pair manually labelled audit sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiment_utils import validate_probability_frame
from text_features import category_parts, query_token_count


def balanced_band_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["query_length_band"] = pd.cut(
        frame["query"].map(query_token_count), bins=[-1, 1, 2, 10_000], labels=["1", "2", "3+"]
    ).astype(str)
    frame["top_category"] = frame["category"].map(lambda x: category_parts(x)["top_category"])
    frame["stratum"] = frame["query_length_band"] + "|" + frame["top_category"].fillna("")
    shuffled = frame.sample(frac=1, random_state=seed)
    # Round-robin after sorting by within-stratum rank spreads large categories and query lengths.
    shuffled["within_stratum"] = shuffled.groupby("stratum").cumcount()
    return shuffled.sort_values(["within_stratum", "stratum", "id"]).head(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gold_audit_2000.csv"))
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    scores = pd.read_parquet(args.probabilities)
    validate_probability_frame(scores)
    pairs = pd.read_csv(args.data_dir / "submission_pairs.csv", dtype="string")
    terms = pd.read_csv(args.data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string")
    items = pd.read_csv(args.data_dir / "items.csv", usecols=["item_id", "title", "category", "brand", "attributes"], dtype="string")
    frame = scores.merge(pairs, on=["id", "term_id", "item_id"], validate="one_to_one")
    frame = frame.merge(terms, on="term_id", validate="many_to_one").merge(items, on="item_id", validate="many_to_one")
    frame["score_band"] = pd.qcut(frame["probability"].rank(method="first"), 5, labels=["q1", "q2", "q3", "q4", "q5"])
    per_band = args.size // 5
    selected = [balanced_band_sample(group, per_band, args.seed + i) for i, (_, group) in enumerate(frame.groupby("score_band", observed=True))]
    audit = pd.concat(selected, ignore_index=True).sort_values(["score_band", "id"]).reset_index(drop=True)
    within_band = audit.groupby("score_band", observed=True).cumcount()
    band_sizes = audit.groupby("score_band", observed=True)["id"].transform("size")
    audit["audit_split"] = (within_band >= (band_sizes // 2)).map({False: "calibration", True: "final"})
    audit["human_label"] = pd.Series(pd.NA, index=audit.index, dtype="Int8")
    audit["uncertain"] = 0
    audit["notes"] = ""
    columns = ["id", "term_id", "item_id", "query", "title", "category", "brand", "attributes", "probability", "score_band", "audit_split", "human_label", "uncertain", "notes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit[columns].to_csv(args.output, index=False)
    print(f"{args.output}: {len(audit):,} rows")


if __name__ == "__main__":
    main()
