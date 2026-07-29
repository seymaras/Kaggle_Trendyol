#!/usr/bin/env python3
"""Build a test-domain adaptation set without treating pseudo rows as OOF gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--base-train", type=Path,
        default=Path("artifacts/experiments/query_mining_v1/train_query_negatives.parquet"),
    )
    parser.add_argument(
        "--pseudopositives", type=Path,
        default=Path("artifacts/structural_pseudolabels/test_structural_pseudopositives.parquet"),
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
        "--probabilities", type=Path,
        default=Path("artifacts/experiments/classical_v2/probabilities.parquet"),
    )
    parser.add_argument("--max-negatives-per-term", type=int, default=30)
    parser.add_argument("--negative-weight", type=float, default=0.35)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/transductive_training"),
    )
    args = parser.parse_args()

    pairs = pd.read_csv(
        args.data_dir / "submission_pairs.csv", usecols=["id", "term_id", "item_id"], dtype="string",
    )
    pairs["candidate_position"] = pairs.groupby("term_id", sort=False).cumcount().add(1).astype(np.int32)
    pairs["candidate_count"] = pairs.groupby("term_id", sort=False)["item_id"].transform("size").astype(np.int32)
    anchor = pd.read_csv(args.anchor, dtype={"id": "string", "prediction": "int8"})
    secondary = pd.read_csv(args.secondary, dtype={"id": "string", "prediction": "int8"})
    probability = pd.read_parquet(args.probabilities, columns=["id", "probability"])
    for name, frame in (("anchor", anchor), ("secondary", secondary), ("probability", probability)):
        if len(frame) != len(pairs) or not frame["id"].astype("string").equals(pairs["id"]):
            raise ValueError(f"{name} IDs/order do not match submission pairs")
    pairs["anchor_prediction"] = anchor["prediction"].to_numpy(dtype=np.int8)
    pairs["secondary_prediction"] = secondary["prediction"].to_numpy(dtype=np.int8)
    pairs["anchor_probability"] = probability["probability"].to_numpy(dtype=np.float32)

    pseudo = pd.read_parquet(args.pseudopositives)
    positive_ids = set(pseudo["id"].astype(str))
    positive = pairs[pairs["id"].isin(positive_ids)].merge(
        pseudo[["id", "sample_weight", "confidence_tier", "fast_retrieval_score"]],
        on="id", how="left", validate="one_to_one",
    )
    target_terms = set(positive["term_id"].astype(str))
    negative = pairs[
        pairs["term_id"].isin(target_terms)
        & pairs["anchor_prediction"].eq(0)
        & pairs["secondary_prediction"].eq(0)
        & ~pairs["id"].isin(positive_ids)
    ].copy()
    negative = (
        negative.sort_values(["term_id", "anchor_probability", "id"])
        .groupby("term_id", sort=False).head(args.max_negatives_per_term)
    )
    selected = pd.concat([
        positive.assign(label=np.int8(1), pseudo_kind="structural_positive"),
        negative.assign(
            label=np.int8(0), sample_weight=np.float32(args.negative_weight),
            confidence_tier="two_model_negative", fast_retrieval_score=np.float32(0),
            pseudo_kind="consensus_negative",
        ),
    ], ignore_index=True)
    terms = pd.read_csv(args.data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string")
    items = pd.read_csv(args.data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype="string")
    selected = (
        selected.merge(terms, on="term_id", how="left", validate="many_to_one")
        .merge(items, on="item_id", how="left", validate="many_to_one")
    )
    selected["negative_type"] = selected["pseudo_kind"].map({
        "structural_positive": "test_structural_positive",
        "consensus_negative": "test_consensus_negative",
    })
    selected["negative_confidence"] = selected["confidence_tier"]
    selected["retrieval_source"] = "test_transductive"
    selected["retrieval_rank"] = selected["candidate_position"].astype(np.int32)
    selected["retrieval_score"] = selected["fast_retrieval_score"].astype(np.float32)
    selected["base_sample_weight"] = selected["sample_weight"].astype(np.float32)
    selected["is_transductive"] = True

    base = pd.read_parquet(args.base_train)
    base["is_transductive"] = False
    columns = list(base.columns)
    for column in columns:
        if column not in selected:
            selected[column] = pd.NA
    adaptation = selected[columns].copy()
    combined = pd.concat([base[columns], adaptation], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adaptation.to_parquet(args.output_dir / "test_domain_adaptation_pairs.parquet", index=False)
    combined.to_parquet(args.output_dir / "combined_train_transductive.parquet", index=False)
    report = {
        "base_rows": len(base),
        "pseudo_positive_rows": int(adaptation["label"].eq(1).sum()),
        "pseudo_negative_rows": int(adaptation["label"].eq(0).sum()),
        "pseudo_terms": int(adaptation["term_id"].nunique()),
        "combined_rows": len(combined),
        "pseudo_positive_weight_mean": float(adaptation.loc[adaptation["label"].eq(1), "sample_weight"].mean()),
        "pseudo_negative_weight": args.negative_weight,
    }
    (args.output_dir / "transductive_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
