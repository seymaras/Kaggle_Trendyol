#!/usr/bin/env python3
"""Evaluate any probability artifact on fixed gold calibration/final slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from experiment_utils import validate_probability_frame
from text_features import category_parts, query_token_count


def best_threshold(labels, probability):
    candidates = [(f1_score(labels, probability >= t, average="macro"), t) for t in np.arange(0.05, 0.951, 0.005)]
    score, threshold = max(candidates)
    return float(threshold), float(score)


def metrics(labels, prediction):
    return {
        "macro_f1": f1_score(labels, prediction, average="macro"),
        "f1_0": f1_score(labels, prediction, pos_label=0),
        "f1_1": f1_score(labels, prediction, pos_label=1),
        "rows": len(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = pd.read_parquet(args.probabilities)
    validate_probability_frame(scores)
    gold = pd.read_csv(args.gold, dtype={"id": "string", "human_label": "Int8", "uncertain": "int8"})
    labelled = gold[gold["human_label"].notna() & gold["uncertain"].eq(0)].merge(scores[["id", "probability"]], on="id", validate="one_to_one")
    calibration = labelled[labelled["audit_split"].eq("calibration")]
    final = labelled[labelled["audit_split"].eq("final")].copy()
    threshold, calibration_score = best_threshold(calibration["human_label"].to_numpy(), calibration["probability"].to_numpy())
    final["prediction"] = (final["probability"] >= threshold).astype(int)
    final["query_length"] = pd.cut(final["query"].map(query_token_count), [-1, 1, 2, 10_000], labels=["1", "2", "3+"]).astype(str)
    final["top_category"] = final["category"].map(lambda value: category_parts(value)["top_category"])
    rows = [{"slice": "all", "value": "all", "threshold": threshold, "calibration_macro_f1": calibration_score, **metrics(final["human_label"], final["prediction"])}]
    for column in ("query_length", "top_category"):
        for value, group in final.groupby(column):
            if len(group) >= 10 and group["human_label"].nunique() == 2:
                rows.append({"slice": column, "value": value, "threshold": threshold, "calibration_macro_f1": calibration_score, **metrics(group["human_label"], group["prediction"])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
