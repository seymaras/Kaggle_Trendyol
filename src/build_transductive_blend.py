#!/usr/bin/env python3
"""Select a v2/transductive blend only on original-label OOF rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("artifacts/experiments/classical_v2"))
    parser.add_argument("--transductive-dir", type=Path, default=Path("artifacts/experiments/classical_transductive_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transductive_blend"))
    args = parser.parse_args()

    base = pd.read_parquet(args.base_dir / "oof_probabilities.parquet")
    trans = pd.read_parquet(args.transductive_dir / "oof_probabilities.parquet")
    if "is_transductive" in trans:
        trans = trans[~trans["is_transductive"]].reset_index(drop=True)
    keys = ["term_id", "item_id", "label"]
    if not base[keys].astype(str).equals(trans[keys].astype(str)):
        raise ValueError("base/transductive OOF rows do not align")
    y = base["label"].to_numpy(dtype=np.int8)
    p_base = base["probability"].to_numpy(dtype=np.float32)
    p_trans = trans["probability"].to_numpy(dtype=np.float32)
    grid = []
    for weight in np.arange(0.0, 1.0001, 0.05):
        probability = (1.0 - weight) * p_base + weight * p_trans
        for threshold in np.arange(0.60, 0.8001, 0.005):
            score = f1_score(y, probability >= threshold, average="macro")
            grid.append({"weight": float(weight), "threshold": float(threshold), "macro_f1": float(score)})
    grid_frame = pd.DataFrame(grid).sort_values(
        ["macro_f1", "weight"], ascending=[False, True], ignore_index=True,
    )
    best = grid_frame.iloc[0]

    base_test = pd.read_parquet(args.base_dir / "probabilities.parquet")
    trans_test = pd.read_parquet(args.transductive_dir / "probabilities.parquet")
    keys = ["id", "term_id", "item_id"]
    if not base_test[keys].astype(str).equals(trans_test[keys].astype(str)):
        raise ValueError("base/transductive test rows do not align")
    probability = (
        (1.0 - float(best["weight"])) * base_test["probability"].to_numpy(dtype=np.float32)
        + float(best["weight"]) * trans_test["probability"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    prediction = (probability >= float(best["threshold"])).astype(np.int8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id": base_test["id"], "term_id": base_test["term_id"], "item_id": base_test["item_id"],
        "probability": probability,
    }).to_parquet(args.output_dir / "probabilities.parquet", index=False)
    pd.DataFrame({"id": base_test["id"], "prediction": prediction}).to_csv(
        args.output_dir / "submission.csv", index=False,
    )
    grid_frame.to_csv(args.output_dir / "oof_blend_grid.csv", index=False)
    base_submission = pd.read_csv(args.base_dir / "submission.csv", usecols=["prediction"])
    report = {
        "best_weight": float(best["weight"]),
        "best_threshold": float(best["threshold"]),
        "best_oof_macro_f1": float(best["macro_f1"]),
        "base_oof_macro_f1_at_grid": float(grid_frame[grid_frame["weight"].eq(0)].iloc[0]["macro_f1"]),
        "positive_rate": float(prediction.mean()),
        "changes_vs_base": int((prediction != base_submission["prediction"].to_numpy()).sum()),
        "competition_submission_called": False,
    }
    (args.output_dir / "blend_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
