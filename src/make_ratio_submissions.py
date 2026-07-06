#!/usr/bin/env python3
"""Create deterministic global-ratio submissions from saved probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import validate_probability_frame, validate_submission


def prediction_at_ratio(probability: np.ndarray, ids: np.ndarray, ratio: float) -> np.ndarray:
    if not 0 < ratio < 1:
        raise ValueError("ratio 0 ile 1 arasında olmalıdır")
    n_positive = int(round(len(probability) * ratio))
    # id is a deterministic secondary key for exact tie handling.
    order = np.lexsort((ids.astype(str), -probability.astype(float)))
    prediction = np.zeros(len(probability), dtype=np.int8)
    prediction[order[:n_positive]] = 1
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.22, 0.24, 0.26])
    args = parser.parse_args()

    scores = pd.read_parquet(args.probabilities)
    sample = pd.read_csv(args.sample_submission, dtype={"id": "string", "prediction": "int8"})
    validate_probability_frame(scores, expected_rows=len(sample))
    indexed = scores.set_index("id", verify_integrity=True).loc[sample["id"].astype(str)].reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ratio in args.ratios:
        result = pd.DataFrame({
            "id": sample["id"],
            "prediction": prediction_at_ratio(
                indexed["probability"].to_numpy(), indexed["id"].astype(str).to_numpy(), ratio
            ),
        })
        validate_submission(result, sample)
        path = args.output_dir / f"submission_ratio_{ratio:.3f}.csv"
        result.to_csv(path, index=False)
        print(f"{path}: positive_ratio={result.prediction.mean():.6f}")


if __name__ == "__main__":
    main()
