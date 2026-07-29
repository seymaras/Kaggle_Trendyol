#!/usr/bin/env python3
"""Kaydedilmiş (id, prob) olasılıklarından submission üretir.

İki mod:
  ratio  : global sıralamada en yüksek olasılıklı %positive-rate'i 1 yapar
           (bkz. make_ratio_submissions.py).
  hybrid : ratio ile aynı global kesim + her term_id için en az --min-per-query
           pozitif garantisi (adaylar retrieval'dan geldiği için her sorguda
           en az bir alakalı ürün olması yüksek olasılıklı).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_utils import validate_probability_frame, validate_submission  # noqa: E402

ROOT = SRC_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", type=Path, default=ROOT / "artifacts" / "baseline_probabilities.parquet")
    parser.add_argument("--submission-pairs", type=Path, default=ROOT / "data" / "submission_pairs.csv")
    parser.add_argument("--sample-submission", type=Path, default=ROOT / "data" / "sample_submission.csv")
    parser.add_argument("--mode", choices=["ratio", "hybrid"], default="hybrid")
    parser.add_argument("--positive-rate", type=float, default=0.23)
    parser.add_argument("--min-per-query", type=int, default=1, help="hybrid modda sorgu başına minimum pozitif")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_scores(probabilities_path: Path, pairs: pd.DataFrame) -> pd.DataFrame:
    scores = pd.read_parquet(probabilities_path)
    if "prob" in scores.columns and "probability" not in scores.columns:
        scores = scores.rename(columns={"prob": "probability"})
    scores["id"] = scores["id"].astype("string")
    if "term_id" not in scores.columns or "item_id" not in scores.columns:
        scores = pairs.merge(scores[["id", "probability"]], on="id", how="left", validate="one_to_one")
    return scores


def global_ratio_prediction(probability: np.ndarray, ids: np.ndarray, rate: float) -> np.ndarray:
    if not 0 < rate < 1:
        raise ValueError("positive-rate 0 ile 1 arasında olmalıdır")
    n_positive = int(round(len(probability) * rate))
    # id, eşit olasılıklarda deterministik ikincil sıralama anahtarıdır.
    order = np.lexsort((ids.astype(str), -probability.astype(float)))
    prediction = np.zeros(len(probability), dtype=np.int8)
    prediction[order[:n_positive]] = 1
    return prediction


def enforce_min_per_query(
    prediction: np.ndarray, term_ids: np.ndarray, probability: np.ndarray, min_per_query: int
) -> np.ndarray:
    if min_per_query <= 0:
        return prediction
    prediction = prediction.copy()
    frame = pd.DataFrame({"term_id": term_ids, "probability": probability, "prediction": prediction})
    frame["row"] = np.arange(len(frame))
    for _, group in frame.groupby("term_id", sort=False):
        positive_count = int(group["prediction"].sum())
        if positive_count >= min_per_query:
            continue
        needed = min_per_query - positive_count
        top_up = group[group["prediction"].eq(0)].sort_values("probability", ascending=False).head(needed)
        prediction[top_up["row"].to_numpy()] = 1
    return prediction


def main() -> None:
    args = parse_args()
    pairs = pd.read_csv(
        args.submission_pairs, usecols=["id", "term_id", "item_id"],
        dtype={"id": "string", "term_id": "string", "item_id": "string"},
    )
    sample = pd.read_csv(args.sample_submission, dtype={"id": "string", "prediction": "int8"})

    scores = load_scores(args.probabilities, pairs)
    if scores["probability"].isna().any():
        raise RuntimeError("Bazı submission id'leri için olasılık bulunamadı")
    validate_probability_frame(scores, expected_rows=len(sample))

    indexed = scores.set_index("id", verify_integrity=True).loc[sample["id"].astype(str)].reset_index()
    probability = indexed["probability"].to_numpy()
    ids = indexed["id"].astype(str).to_numpy()
    prediction = global_ratio_prediction(probability, ids, args.positive_rate)

    if args.mode == "hybrid":
        prediction = enforce_min_per_query(
            prediction, indexed["term_id"].to_numpy(), probability, args.min_per_query
        )

    result = pd.DataFrame({"id": sample["id"], "prediction": prediction})
    validate_submission(result, sample)

    output = args.output
    if output is None:
        output = ROOT / "artifacts" / f"submission_{args.mode}_rate{args.positive_rate:.2f}_minq{args.min_per_query}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(
        f"{output}: mode={args.mode} hedef_rate={args.positive_rate} "
        f"gerçek_rate={result['prediction'].mean():.6f} satır={len(result):,}"
    )


if __name__ == "__main__":
    main()
