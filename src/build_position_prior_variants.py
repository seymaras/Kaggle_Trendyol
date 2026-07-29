#!/usr/bin/env python3
"""Calibrate v2 probabilities with the hidden candidate-position prior.

The row stream preserves each term's occurrence order even though terms are
interleaved.  Training retrieval candidates estimate a monotone rank prior.
Several local-only variants are written so the prior strength remains an
explicit ablation rather than an untracked rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BIN_EDGES = np.array([0, 1, 3, 5, 10, 20, 50, 100], dtype=np.int32)


def learn_rank_log_odds(train: pd.DataFrame) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    observed = train[train["retrieval_rank"].between(1, 100)].copy()
    observed["rank_bin"] = pd.cut(observed["retrieval_rank"], BIN_EDGES, labels=False)
    global_rate = float(observed["label"].mean())
    rows = []
    bin_log_odds = np.zeros(len(BIN_EDGES) - 1, dtype=np.float32)
    global_odds = global_rate / (1.0 - global_rate)
    for bin_index in range(len(bin_log_odds)):
        group = observed[observed["rank_bin"].eq(bin_index)]
        positives = float(group["label"].sum())
        rate = (positives + 20.0 * global_rate) / (len(group) + 20.0)
        odds = rate / (1.0 - rate)
        bin_log_odds[bin_index] = np.log(odds / global_odds)
        rows.append({
            "rank_min": int(BIN_EDGES[bin_index] + 1),
            "rank_max": int(BIN_EDGES[bin_index + 1]),
            "rows": int(len(group)),
            "known_positive_rate": float(rate),
            "relative_log_odds": float(bin_log_odds[bin_index]),
        })
    return bin_log_odds, global_rate, rows


def rank_adjusted_probability(
    probability: np.ndarray,
    position: np.ndarray,
    bin_log_odds: np.ndarray,
    alpha: float,
) -> np.ndarray:
    clipped = np.clip(probability.astype(np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    top100 = position <= 100
    bin_index = np.searchsorted(BIN_EDGES[1:], position[top100], side="left")
    logit[top100] += alpha * bin_log_odds[bin_index]
    adjusted = 1.0 / (1.0 + np.exp(-logit))
    adjusted[~top100] = 1.0
    return adjusted.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/submission_pairs.csv"))
    parser.add_argument(
        "--probabilities", type=Path,
        default=Path("artifacts/experiments/classical_v2/probabilities.parquet"),
    )
    parser.add_argument(
        "--anchor", type=Path,
        default=Path("artifacts/experiments/classical_v2/submission.csv"),
    )
    parser.add_argument(
        "--retrieval-train", type=Path, default=Path("artifacts/train_testlike.parquet"),
    )
    parser.add_argument("--threshold", type=float, default=0.69)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/position_prior_variants"),
    )
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs, usecols=["id", "term_id"], dtype="string")
    probability = pd.read_parquet(args.probabilities, columns=["id", "probability"])
    anchor = pd.read_csv(args.anchor, dtype={"id": "string", "prediction": "int8"})
    if not probability["id"].astype("string").equals(pairs["id"]):
        raise ValueError("probability IDs/order do not match submission pairs")
    if not anchor["id"].equals(pairs["id"]):
        raise ValueError("anchor IDs/order do not match submission pairs")
    position = pairs.groupby("term_id", sort=False).cumcount().add(1).to_numpy(dtype=np.int32)
    train = pd.read_parquet(args.retrieval_train, columns=["retrieval_rank", "label"])
    bin_log_odds, global_rate, prior_table = learn_rank_log_odds(train)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = []
    direct = anchor.copy()
    direct_mask = position > 100
    direct.loc[direct_mask, "prediction"] = np.int8(1)
    direct_path = args.output_dir / "anchor_plus_position_gt100.csv"
    direct.to_csv(direct_path, index=False)
    variants.append({
        "variant": direct_path.stem,
        "changes": int((direct["prediction"] != anchor["prediction"]).sum()),
        "zero_to_one": int((direct["prediction"].gt(anchor["prediction"])).sum()),
        "one_to_zero": 0,
        "positive_rate": float(direct["prediction"].mean()),
    })

    base_probability = probability["probability"].to_numpy(dtype=np.float32)
    for alpha in args.alphas:
        adjusted = rank_adjusted_probability(base_probability, position, bin_log_odds, alpha)
        prediction = (adjusted >= args.threshold).astype(np.int8)
        output = pd.DataFrame({"id": pairs["id"], "prediction": prediction})
        name = f"position_prior_alpha{alpha:.2f}".replace(".", "p")
        output.to_csv(args.output_dir / f"{name}.csv", index=False)
        pd.DataFrame({
            "id": pairs["id"], "candidate_position": position,
            "base_probability": base_probability, "adjusted_probability": adjusted,
        }).to_parquet(args.output_dir / f"{name}_probabilities.parquet", index=False)
        variants.append({
            "variant": name,
            "alpha": alpha,
            "changes": int((prediction != anchor["prediction"].to_numpy()).sum()),
            "zero_to_one": int(((prediction == 1) & anchor["prediction"].eq(0).to_numpy()).sum()),
            "one_to_zero": int(((prediction == 0) & anchor["prediction"].eq(1).to_numpy()).sum()),
            "positive_rate": float(prediction.mean()),
        })

    report = {
        "threshold": args.threshold,
        "retrieval_known_positive_rate_top100": global_rate,
        "rank_prior": prior_table,
        "position_gt100_rows": int(direct_mask.sum()),
        "position_gt100_anchor_positive_rate": float(anchor.loc[direct_mask, "prediction"].mean()),
        "variants": variants,
        "competition_submission_called": False,
    }
    (args.output_dir / "position_prior_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
