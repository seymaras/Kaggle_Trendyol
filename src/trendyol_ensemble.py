#!/usr/bin/env python3
"""Gold-calibrated ensemble, optional query calibrator and final submissions."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss

from experiment_utils import experiment_dir, validate_probability_frame, validate_submission, write_config
from make_ratio_submissions import prediction_at_ratio
from text_features import query_token_count


def temperature_scale(probability: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    best_temp, best_loss = 1.0, float("inf")
    for temperature in np.arange(0.5, 3.01, 0.05):
        scaled = 1 / (1 + np.exp(-logit / temperature))
        loss = log_loss(labels, scaled, labels=[0, 1])
        if loss < best_loss:
            best_temp, best_loss = float(temperature), float(loss)
    return 1 / (1 + np.exp(-logit / best_temp)), best_temp


def best_threshold(labels: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for threshold in np.arange(0.05, 0.951, 0.005):
        metric = f1_score(labels, score >= threshold, average="macro")
        if metric > best[1]:
            best = float(threshold), float(metric)
    return best


def parse_score(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Score NAME=PATH biçiminde olmalıdır")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs=3, type=parse_score, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--experiment-id", default="ensemble_v1")
    parser.add_argument("--weight-step", type=float, default=0.05)
    args = parser.parse_args()
    out = experiment_dir(args.artifacts_dir, args.experiment_id)
    write_config(out / "config.json", {"scores": [(n, str(p)) for n, p in args.scores], "gold": args.gold, "weight_step": args.weight_step})

    merged = None
    names = []
    for name, path in args.scores:
        score = pd.read_parquet(path)[["id", "term_id", "item_id", "probability"]]
        validate_probability_frame(score)
        score = score.rename(columns={"probability": name})
        merged = score if merged is None else merged.merge(score[["id", name]], on="id", validate="one_to_one")
        names.append(name)
    gold = pd.read_csv(args.gold, dtype={"id": "string", "human_label": "Int8", "uncertain": "int8"})
    labelled = gold[gold["human_label"].notna() & gold["uncertain"].eq(0)].copy()
    if len(labelled) < 1800:
        raise ValueError(f"En az 1800 kesin gold etiket gerekli; bulunan {len(labelled)}")
    evaluation = labelled[["id", "audit_split", "human_label", "query"]].merge(merged, on="id", validate="one_to_one")
    calibration = evaluation[evaluation["audit_split"].eq("calibration")].copy()
    final = evaluation[evaluation["audit_split"].eq("final")].copy()

    temperatures = {}
    for name in names:
        calibrated, temperature = temperature_scale(calibration[name].to_numpy(), calibration["human_label"].to_numpy())
        temperatures[name] = temperature
        logit_all = np.log(np.clip(merged[name], 1e-6, 1 - 1e-6) / np.clip(1 - merged[name], 1e-6, 1))
        merged[f"{name}_cal"] = 1 / (1 + np.exp(-logit_all / temperature))
    evaluation = labelled[["id", "audit_split", "human_label", "query"]].merge(merged, on="id", validate="one_to_one")
    calibration = evaluation[evaluation["audit_split"].eq("calibration")]
    final = evaluation[evaluation["audit_split"].eq("final")]

    units = int(round(1 / args.weight_step))
    best = None
    for parts in itertools.product(range(units + 1), repeat=3):
        if sum(parts) != units:
            continue
        weights = np.array(parts, dtype=float) / units
        score = sum(weights[i] * calibration[f"{name}_cal"].to_numpy() for i, name in enumerate(names))
        threshold, metric = best_threshold(calibration["human_label"].to_numpy(), score)
        candidate = (metric, threshold, weights)
        if best is None or candidate[0] > best[0]:
            best = candidate
    calibration_metric, threshold, weights = best
    merged["ensemble_score"] = sum(weights[i] * merged[f"{name}_cal"] for i, name in enumerate(names))
    final_scored = final[["id", "human_label"]].merge(merged[["id", "ensemble_score"]], on="id")
    global_final_metric = f1_score(final_scored["human_label"], final_scored["ensemble_score"] >= threshold, average="macro")

    merged["score_percentile"] = merged.groupby("term_id")["ensemble_score"].rank(pct=True)
    merged["score_gap_to_top"] = merged.groupby("term_id")["ensemble_score"].transform("max") - merged["ensemble_score"]
    terms = pd.read_csv(args.data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string")
    merged = merged.merge(terms.assign(query_token_count=terms["query"].map(query_token_count))[["term_id", "query_token_count"]], on="term_id", validate="many_to_one")
    query_features = ["ensemble_score", "score_percentile", "score_gap_to_top", "query_token_count"]
    calibration_rows = labelled[labelled["audit_split"].eq("calibration")][["id", "human_label"]].merge(merged[["id", *query_features]], on="id")
    final_rows = labelled[labelled["audit_split"].eq("final")][["id", "human_label"]].merge(merged[["id", *query_features]], on="id")
    calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(calibration_rows[query_features], calibration_rows["human_label"])
    cal_probability = calibrator.predict_proba(calibration_rows[query_features])[:, 1]
    query_threshold, _ = best_threshold(calibration_rows["human_label"].to_numpy(), cal_probability)
    query_final_probability = calibrator.predict_proba(final_rows[query_features])[:, 1]
    query_final_metric = f1_score(final_rows["human_label"], query_final_probability >= query_threshold, average="macro")
    use_query_calibrator = query_final_metric >= global_final_metric + 0.005
    if use_query_calibrator:
        merged["final_score"] = calibrator.predict_proba(merged[query_features])[:, 1]
        final_threshold = query_threshold
    else:
        merged["final_score"] = merged["ensemble_score"]
        final_threshold = threshold

    sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"id": "string", "prediction": "int8"})
    ordered = sample[["id"]].merge(merged[["id", "term_id", "item_id", "final_score"]], on="id", validate="one_to_one")
    final_scores = ordered.rename(columns={"final_score": "probability"})
    validate_probability_frame(final_scores, len(sample))
    final_scores.to_parquet(out / "probabilities.parquet", index=False)
    submission = pd.DataFrame({"id": sample["id"], "prediction": (ordered["final_score"] >= final_threshold).astype(np.int8)})
    validate_submission(submission, sample)
    submission.to_csv(out / "submission.csv", index=False)
    for ratio in (0.23, 0.24, 0.25):
        ratio_submission = pd.DataFrame({
            "id": sample["id"],
            "prediction": prediction_at_ratio(ordered["final_score"].to_numpy(), ordered["id"].to_numpy(), ratio),
        })
        validate_submission(ratio_submission, sample)
        ratio_submission.to_csv(out / f"submission_ratio_{ratio:.2f}.csv", index=False)
    metrics = {
        "temperatures": temperatures, "weights": dict(zip(names, weights.tolist())),
        "calibration_macro_f1": calibration_metric, "global_final_macro_f1": global_final_metric,
        "query_final_macro_f1": query_final_metric, "use_query_calibrator": use_query_calibrator,
        "threshold": final_threshold, "positive_ratio": float(submission["prediction"].mean()),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
