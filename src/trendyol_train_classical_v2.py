#!/usr/bin/env python3
"""Train LightGBM classical v2 and run resumable two-pass inference."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from classical_features import FieldedTfidf, add_query_level_features, build_classical_features, model_feature_names
from experiment_utils import experiment_dir, stable_term_folds, validate_probability_frame, validate_submission, write_config
from text_features import category_parts, normalize_category_path, normalize_text

ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]


def require_lightgbm():
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    except ImportError as exc:
        raise RuntimeError("LightGBM gerekli: pip install -r requirements-gpu.txt") from exc
    return LGBMClassifier, early_stopping, log_evaluation


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["category"] = frame["category"].fillna("").map(normalize_category_path)
    for col in ["query", "title", "brand", "gender", "age_group", "attributes"]:
        frame[col] = frame[col].fillna("").map(normalize_text)
    frame["top_category"] = frame["category"].map(lambda x: category_parts(x)["top_category"])
    frame["leaf_category"] = frame["category"].map(lambda x: category_parts(x)["leaf_category"])
    return frame


def make_model(seed: int, n_estimators: int = 1200):
    LGBMClassifier, _, _ = require_lightgbm()
    return LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.03, num_leaves=63,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, objective="binary", random_state=seed, n_jobs=-1,
    )


def tune_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for threshold in np.arange(0.1, 0.901, 0.01):
        score = f1_score(y, p >= threshold, average="macro")
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def enrich_pairs(pairs: pd.DataFrame, terms: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    return normalize_frame(pairs.merge(terms, on="term_id", how="left", validate="many_to_one").merge(items, on="item_id", how="left", validate="many_to_one"))


def train(args, out: Path):
    train = pd.read_parquet(args.train)
    train = normalize_frame(train)
    train.loc[train["label"].eq(0), "sample_weight"] *= args.negative_weight_multiplier
    items = normalize_frame(pd.read_csv(args.data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype="string").assign(query=""))
    fielded = FieldedTfidf(args.tfidf_max_features).fit(items)
    base = build_classical_features(train, fielded)
    meta = train[["term_id", "item_id", "label", "sample_weight", "top_category", "leaf_category"]].reset_index(drop=True)
    features = add_query_level_features(pd.concat([meta, base.reset_index(drop=True)], axis=1))
    names = model_feature_names(features)
    folds = stable_term_folds(train["term_id"], args.n_splits, args.seed)
    fold_map = folds.set_index("term_id")["fold"]
    row_fold = train["term_id"].map(fold_map).to_numpy()
    oof = np.zeros(len(train), dtype=np.float32)
    reports = []
    _, early_stopping, log_evaluation = require_lightgbm()
    for fold in range(args.n_splits):
        tr, va = np.flatnonzero(row_fold != fold), np.flatnonzero(row_fold == fold)
        model = make_model(args.seed + fold)
        model.fit(
            features.iloc[tr][names], train.iloc[tr]["label"], sample_weight=train.iloc[tr]["sample_weight"],
            eval_set=[(features.iloc[va][names], train.iloc[va]["label"])], eval_metric="binary_logloss",
            eval_sample_weight=[train.iloc[va]["sample_weight"]],
            callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
        )
        oof[va] = model.predict_proba(features.iloc[va][names])[:, 1]
        threshold, score = tune_threshold(train.iloc[va]["label"].to_numpy(), oof[va])
        reports.append({
            "fold": fold, "rows": len(va), "macro_f1": score,
            "threshold": threshold, "best_iteration": model.best_iteration_,
            "positive_prediction_rate": float((oof[va] >= threshold).mean()),
        })
    threshold, score = tune_threshold(train["label"].to_numpy(), oof)
    reports.append({
        "fold": "oof", "rows": len(train), "macro_f1": score,
        "threshold": threshold,
        "best_iteration": int(np.median([r["best_iteration"] for r in reports])),
        "positive_prediction_rate": float((oof >= threshold).mean()),
    })
    final_model = make_model(args.seed, int(reports[-1]["best_iteration"] * 1.1))
    final_model.fit(features[names], train["label"], sample_weight=train["sample_weight"])
    pd.DataFrame(reports).to_csv(out / "cv_report.csv", index=False)
    folds.to_parquet(out / "term_folds.parquet", index=False)
    pd.DataFrame({"term_id": train["term_id"], "item_id": train["item_id"], "label": train["label"], "probability": oof}).to_parquet(out / "oof_probabilities.parquet", index=False)
    joblib.dump({"model": final_model, "fielded_tfidf": fielded, "features": names, "threshold": threshold}, out / "model.joblib")
    return final_model, fielded, names, threshold, items


def inference(args, out: Path, model, fielded, names, threshold, items):
    pairs = pd.read_csv(args.data_dir / "submission_pairs.csv", dtype="string")
    terms = pd.read_csv(args.data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string")
    raw_dir = out / "feature_shards"
    raw_dir.mkdir(exist_ok=True)
    shard_paths = []
    for shard, start in enumerate(range(0, len(pairs), args.chunk_size)):
        path = raw_dir / f"part-{shard:05d}.parquet"
        shard_paths.append(path)
        if path.exists():
            continue
        chunk = pairs.iloc[start:start + args.chunk_size]
        enriched = enrich_pairs(chunk, terms, items[ITEM_COLUMNS])
        base = build_classical_features(enriched, fielded)
        result = pd.concat([
            enriched[["id", "term_id", "item_id", "top_category", "leaf_category"]].reset_index(drop=True),
            base.reset_index(drop=True),
        ], axis=1)
        result.to_parquet(path, index=False)
        del enriched, base, result
        gc.collect()
    all_features = pd.concat([pd.read_parquet(path) for path in shard_paths], ignore_index=True)
    all_features = add_query_level_features(all_features)
    probability = model.predict_proba(all_features[names])[:, 1].astype(np.float32)
    scores = all_features[["id", "term_id", "item_id"]].copy()
    scores["probability"] = probability
    validate_probability_frame(scores, len(pairs))
    scores.to_parquet(out / "probabilities.parquet", index=False)
    submission = pd.DataFrame({"id": scores["id"], "prediction": (probability >= threshold).astype(np.int8)})
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"id": "string", "prediction": "int8"})
    validate_submission(submission, sample)
    submission.to_csv(out / "submission.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "train-infer"], default="train-infer")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--train", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--experiment-id", default="classical_v3")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tfidf-max-features", type=int, default=40_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--negative-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    if args.skip_inference:
        args.mode = "train"
    if args.mode in {"train", "train-infer"} and args.train is None:
        parser.error("--train eğitim modunda zorunludur")
    out = experiment_dir(args.artifacts_dir, args.experiment_id)
    config_name = "inference_config.json" if args.mode == "infer" else "config.json"
    write_config(out / config_name, vars(args))
    trained = None
    if args.mode in {"train", "train-infer"}:
        trained = train(args, out)
    if args.mode == "train-infer":
        inference(args, out, *trained)
    elif args.mode == "infer":
        model_path = args.model_path or out / "model.joblib"
        bundle = joblib.load(model_path)
        items = normalize_frame(
            pd.read_csv(args.data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype="string").assign(query="")
        )
        inference(
            args, out, bundle["model"], bundle["fielded_tfidf"],
            bundle["features"], bundle["threshold"], items,
        )


if __name__ == "__main__":
    main()
