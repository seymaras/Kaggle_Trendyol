#!/usr/bin/env python3
"""Optuna search for CatBoost ranker parameters on fixed query folds.

This is intentionally a tuner only: it writes parameters and never creates a
submission. Run it after the first YetiRankPairwise/PairLogit A/B comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import stable_term_folds
from trendyol_train_catboost_ranker import build_pair_features, normalize_items, require_catboost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna gerekli: pip install -r requirements-gpu.txt") from exc
    CatBoostRanker, Pool = require_catboost()

    train = pd.read_parquet(args.train)
    items = normalize_items(pd.read_csv(args.data_dir / "items.csv", dtype="string"))
    features, names = build_pair_features(train, items)
    fold_map = stable_term_folds(train["term_id"], 5, args.seed).set_index("term_id")["fold"]
    valid_idx = np.flatnonzero(train["term_id"].map(fold_map).to_numpy() == 0)
    train_idx = np.flatnonzero(train["term_id"].map(fold_map).to_numpy() != 0)
    train_idx = train_idx[np.argsort(train.iloc[train_idx]["term_id"].astype(str).to_numpy(), kind="stable")]
    valid_idx = valid_idx[np.argsort(train.iloc[valid_idx]["term_id"].astype(str).to_numpy(), kind="stable")]
    train_groups = pd.factorize(train.iloc[train_idx]["term_id"].astype(str), sort=True)[0]
    valid_groups = pd.factorize(train.iloc[valid_idx]["term_id"].astype(str), sort=True)[0]
    train_pool = Pool(features.iloc[train_idx][names], label=train.iloc[train_idx]["label"], group_id=train_groups, weight=train.iloc[train_idx].get("sample_weight"))
    valid_pool = Pool(features.iloc[valid_idx][names], label=train.iloc[valid_idx]["label"], group_id=valid_groups, weight=train.iloc[valid_idx].get("sample_weight"))

    def objective(trial):
        model = CatBoostRanker(
            loss_function=trial.suggest_categorical("loss_function", ["YetiRankPairwise", "PairLogit"]),
            iterations=trial.suggest_int("iterations", 500, 1600, step=100),
            depth=trial.suggest_int("depth", 6, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            random_seed=args.seed + trial.number, verbose=False,
            task_type=args.task_type, devices=args.devices,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        score = model.get_best_score().get("validation", {}).get("NDCG:top=20", float("nan"))
        if not np.isfinite(score):
            score = model.get_best_score().get("validation", {}).get("NDCG", 0.0)
        return float(score)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2) + "\n")
    print(json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2))


if __name__ == "__main__":
    main()
