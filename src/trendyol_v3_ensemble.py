#!/usr/bin/env python3
"""OOF-safe ensemble, experiment logging, teacher selection and final submission."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from trendyol_v3_validation import (
    cross_fit_calibration_and_threshold,
    make_calibrator,
    query_postprocess_predictions,
    select_threshold,
    sigmoid,
)


EnsembleMethod = Literal["average", "weighted", "rank", "geometric", "logistic"]

EXPERIMENT_COLUMNS: tuple[str, ...] = (
    "experiment_id", "model", "backbone", "fold", "seed",
    "train_positive_count", "negative_count", "negative_ratio", "negative_types",
    "max_length", "loss", "learning_rate", "epoch", "validation_type",
    "class_0_f1", "class_1_f1", "macro_f1", "precision", "recall",
    "selected_threshold", "calibration_method", "inference_time",
    "checkpoint_path", "oof_prediction_path", "test_probability_path",
)


def probability_to_logit(values: np.ndarray | pd.Series) -> np.ndarray:
    """Convert probabilities to finite logits without relying on NumPy special functions."""

    probability = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(probability) - np.log1p(-probability)


@dataclass
class ExperimentRecord:
    """One immutable-result experiment row with the required competition fields."""

    experiment_id: str
    model: str
    backbone: str
    fold: int | str
    seed: int
    train_positive_count: int
    negative_count: int
    negative_ratio: float
    negative_types: str
    max_length: int
    loss: str
    learning_rate: float
    epoch: float
    validation_type: str
    class_0_f1: float
    class_1_f1: float
    macro_f1: float
    precision: float
    recall: float
    selected_threshold: float
    calibration_method: str
    inference_time: float
    checkpoint_path: str
    oof_prediction_path: str
    test_probability_path: str

    def validate(self) -> None:
        """Validate IDs, metrics and file-path fields before logging."""

        if not self.experiment_id or "/" in self.experiment_id or ".." in self.experiment_id:
            raise ValueError("experiment_id güvenli tek klasör adı olmalıdır")
        for name in ("class_0_f1", "class_1_f1", "macro_f1", "precision", "recall"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} [0,1] aralığında olmalıdır")
        if not 0 <= self.selected_threshold <= 1:
            raise ValueError("selected_threshold [0,1] aralığında olmalıdır")
        if self.train_positive_count < 0 or self.negative_count < 0 or self.inference_time < 0:
            raise ValueError("Experiment count/time değerleri negatif olamaz")


class ExperimentLogger:
    """Append-safe CSV and JSON experiment registry."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize registry paths and ensure their parent directory exists."""

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "experiments.csv"
        self.json_path = self.output_dir / "experiments.json"

    def append(self, record: ExperimentRecord) -> pd.DataFrame:
        """Append or replace the exact experiment/fold/seed row and persist both formats."""

        record.validate()
        row = asdict(record)
        if tuple(row) != EXPERIMENT_COLUMNS:
            raise RuntimeError("ExperimentRecord kolon sırası zorunlu şemayla uyuşmuyor")
        if self.csv_path.exists():
            frame = pd.read_csv(self.csv_path)
            missing = set(EXPERIMENT_COLUMNS) - set(frame.columns)
            if missing:
                raise ValueError(f"Mevcut experiment log kolonları eksik: {sorted(missing)}")
        else:
            frame = pd.DataFrame(columns=EXPERIMENT_COLUMNS)
        key_mask = (
            frame["experiment_id"].astype(str).eq(str(record.experiment_id))
            & frame["fold"].astype(str).eq(str(record.fold))
            & pd.to_numeric(frame["seed"], errors="coerce").eq(record.seed)
        ) if len(frame) else pd.Series(False, index=frame.index)
        frame = frame.loc[~key_mask]
        frame = (
            pd.DataFrame([row])
            if frame.empty
            else pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        )
        frame = frame[list(EXPERIMENT_COLUMNS)].sort_values(["experiment_id", "fold", "seed"])
        frame.to_csv(self.csv_path, index=False)
        self.json_path.write_text(
            json.dumps(frame.to_dict("records"), ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return frame


def _read_score_frame(path: Path, *, oof: bool = False) -> pd.DataFrame:
    """Read CSV/Parquet scores and retain an explicit probability column."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Score dosyası bulunamadı: {path}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    preferred = (
        "raw_probability", "probability_raw", "probability", "prob", "calibrated_probability"
    )
    probability_col = next((column for column in preferred if column in frame), None)
    if probability_col is None:
        if "raw_logit" not in frame:
            raise ValueError(f"{path} probability veya raw_logit içermiyor")
        frame["_probability"] = sigmoid(pd.to_numeric(frame["raw_logit"], errors="raise").to_numpy())
    else:
        if probability_col == "calibrated_probability" and oof and "fit_folds" not in frame:
            raise ValueError(
                f"{path} yalnız calibrated_probability içeriyor ancak OOF calibration fold lineage yok"
            )
        frame["_probability"] = pd.to_numeric(frame[probability_col], errors="raise")
    if not np.isfinite(frame["_probability"]).all() or not frame["_probability"].between(0, 1).all():
        raise ValueError(f"{path} probability sonlu [0,1] aralığında değil")
    return frame


def align_model_scores(
    score_paths: Mapping[str, Path],
    *,
    oof: bool,
) -> pd.DataFrame:
    """Align model scores one-to-one on pair_uid (OOF) or id (test)."""

    if len(score_paths) < 2:
        raise ValueError("Ensemble en az iki model skoru gerektirir")
    key = "pair_uid" if oof else "id"
    merged: pd.DataFrame | None = None
    reference_pairs: set[tuple[str, str]] | None = None
    reference_metadata: pd.DataFrame | None = None
    for model_key, path in score_paths.items():
        if not model_key or any(character in model_key for character in " /\\"):
            raise ValueError(f"Geçersiz model key: {model_key}")
        frame = _read_score_frame(path, oof=oof)
        required = {key, "term_id", "item_id"}
        if oof:
            required.update({"label", "fold"})
        if missing := required - set(frame.columns):
            raise ValueError(f"{model_key} score kolonları eksik: {sorted(missing)}")
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{model_key} {key} boş veya duplicate")
        pair_set = set(zip(frame["term_id"].astype(str), frame["item_id"].astype(str)))
        if reference_pairs is not None and pair_set != reference_pairs:
            raise ValueError(f"{model_key} farklı candidate universe kullanıyor")
        reference_pairs = pair_set
        base_columns = [key, "term_id", "item_id"]
        if oof:
            base_columns += ["label", "fold"]
        metadata = frame[base_columns].copy()
        for column in base_columns:
            metadata[column] = metadata[column].astype(str)
        metadata = metadata.set_index(key).sort_index()
        if reference_metadata is None:
            reference_metadata = metadata
        elif not metadata.equals(reference_metadata):
            raise ValueError(
                f"{model_key} {key}->term/item/label/fold eşleşmesi referans modelle aynı değil"
            )
        current = frame[base_columns + ["_probability"]].rename(columns={"_probability": f"score__{model_key}"})
        if merged is None:
            merged = current
        else:
            metadata = current[[key, f"score__{model_key}"]]
            merged = merged.merge(metadata, on=key, how="inner", validate="one_to_one", sort=False)
    if merged is None:
        raise RuntimeError("Score alignment boş")
    if len(merged) != len(reference_pairs or ()):
        raise RuntimeError("Score alignment satır kaybetti")
    return merged


def score_correlation_matrix(aligned: pd.DataFrame) -> pd.DataFrame:
    """Return Spearman prediction correlation for ensemble diversity audit."""

    columns = [column for column in aligned if column.startswith("score__")]
    if len(columns) < 2:
        raise ValueError("Correlation için en az iki score kolonu gerekli")
    correlation = aligned[columns].corr(method="spearman")
    correlation.index = [value.removeprefix("score__") for value in correlation.index]
    correlation.columns = [value.removeprefix("score__") for value in correlation.columns]
    return correlation


def _fit_weights(matrix: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Fit non-negative sum-to-one weights by OOF log loss."""

    if matrix.ndim != 2 or matrix.shape[0] != len(label) or matrix.shape[1] < 2:
        raise ValueError("Weighted ensemble matrix shape geçersiz")
    initial = np.full(matrix.shape[1], 1.0 / matrix.shape[1], dtype=np.float64)

    def objective(weights: np.ndarray) -> float:
        """Return OOF log loss for one simplex weight vector."""

        probability = np.clip(matrix @ weights, 1e-7, 1 - 1e-7)
        return float(log_loss(label, probability, labels=[0, 1]))

    result = minimize(
        objective, initial, method="SLSQP",
        bounds=[(0.0, 1.0)] * matrix.shape[1],
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Ensemble weight optimization başarısız: {result.message}")
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def _rank_scores(frame: pd.DataFrame, score_columns: Sequence[str]) -> np.ndarray:
    """Compute query-local percentile rank average without contiguous-group assumptions."""

    ranks = []
    for column in score_columns:
        group = frame.groupby("term_id", sort=False)[column]
        rank = group.rank(method="average", pct=True, ascending=True).to_numpy(dtype=np.float64)
        ranks.append(rank)
    return np.mean(ranks, axis=0)


def _apply_ensemble_method(
    method: EnsembleMethod,
    fit_frame: pd.DataFrame,
    apply_frame: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one ensemble method on fit rows and apply it to held-out rows."""

    x_fit = fit_frame[list(score_columns)].to_numpy(dtype=np.float64)
    x_apply = apply_frame[list(score_columns)].to_numpy(dtype=np.float64)
    label = fit_frame["label"].to_numpy(dtype=np.int8)
    if method == "average":
        return x_apply.mean(axis=1), {"weights": [1 / len(score_columns)] * len(score_columns)}
    if method == "geometric":
        probability = np.exp(np.log(np.clip(x_apply, 1e-7, 1)).mean(axis=1))
        return probability, {}
    if method == "rank":
        return _rank_scores(apply_frame, score_columns), {}
    if method == "weighted":
        weights = _fit_weights(x_fit, label)
        return x_apply @ weights, {"weights": weights.tolist()}
    model = LogisticRegression(C=1.0, max_iter=500, class_weight=None, random_state=seed)
    model.fit(probability_to_logit(x_fit), label)
    probability = model.predict_proba(probability_to_logit(x_apply))[:, 1]
    return probability, {
        "coef": model.coef_.ravel().tolist(), "intercept": model.intercept_.tolist()
    }


def cross_fit_ensembles(
    aligned_oof: pd.DataFrame,
    *,
    methods: Sequence[EnsembleMethod] = ("average", "weighted", "rank", "geometric", "logistic"),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit all ensemble methods on outer OOF folds and report uncalibrated scores."""

    required = {"pair_uid", "term_id", "item_id", "label", "fold"}
    if missing := required - set(aligned_oof.columns):
        raise ValueError(f"Aligned OOF kolonları eksik: {sorted(missing)}")
    score_columns = [column for column in aligned_oof if column.startswith("score__")]
    working = aligned_oof.copy()
    working["fold"] = pd.to_numeric(working["fold"], errors="raise").astype(int)
    folds = sorted(working["fold"].unique())
    if len(folds) < 2:
        raise ValueError("Cross-fit ensemble en az iki fold gerektirir")
    outputs: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for method in methods:
        parts: list[pd.DataFrame] = []
        parameters: list[dict[str, Any]] = []
        for fold in folds:
            fit = working[working["fold"].ne(fold)]
            apply = working[working["fold"].eq(fold)]
            probability, fitted = _apply_ensemble_method(
                method, fit, apply, score_columns, seed=seed + fold
            )
            part = apply[["pair_uid", "term_id", "item_id", "label", "fold"]].copy()
            part["ensemble_method"] = method
            part["raw_probability"] = np.clip(probability, 1e-6, 1 - 1e-6).astype(np.float32)
            part["raw_logit"] = probability_to_logit(part["raw_probability"]).astype(np.float32)
            part["fit_folds"] = ",".join(str(value) for value in folds if value != fold)
            parts.append(part)
            parameters.append({"fold": fold, **fitted})
        result = pd.concat(parts, ignore_index=True)
        if len(result) != len(working) or result["pair_uid"].duplicated().any():
            raise RuntimeError(f"{method} ensemble OOF coverage bozuk")
        _, meta_report = cross_fit_calibration_and_threshold(
            result, methods=("none",), seed=seed, threshold_grid_size=501
        )
        metrics = meta_report.iloc[0].to_dict()
        metrics["threshold"] = None
        metrics.update({"ensemble_method": method, "parameters": json.dumps(parameters, default=str)})
        outputs.append(result)
        reports.append(metrics)
    report = pd.DataFrame(reports).sort_values(["macro_f1", "log_loss"], ascending=[False, True])
    return pd.concat(outputs, ignore_index=True), report.reset_index(drop=True)


def fit_full_ensemble_and_predict(
    aligned_oof: pd.DataFrame,
    aligned_test: pd.DataFrame,
    method: EnsembleMethod,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the selected ensemble on all OOF rows and predict aligned test rows."""

    score_columns = [column for column in aligned_oof if column.startswith("score__")]
    if score_columns != [column for column in aligned_test if column.startswith("score__")]:
        raise ValueError("OOF ve test ensemble model kolonları/sırası uyuşmuyor")
    probability, parameters = _apply_ensemble_method(
        method, aligned_oof, aligned_test, score_columns, seed=seed
    )
    output = aligned_test[["id", "term_id", "item_id"]].copy()
    output["ensemble_method"] = method
    output["raw_probability"] = np.clip(probability, 1e-6, 1 - 1e-6).astype(np.float32)
    output["raw_logit"] = probability_to_logit(output["raw_probability"]).astype(np.float32)
    parameters["score_columns"] = list(score_columns)
    return output, parameters


def calibrate_final_ensemble(
    ensemble_oof: pd.DataFrame,
    ensemble_test: pd.DataFrame,
    *,
    method: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, float, Any]:
    """Fit the chosen calibrator on all OOF only, apply test and select OOF threshold."""

    required_oof = {"raw_logit", "label"}
    if missing := required_oof - set(ensemble_oof.columns):
        raise ValueError(f"Ensemble OOF calibration kolonları eksik: {sorted(missing)}")
    if "raw_logit" not in ensemble_test:
        raise ValueError("Ensemble test raw_logit içermeli")
    calibrator = make_calibrator(method, seed).fit(
        ensemble_oof["raw_logit"].to_numpy(), ensemble_oof["label"].to_numpy()
    )
    oof_probability = calibrator.predict_proba(ensemble_oof["raw_logit"].to_numpy())
    threshold = select_threshold(ensemble_oof["label"].to_numpy(), oof_probability)
    output = ensemble_test.copy()
    output["calibrated_probability"] = calibrator.predict_proba(
        output["raw_logit"].to_numpy()
    ).astype(np.float32)
    return output, threshold, calibrator


def select_teacher_pairs(
    frame: pd.DataFrame,
    *,
    max_rows: int = 250_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Select positives, hard, uncertain, disagreement and test-like rows for 4B scoring."""

    required = {"term_id", "item_id", "label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Teacher selection kolonları eksik: {sorted(missing)}")
    if max_rows < 1:
        raise ValueError("max_rows pozitif olmalıdır")
    out = frame.copy()
    probability_source = (
        out["student_probability"]
        if "student_probability" in out
        else out["probability"] if "probability" in out
        else pd.Series(0.5, index=out.index)
    )
    probability = pd.to_numeric(probability_source, errors="coerce").fillna(0.5)
    hardness = pd.to_numeric(out.get("hardness_score", 0.0), errors="coerce")
    if not isinstance(hardness, pd.Series):
        hardness = pd.Series(float(hardness), index=out.index)
    hardness = hardness.fillna(0.0)
    disagreement = pd.to_numeric(out.get("model_disagreement", 0.0), errors="coerce")
    if not isinstance(disagreement, pd.Series):
        disagreement = pd.Series(float(disagreement), index=out.index)
    disagreement = disagreement.fillna(0.0)
    test_like = out.get("is_test_like", pd.Series(False, index=out.index)).astype(bool)
    out["teacher_reason"] = np.select(
        [
            out["label"].astype(int).eq(1),
            hardness.ge(0.75),
            probability.between(0.35, 0.65),
            disagreement.ge(0.20),
            test_like,
        ],
        ["positive", "hard_negative", "student_uncertain", "model_disagreement", "test_like"],
        default="not_selected",
    )
    selected = out[out["teacher_reason"].ne("not_selected")].copy()
    priority = {"positive": 0, "hard_negative": 1, "student_uncertain": 2, "model_disagreement": 3, "test_like": 4}
    selected["_priority"] = selected["teacher_reason"].map(priority).astype(int)
    selected["_tie"] = [
        int.from_bytes(__import__("hashlib").blake2b(f"{seed}:{term}:{item}".encode(), digest_size=8).digest(), "little")
        for term, item in zip(selected["term_id"], selected["item_id"])
    ]
    selected = selected.sort_values(["_priority", "_tie"]).head(max_rows).drop(columns=["_priority", "_tie"])
    return selected.reset_index(drop=True)


def ablation_table(results: pd.DataFrame) -> pd.DataFrame:
    """Create previous/new Macro-F1 deltas in declared experiment order."""

    required = {"component", "macro_f1"}
    if missing := required - set(results.columns):
        raise ValueError(f"Ablation kolonları eksik: {sorted(missing)}")
    if len(results) < 2:
        raise ValueError("Ablation en az baseline ve bir bileşen gerektirir")
    out = results[["component", "macro_f1"]].copy().reset_index(drop=True)
    out["previous_macro_f1"] = out["macro_f1"].shift(1)
    out["new_macro_f1"] = out["macro_f1"]
    out["difference"] = out["new_macro_f1"] - out["previous_macro_f1"]
    if "inference_time" in results:
        out["inference_time"] = results["inference_time"].to_numpy()
        out["low_roi_warning"] = (
            out["difference"].between(0.0, 0.002, inclusive="both")
            & out["inference_time"].gt(1.5 * out["inference_time"].shift(1))
        )
    return out


def validate_and_write_submission(
    scores: pd.DataFrame,
    sample_submission_path: Path,
    submission_pairs_path: Path,
    output_csv: Path,
    *,
    threshold: float,
    probability_col: str = "calibrated_probability",
    ensure_one_per_query: bool = False,
    top_k: int | None = None,
    contradiction_penalty: float = 0.0,
) -> dict[str, Any]:
    """Align IDs, apply OOF-selected policy, validate binary output and persist probabilities."""

    if not 0 <= threshold <= 1:
        raise ValueError("Submission threshold [0,1] aralığında olmalıdır")
    sample = pd.read_csv(sample_submission_path, dtype={"id": "string"})
    pairs = pd.read_csv(submission_pairs_path, dtype="string")
    if not {"id", "term_id", "item_id"}.issubset(pairs.columns):
        raise ValueError("submission_pairs id,term_id,item_id içermeli")
    if len(sample) != len(pairs) or not sample["id"].astype(str).equals(pairs["id"].astype(str)):
        raise ValueError("Sample submission ve submission_pairs ID sırası uyuşmuyor")
    required = {"id", probability_col}
    if missing := required - set(scores.columns):
        raise ValueError(f"Submission score kolonları eksik: {sorted(missing)}")
    if scores["id"].isna().any() or scores["id"].duplicated().any():
        raise ValueError("Score id boş veya duplicate")
    if len(scores) != len(pairs) or set(scores["id"].astype(str)) != set(pairs["id"].astype(str)):
        raise ValueError("Score ID kümesi submission_pairs ile birebir aynı değil")
    if {"term_id", "item_id"}.issubset(scores.columns):
        pair_metadata = pairs[["id", "term_id", "item_id"]].astype(str).set_index("id").sort_index()
        score_metadata = scores[["id", "term_id", "item_id"]].astype(str).set_index("id").sort_index()
        if not score_metadata.equals(pair_metadata):
            raise ValueError("Score id->term_id/item_id eşleşmesi submission_pairs ile aynı değil")
    aligned = pairs.merge(scores, on="id", how="left", validate="one_to_one", sort=False, suffixes=("", "_score"))
    if len(aligned) != len(sample) or aligned[probability_col].isna().any():
        raise ValueError("Bazı sample ID'leri için score eksik")
    if not aligned["id"].astype(str).equals(sample["id"].astype(str)):
        raise RuntimeError("Score merge sample ID sırasını bozdu")
    probability = pd.to_numeric(aligned[probability_col], errors="raise")
    if not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise ValueError("Final probability NaN/Inf veya [0,1] dışında")
    prediction = query_postprocess_predictions(
        aligned, probability_col=probability_col, threshold=threshold,
        ensure_one=ensure_one_per_query, top_k=top_k,
        contradiction_penalty=contradiction_penalty,
    )
    submission = pd.DataFrame({"id": sample["id"].astype("string"), "prediction": prediction.astype(np.int8)})
    if submission["id"].duplicated().any() or not set(submission["prediction"].unique()).issubset({0, 1}):
        raise RuntimeError("Final submission duplicate ID veya binary olmayan prediction içeriyor")
    if len(submission) != len(sample) or not submission["id"].equals(sample["id"].astype("string")):
        raise RuntimeError("Final submission satır/ID doğrulaması başarısız")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False)
    probability_path = output_csv.with_name(output_csv.stem + "_probabilities.parquet")
    probability_columns = [
        column for column in ("id", "term_id", "item_id", "raw_logit", "raw_probability", probability_col)
        if column in aligned
    ]
    aligned[probability_columns].to_parquet(probability_path, index=False)
    reread = pd.read_csv(output_csv, dtype={"id": "string", "prediction": "int8"})
    if not reread.equals(submission):
        raise RuntimeError("Diske yazılan submission tekrar okunduğunda değişti")
    summary = {
        "path": str(output_csv), "probability_path": str(probability_path),
        "rows": len(submission), "unique_ids": int(submission["id"].nunique()),
        "positive_count": int(submission["prediction"].sum()),
        "positive_rate": float(submission["prediction"].mean()),
        "threshold": threshold,
        "first_rows": submission.head(5).to_dict("records"),
        "last_rows": submission.tail(5).to_dict("records"),
    }
    (output_csv.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_score_specs(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``model=path`` score arguments."""

    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Score spec model=path biçiminde olmalı: {value}")
        key, path = value.split("=", 1)
        if key in output:
            raise ValueError(f"Duplicate model key: {key}")
        output[key] = Path(path)
    return output


def parse_args() -> argparse.Namespace:
    """Parse ensemble fitting or final-submission commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fit", "submit"], required=True)
    parser.add_argument("--oof-scores", nargs="*", default=[])
    parser.add_argument("--test-scores", nargs="*", default=[])
    parser.add_argument("--method", choices=["average", "weighted", "rank", "geometric", "logistic"], default="logistic")
    parser.add_argument("--calibration", choices=["none", "temperature", "platt", "isotonic", "beta"], default="temperature")
    parser.add_argument("--scores", type=Path, help="submit modunda final ensemble Parquet")
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--submission-pairs", type=Path, default=Path("data/submission_pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3/ensemble"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--ensure-one", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Execute OOF ensemble selection or validated final submission."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "submit":
        if args.scores is None or args.threshold is None:
            raise ValueError("submit modu --scores ve OOF'den seçilmiş --threshold gerektirir")
        scores = _read_score_frame(args.scores)
        if "calibrated_probability" not in scores:
            scores["calibrated_probability"] = scores["_probability"]
        validate_and_write_submission(
            scores, args.sample_submission, args.submission_pairs,
            args.output_dir / "final_submission.csv", threshold=args.threshold,
            ensure_one_per_query=args.ensure_one, top_k=args.top_k,
        )
        return
    oof_paths, test_paths = parse_score_specs(args.oof_scores), parse_score_specs(args.test_scores)
    if set(oof_paths) != set(test_paths):
        raise ValueError("OOF ve test model key kümeleri aynı olmalıdır")
    aligned_oof = align_model_scores(oof_paths, oof=True)
    aligned_test = align_model_scores(test_paths, oof=False)
    correlation = score_correlation_matrix(aligned_oof)
    correlation.to_csv(args.output_dir / "oof_prediction_correlations.csv")
    crossfit, report = cross_fit_ensembles(aligned_oof, seed=args.seed)
    crossfit.to_parquet(args.output_dir / "ensemble_crossfit_oof.parquet", index=False)
    report.to_csv(args.output_dir / "ensemble_methods_report.csv", index=False)
    selected_oof = crossfit[crossfit["ensemble_method"].eq(args.method)].copy()
    calibration_oof, calibration_report = cross_fit_calibration_and_threshold(
        selected_oof, methods=(args.calibration,), seed=args.seed
    )
    calibration_oof.to_parquet(args.output_dir / "calibrated_crossfit_oof.parquet", index=False)
    calibration_report.to_csv(args.output_dir / "calibration_report.csv", index=False)
    test_raw, parameters = fit_full_ensemble_and_predict(
        aligned_oof, aligned_test, args.method, seed=args.seed
    )
    test_final, threshold, _ = calibrate_final_ensemble(
        selected_oof, test_raw, method=args.calibration, seed=args.seed
    )
    test_final.to_parquet(args.output_dir / "ensemble_test_probabilities.parquet", index=False)
    (args.output_dir / "ensemble_parameters.json").write_text(
        json.dumps({"method": args.method, "calibration": args.calibration, "threshold": threshold, **parameters}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
