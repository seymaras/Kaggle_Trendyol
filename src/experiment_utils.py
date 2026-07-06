"""Experiment directories, deterministic folds and artifact validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def experiment_dir(root: Path, experiment_id: str) -> Path:
    if not experiment_id or any(part in experiment_id for part in ("/", "\\", "..")):
        raise ValueError("experiment_id güvenli, tek bir klasör adı olmalıdır")
    path = root / "experiments" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_config(path: Path, config: Any) -> None:
    payload = asdict(config) if is_dataclass(config) else dict(config)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def stable_term_folds(term_ids: pd.Series, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    unique = pd.Series(term_ids.astype("string").dropna().unique(), name="term_id")
    def fold_for(term_id: str) -> int:
        digest = hashlib.blake2b(f"{seed}:{term_id}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little") % n_splits
    return pd.DataFrame({"term_id": unique, "fold": unique.map(fold_for).astype("int8")})


def validate_probability_frame(frame: pd.DataFrame, expected_rows: int | None = None) -> None:
    required = {"id", "term_id", "item_id", "probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Probability kolonları eksik: {sorted(missing)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Satır sayısı {len(frame):,}; beklenen {expected_rows:,}")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Probability id değerleri boş veya duplicate")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise ValueError("Probability değerleri sonlu ve [0, 1] aralığında olmalıdır")


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(frame.columns) != ["id", "prediction"]:
        raise ValueError("Submission kolonları tam olarak id,prediction olmalıdır")
    if len(frame) != len(sample) or not frame["id"].astype(str).equals(sample["id"].astype(str)):
        raise ValueError("Submission satırları sample_submission ile aynı sıra ve id'de değil")
    if frame["id"].duplicated().any() or not set(frame["prediction"].unique()).issubset({0, 1}):
        raise ValueError("Submission duplicate id veya 0/1 dışı prediction içeriyor")

