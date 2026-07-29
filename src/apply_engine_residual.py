#!/usr/bin/env python3
"""Apply structural residual evidence to an arbitrary anchor submission.

Only 0->1 changes are made.  The script writes several explicitly sized local
variants; it never calls the Kaggle API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_anchor(path: Path, sample_path: Path) -> pd.DataFrame:
    anchor = pd.read_csv(path, dtype={"id": "string", "prediction": "int8"})
    sample = pd.read_csv(sample_path, dtype={"id": "string", "prediction": "int8"})
    if list(anchor.columns) != ["id", "prediction"]:
        raise ValueError("anchor must contain exactly id,prediction")
    if len(anchor) != len(sample) or not anchor["id"].equals(sample["id"]):
        raise ValueError("anchor IDs/order do not match sample_submission.csv")
    if not set(anchor["prediction"].unique()).issubset({0, 1}):
        raise ValueError("anchor prediction must be binary")
    return anchor


def selected_ids(residual: pd.DataFrame, *, minimum_agreement: int, top_k: int | None = None) -> pd.Series:
    selected = residual[residual["outside_agreement"].ge(minimum_agreement)].copy()
    selected = selected.sort_values(["forced_confidence", "forced_margin", "id"], ascending=[False, False, True])
    if top_k:
        selected = selected.head(top_k)
    return selected["id"].astype("string")


def apply_positive_flips(anchor: pd.DataFrame, ids: pd.Series) -> tuple[pd.DataFrame, int]:
    output = anchor.copy()
    mask = output["id"].isin(set(ids.astype(str))) & output["prediction"].eq(0)
    output.loc[mask, "prediction"] = np.int8(1)
    return output, int(mask.sum())


def write_variant(
    anchor: pd.DataFrame,
    residual: pd.DataFrame,
    output_dir: Path,
    name: str,
    *,
    minimum_agreement: int,
    top_k: int | None = None,
) -> dict[str, object]:
    ids = selected_ids(residual, minimum_agreement=minimum_agreement, top_k=top_k)
    output, flips = apply_positive_flips(anchor, ids)
    path = output_dir / f"{name}.csv"
    output.to_csv(path, index=False)
    return {
        "variant": name,
        "path": str(path),
        "candidate_ids": int(len(ids)),
        "zero_to_one_flips": flips,
        "positive_rate": float(output["prediction"].mean()),
        "minimum_agreement": minimum_agreement,
        "top_k": top_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--residual", type=Path, default=Path("artifacts/engine_replica/structural_residual_candidates.parquet"))
    parser.add_argument("--sample", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/engine_replica/submission_variants"))
    parser.add_argument("--top-k", type=int, nargs="*", default=[10_000, 30_000, 60_000])
    args = parser.parse_args()

    anchor = load_anchor(args.anchor, args.sample)
    residual = pd.read_parquet(args.residual)
    required = {"id", "outside_agreement", "forced_confidence", "forced_margin"}
    if missing := required - set(residual.columns):
        raise ValueError(f"residual columns missing: {sorted(missing)}")
    if residual["id"].duplicated().any():
        raise ValueError("residual IDs are not unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        write_variant(anchor, residual, args.output_dir, "anchor_plus_residual_agree3", minimum_agreement=3),
        write_variant(anchor, residual, args.output_dir, "anchor_plus_residual_agree2", minimum_agreement=2),
        write_variant(anchor, residual, args.output_dir, "anchor_plus_residual_all", minimum_agreement=1),
    ]
    for top_k in args.top_k:
        reports.append(write_variant(
            anchor, residual, args.output_dir, f"anchor_plus_residual_top{top_k}",
            minimum_agreement=2, top_k=top_k,
        ))
    summary = {
        "anchor": str(args.anchor),
        "anchor_positive_rate": float(anchor["prediction"].mean()),
        "residual_rows": len(residual),
        "variants": reports,
    }
    (args.output_dir / "variant_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
