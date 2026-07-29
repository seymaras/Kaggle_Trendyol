#!/usr/bin/env python3
"""Apply order/lexical structural evidence to an anchor with 0->1 flips only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from apply_engine_residual import load_anchor


def write_variant(anchor: pd.DataFrame, ids: pd.Series, path: Path) -> dict[str, object]:
    output = anchor.copy()
    mask = output["id"].isin(set(ids.astype(str))) & output["prediction"].eq(0)
    output.loc[mask, "prediction"] = np.int8(1)
    output.to_csv(path, index=False)
    return {
        "variant": path.stem,
        "evidence_ids": int(ids.nunique()),
        "zero_to_one_flips": int(mask.sum()),
        "positive_rate": float(output["prediction"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument(
        "--pseudolabels", type=Path,
        default=Path("artifacts/structural_pseudolabels/test_structural_pseudopositives.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/structural_pseudolabels/anchor_variants"),
    )
    args = parser.parse_args()

    anchor = load_anchor(args.anchor, args.sample)
    pseudo = pd.read_parquet(args.pseudolabels)
    required = {"id", "is_structural_residual", "is_order_residual", "anchor_prediction", "secondary_prediction"}
    if missing := required - set(pseudo.columns):
        raise ValueError(f"pseudolabel columns missing: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks = {
        "anchor_plus_structure_intersection": pseudo["is_structural_residual"] & pseudo["is_order_residual"],
        "anchor_plus_fast_structure": pseudo["is_structural_residual"],
        "anchor_plus_order_gt100": pseudo["is_order_residual"],
        "anchor_plus_structure_union": pseudo["is_structural_residual"] | pseudo["is_order_residual"],
        "anchor_plus_structure_secondary": (
            (pseudo["is_structural_residual"] | pseudo["is_order_residual"])
            & pseudo["secondary_prediction"].eq(1)
        ),
    }
    variants = [
        write_variant(anchor, pseudo.loc[mask, "id"], args.output_dir / f"{name}.csv")
        for name, mask in masks.items()
    ]
    report = {
        "anchor": str(args.anchor),
        "anchor_positive_rate": float(anchor["prediction"].mean()),
        "variants": variants,
        "competition_submission_called": False,
    }
    (args.output_dir / "structural_variant_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
