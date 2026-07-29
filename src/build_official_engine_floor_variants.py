#!/usr/bin/env python3
"""Build v6 floor variants from the full official-domain engine residual."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_anchor(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=["id", "prediction"])
        frame["id"] = frame["id"].astype("string")
        frame["prediction"] = frame["prediction"].astype("int8")
        return frame
    return pd.read_csv(path, dtype={"id": "string", "prediction": "int8"})


def load_relevance_score(
    sources: list[list[str]], anchor_ids: pd.Series,
) -> pd.Series:
    if not sources:
        return pd.Series(np.zeros(len(anchor_ids), dtype=np.float32), index=anchor_ids.astype(str))
    total = np.zeros(len(anchor_ids), dtype=np.float32)
    for raw_path, column in sources:
        path = Path(raw_path)
        frame = pd.read_parquet(path, columns=["id", column])
        if len(frame) != len(anchor_ids):
            raise ValueError(f"relevance row count mismatch: {path}")
        if frame["id"].astype("string").equals(anchor_ids):
            values = frame[column]
        else:
            values = frame.set_index("id")[column].reindex(anchor_ids).reset_index(drop=True)
            if values.isna().any():
                raise ValueError(f"relevance IDs missing: {path}")
        total += values.rank(pct=True).to_numpy(dtype=np.float32)
        del frame, values
        gc.collect()
    return pd.Series(total / len(sources), index=anchor_ids.astype(str))


def load_precomputed_relevance(path: Path) -> pd.Series:
    frame = pd.read_parquet(path, columns=["id", "relevance_score"])
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError(f"duplicate relevance IDs: {path}")
    if frame["relevance_score"].isna().any():
        raise ValueError(f"missing relevance scores: {path}")
    return frame.set_index("id")["relevance_score"].astype(np.float32)


def select_rows(
    residual: pd.DataFrame,
    term_stats: pd.DataFrame,
    certainty_min: float,
) -> tuple[set[str], dict[str, int]]:
    eligible = term_stats[
        term_stats["deficit"].gt(0) & term_stats["certainty"].ge(certainty_min)
    ]
    selected: set[str] = set()
    agreement_counts: dict[str, int] = {}
    for term_id, stats in eligible.iterrows():
        need = int(stats["deficit"])
        candidates = residual[
            residual["term_id"].eq(term_id) & residual["anchor_prediction"].eq(0)
        ].sort_values(
            ["outside_agreement", "relevance_score", "forced_confidence", "forced_margin", "id"],
            ascending=[False, False, False, False, True],
        ).head(need)
        if len(candidates) != need:
            raise RuntimeError(f"{term_id}: floor needs {need}, residual has {len(candidates)} anchor-zero rows")
        for agreement, count in candidates["outside_agreement"].value_counts().items():
            key = str(int(agreement))
            agreement_counts[key] = agreement_counts.get(key, 0) + int(count)
        selected.update(candidates["id"].astype(str))
    return selected, agreement_counts


def write_variant(anchor: pd.DataFrame, ids: set[str], path: Path) -> dict[str, object]:
    output = anchor.copy()
    mask = output["id"].isin(ids) & output["prediction"].eq(0)
    output.loc[mask, "prediction"] = np.int8(1)
    output.to_csv(path, index=False)
    return {
        "file": path.name,
        "zero_to_one_flips": int(mask.sum()),
        "one_to_zero_flips": 0,
        "positive_rate": float(output["prediction"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument(
        "--relevance-source", nargs=2, action="append", default=[],
        metavar=("PARQUET", "COLUMN"),
    )
    parser.add_argument(
        "--precomputed-relevance", type=Path,
        help="Slim id/relevance_score parquet computed from full global ranks.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    anchor = read_anchor(args.anchor)
    pairs = pd.read_csv(args.pairs, usecols=["id", "term_id"], dtype="string")
    if len(anchor) != len(pairs) or not anchor["id"].equals(pairs["id"]):
        raise ValueError("anchor IDs/order do not match submission pairs")
    frame = pairs.copy()
    frame["prediction"] = anchor["prediction"].to_numpy(dtype=np.int8)
    term_stats = frame.groupby("term_id", sort=False)["prediction"].agg(["size", "sum"])
    term_stats["floor"] = (term_stats["size"] - 100).clip(lower=0)
    term_stats["deficit"] = (term_stats["floor"] - term_stats["sum"]).clip(lower=0)
    term_stats["zeros"] = term_stats["size"] - term_stats["sum"]
    term_stats["certainty"] = term_stats["deficit"] / term_stats["zeros"].clip(lower=1)

    residual = pd.read_parquet(args.residual)
    required = {
        "id", "term_id", "outside_agreement", "forced_confidence", "forced_margin",
    }
    if missing := required - set(residual.columns):
        raise ValueError(f"engine residual columns missing: {sorted(missing)}")
    anchor_map = anchor.set_index("id")["prediction"]
    residual["anchor_prediction"] = residual["id"].map(anchor_map).astype(np.int8)
    if args.precomputed_relevance and args.relevance_source:
        raise ValueError("use either --precomputed-relevance or --relevance-source")
    if args.precomputed_relevance:
        relevance = load_precomputed_relevance(args.precomputed_relevance)
    else:
        relevance = load_relevance_score(args.relevance_source, anchor["id"])
    residual["relevance_score"] = residual["id"].astype(str).map(relevance)
    if residual["relevance_score"].isna().any():
        missing = int(residual["relevance_score"].isna().sum())
        raise ValueError(f"relevance score missing for {missing} engine residual rows")
    residual["relevance_score"] = residual["relevance_score"].astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("official_v6_floor_highcert_6008.csv", 0.50),
        ("official_v6_floor_midcert_8304.csv", 0.40),
        ("official_v6_floor_full_18022.csv", 0.00),
    ]
    variants = []
    for filename, certainty in specs:
        ids, agreements = select_rows(residual, term_stats, certainty)
        report = write_variant(anchor, ids, args.output_dir / filename)
        report.update({"certainty_min": certainty, "agreement_counts": agreements})
        variants.append(report)
    report = {
        "engine_residual_rows": len(residual),
        "groups_below_floor": int(term_stats["deficit"].gt(0).sum()),
        "total_floor_deficit": int(term_stats["deficit"].sum()),
        "relevance_models": 7 if args.precomputed_relevance else len(args.relevance_source),
        "relevance_mode": "precomputed_global_rank" if args.precomputed_relevance else "full_sources",
        "variants": variants,
        "competition_submission_called": False,
    }
    (args.output_dir / "official_floor_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
