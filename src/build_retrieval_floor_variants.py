#!/usr/bin/env python3
"""Fill the provable n-100 positive floor using retrieval-miss evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_submission(anchor: pd.DataFrame, selected: set[str], path: Path) -> dict[str, object]:
    output = anchor.copy()
    mask = output["id"].isin(selected) & output["prediction"].eq(0)
    output.loc[mask, "prediction"] = np.int8(1)
    output.to_csv(path, index=False)
    return {
        "variant": path.stem,
        "selected_ids": len(selected),
        "zero_to_one_flips": int(mask.sum()),
        "positive_rate": float(output["prediction"].mean()),
    }


def select_floor_rows(
    candidates: pd.DataFrame,
    term_stats: pd.DataFrame,
    *,
    certainty_min: float,
    evidence: str,
    use_relevance: bool = False,
) -> tuple[set[str], dict[str, int]]:
    eligible = term_stats[
        term_stats["deficit"].gt(0) & term_stats["certainty"].ge(certainty_min)
    ]
    selected: set[str] = set()
    tier_counts: dict[str, int] = {}
    for term_id, stats in eligible.iterrows():
        need = int(stats["deficit"])
        group = candidates[
            candidates["term_id"].eq(term_id) & candidates["anchor_prediction_current"].eq(0)
        ].copy()
        if evidence == "intersection":
            group = group[group["is_structural_residual"] & group["is_order_residual"]]
        elif evidence == "fast":
            group = group[group["is_structural_residual"]]
        elif evidence != "union":
            raise ValueError(evidence)
        sort_columns = ["structure_tier"]
        ascending = [False]
        if use_relevance:
            sort_columns.append("relevance_score")
            ascending.append(False)
        sort_columns.extend(["retrieval_margin", "fast_retrieval_score", "id"])
        ascending.extend([False, True, True])
        group = group.sort_values(sort_columns, ascending=ascending).head(need)
        for tier, count in group["structure_tier_name"].value_counts().items():
            tier_counts[str(tier)] = tier_counts.get(str(tier), 0) + int(count)
        selected.update(group["id"].astype(str))
    return selected, tier_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=Path("data/submission_pairs.csv"))
    parser.add_argument(
        "--pseudolabels", type=Path,
        default=Path("artifacts/structural_pseudolabels/test_structural_pseudopositives.parquet"),
    )
    parser.add_argument("--compare", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--relevance-source", nargs=2, action="append", default=[],
        metavar=("PARQUET", "COLUMN"),
        help="Repeat for each aligned relevance model; percentile ranks are averaged.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/retrieval_floor_variants"),
    )
    args = parser.parse_args()

    anchor = pd.read_csv(args.anchor, dtype={"id": "string", "prediction": "int8"})
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

    pseudo = pd.read_parquet(args.pseudolabels)
    anchor_map = anchor.set_index("id")["prediction"]
    pseudo["anchor_prediction_current"] = pseudo["id"].map(anchor_map).astype(np.int8)
    both = pseudo["is_structural_residual"] & pseudo["is_order_residual"]
    fast_only = pseudo["is_structural_residual"] & ~pseudo["is_order_residual"]
    pseudo["structure_tier"] = np.select([both, fast_only], [3, 2], default=1).astype(np.int8)
    pseudo["structure_tier_name"] = np.select(
        [both, fast_only], ["intersection", "fast_only"], default="order_only",
    )
    if args.relevance_source:
        relevance_sum = np.zeros(len(anchor), dtype=np.float32)
        for raw_path, column in args.relevance_source:
            source = pd.read_parquet(raw_path, columns=["id", column])
            if len(source) != len(anchor):
                raise ValueError(f"relevance row count mismatch: {raw_path}")
            if source["id"].astype("string").equals(anchor["id"]):
                values = source[column]
            else:
                values = source.set_index("id")[column].reindex(anchor["id"]).reset_index(drop=True)
                if values.isna().any():
                    raise ValueError(f"relevance IDs missing: {raw_path}")
            relevance_sum += values.rank(pct=True).to_numpy(dtype=np.float32)
        relevance_score = pd.Series(
            relevance_sum / len(args.relevance_source), index=anchor["id"].astype(str),
        )
        pseudo["relevance_score"] = pseudo["id"].astype(str).map(relevance_score).astype(np.float32)
    else:
        pseudo["relevance_score"] = np.float32(0)

    specs = [
        ("v6_floor_retrieval_full", 0.0, "union", False),
        ("v6_floor_retrieval_highcert", 0.50, "union", False),
        ("v6_floor_fast_evidence", 0.0, "fast", False),
        ("v6_floor_intersection_evidence", 0.0, "intersection", False),
    ]
    if args.relevance_source:
        specs.extend([
            ("v6_floor_retrieval_relevance_full", 0.0, "union", True),
            ("v6_floor_retrieval_relevance_midcert", 0.40, "union", True),
            ("v6_floor_retrieval_relevance_highcert", 0.50, "union", True),
        ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    selected_by_name: dict[str, set[str]] = {}
    for name, certainty, evidence, use_relevance in specs:
        selected, tier_counts = select_floor_rows(
            pseudo, term_stats, certainty_min=certainty, evidence=evidence,
            use_relevance=use_relevance,
        )
        selected_by_name[name] = selected
        row = write_submission(anchor, selected, args.output_dir / f"{name}.csv")
        row.update({
            "certainty_min": certainty, "evidence": evidence,
            "use_relevance": use_relevance,
            "selected_tiers": tier_counts,
        })
        reports.append(row)

    comparisons = {}
    full = selected_by_name["v6_floor_retrieval_full"]
    for path in args.compare:
        other = pd.read_csv(path, dtype={"id": "string", "prediction": "int8"})
        if len(other) != len(anchor) or not other["id"].equals(anchor["id"]):
            raise ValueError(f"comparison IDs/order do not match: {path}")
        other_flips = set(other.loc[other["prediction"].gt(anchor["prediction"]), "id"].astype(str))
        comparisons[str(path)] = {
            "other_flips": len(other_flips),
            "intersection_with_retrieval_full": len(other_flips & full),
            "jaccard_with_retrieval_full": (
                len(other_flips & full) / len(other_flips | full) if other_flips | full else 1.0
            ),
        }
    report = {
        "anchor": str(args.anchor),
        "anchor_positive_rate": float(anchor["prediction"].mean()),
        "groups_above_100": int(term_stats["size"].gt(100).sum()),
        "groups_below_floor": int(term_stats["deficit"].gt(0).sum()),
        "total_floor_deficit": int(term_stats["deficit"].sum()),
        "highcert_groups": int((term_stats["deficit"].gt(0) & term_stats["certainty"].ge(0.5)).sum()),
        "highcert_floor_deficit": int(term_stats.loc[
            term_stats["deficit"].gt(0) & term_stats["certainty"].ge(0.5), "deficit"
        ].sum()),
        "variants": reports,
        "comparisons": comparisons,
        "competition_submission_called": False,
    }
    (args.output_dir / "retrieval_floor_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
