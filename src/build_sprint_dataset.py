#!/usr/bin/env python3
"""Build a leakage-safe, retrieval-matched CE train/validation dataset.

The input retrieval parquet is expected to contain one candidate universe for
training queries.  Known positive pairs are authoritative; every other pair is
unlabelled (PU), not a verified negative.  High-risk unlabelled rows are kept in
an audit file and excluded from training by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
ASCII_FOLD = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"})


def normalize_query(value: object) -> str:
    """Conservative Turkish lowercase; retain an ASCII-tolerant second view."""

    text = "" if pd.isna(value) else str(value).translate(TR_LOWER).lower()
    text = re.sub(r"[^0-9a-zçğıöşü]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def add_query_views(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["query_normalized"] = out["query"].map(normalize_query).astype("string")
    out["query_ascii"] = out["query_normalized"].str.translate(ASCII_FOLD).astype("string")
    return out


def grouped_holdout(term_ids: pd.Series, valid_size: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Return one immutable assignment per term; no row-level split is possible."""

    terms = pd.Series(term_ids, dtype="string").dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if len(terms) < 2:
        raise ValueError("At least two distinct term_id values are required")
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    train_idx, valid_idx = next(splitter.split(terms, groups=terms))
    split = pd.Series("train", index=terms.index, dtype="string")
    split.iloc[valid_idx] = "valid"
    return pd.DataFrame({"term_id": terms, "split": split})


def label_and_triage(
    candidates: pd.DataFrame,
    positives: pd.DataFrame,
    *,
    ce_threshold: float = 0.90,
    ambiguous_low: float = 0.50,
) -> pd.DataFrame:
    """Attach PU-aware labels, exclusion reasons and sample weights."""

    required = {"term_id", "item_id", "query", "retrieval_rank"}
    if missing := required - set(candidates.columns):
        raise ValueError(f"candidate columns missing: {sorted(missing)}")
    known = positives[["term_id", "item_id"]].astype("string").drop_duplicates().assign(_positive=1)
    out = candidates.copy()
    out[["term_id", "item_id"]] = out[["term_id", "item_id"]].astype("string")
    out = out.merge(known, on=["term_id", "item_id"], how="left", validate="many_to_one")
    out["label"] = out.pop("_positive").fillna(0).astype("int8")

    ce_col = next((c for c in ("ce_prob", "cross_encoder_prob", "probability") if c in out), None)
    ce = pd.to_numeric(out[ce_col], errors="coerce") if ce_col else pd.Series(np.nan, index=out.index)
    explicit_suspicious = out.get("is_suspicious", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    top3_unknown = out["label"].eq(0) & pd.to_numeric(out["retrieval_rank"], errors="coerce").le(3)
    high_ce_unknown = out["label"].eq(0) & ce.gt(ce_threshold)
    out["exclude_from_train"] = explicit_suspicious | top3_unknown | high_ce_unknown
    out["triage_reason"] = np.select(
        [out["label"].eq(1), explicit_suspicious, high_ce_unknown, top3_unknown],
        ["known_positive", "explicit_suspicious", "high_ce_unlabelled", "top3_unlabelled"],
        default="candidate_negative",
    )
    out["sample_weight"] = np.where(out["label"].eq(1), 1.0, np.where(ce.between(ambiguous_low, ce_threshold), 0.5, 1.0)).astype("float32")
    return out


def sample_training_rows(frame: pd.DataFrame, negatives_per_positive: int, seed: int) -> pd.DataFrame:
    """Keep all positives and deterministically cap candidate negatives per term."""

    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")
    usable = frame.loc[~frame["exclude_from_train"]].copy()
    positive = usable.loc[usable["label"].eq(1)]
    negative = usable.loc[usable["label"].eq(0)].copy()
    counts = positive.groupby("term_id").size().mul(negatives_per_positive).to_dict()
    negative["_key"] = [
        int.from_bytes(hashlib.blake2b(f"{seed}:{t}:{i}".encode(), digest_size=8).digest(), "little")
        for t, i in zip(negative["term_id"], negative["item_id"])
    ]
    negative = negative.sort_values(["term_id", "_key"])
    within_term = negative.groupby("term_id", sort=False).cumcount()
    limits = negative["term_id"].map(counts).fillna(0).astype(int)
    negative = negative.loc[within_term.lt(limits)].drop(columns="_key")
    return pd.concat([positive, negative], ignore_index=True).sort_values(["term_id", "label"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("artifacts/train_testlike_top100_retrieval300.parquet"))
    parser.add_argument("--positives", type=Path, default=Path("data/training_pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sprint_v4"))
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidates = pd.read_parquet(args.candidates)
    positives = pd.read_csv(args.positives, usecols=["term_id", "item_id"])
    triaged = add_query_views(label_and_triage(candidates, positives))
    manifest = grouped_holdout(triaged["term_id"], args.valid_size, args.seed)
    train = sample_training_rows(triaged.merge(manifest, on="term_id", validate="many_to_one"), args.negatives_per_positive, args.seed)
    audit = triaged.loc[triaged["exclude_from_train"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.output_dir / "term_holdout_manifest.parquet", index=False)
    train.to_parquet(args.output_dir / "ce_train_retrieval_matched.parquet", index=False)
    audit.to_parquet(args.output_dir / "pu_suspicious_audit.parquet", index=False)
    summary = {
        "seed": args.seed, "valid_size": args.valid_size, "negatives_per_positive": args.negatives_per_positive,
        "candidate_rows": len(candidates), "training_rows": len(train), "audit_rows": len(audit),
        "training_positive_rate": float(train["label"].mean()),
        "train_terms": int(manifest["split"].eq("train").sum()), "valid_terms": int(manifest["split"].eq("valid").sum()),
        "ce_filter_available": any(c in candidates for c in ("ce_prob", "cross_encoder_prob", "probability")),
    }
    (args.output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
