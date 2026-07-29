#!/usr/bin/env python3
"""Create a compact, read-only audit of the current 0.901->0.94 work.

This never calls Kaggle and never creates a candidate from an unavailable
model.  It records which gates are complete, blocked, or pending so a later
GPT-OSS download can be audited without rerunning Qwen/Mistral.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    data = ROOT / "data"
    artifacts = ROOT / "artifacts"
    sub = pd.read_csv(data / "submission_pairs.csv", usecols=["id", "term_id"])
    anchor_path = artifacts / "llm_student_cascade/package/trendyol_llm_student_cascade/input/anchor_v6.parquet"
    anchor = pd.read_parquet(anchor_path)
    candidates_dir = artifacts / "merged_candidates_v1"
    merge = json.loads((candidates_dir / "merge_report.json").read_text())

    candidate_checks = {}
    for name, info in merge.get("candidates", {}).items():
        path = candidates_dir / info["file"]
        frame = pd.read_csv(path)
        candidate_checks[name] = {
            "file": str(path.relative_to(ROOT)),
            "rows": int(len(frame)),
            "id_order_matches_submission": bool(frame["id"].astype(str).equals(sub["id"].astype(str))),
            "binary": bool(set(frame["prediction"].unique().tolist()) <= {0, 1}),
            "positive_count": int(frame["prediction"].sum()),
            "positive_rate": float(frame["prediction"].mean()),
            "sha256": sha256(path),
            "sha256_matches_merge_report": bool(sha256(path) == info.get("sha256")),
        }

    full_prep = artifacts / "retrieval_replica_v2/full"
    queries = pd.read_parquet(full_prep / "queries.parquet") if (full_prep / "queries.parquet").exists() else None
    retrieval_gate = {"status": "not_run"}
    ft_report = artifacts / "retrieval_replica_v2/full/finetuned_report.json"
    if ft_report.exists():
        retrieval_gate = json.loads(ft_report.read_text())
    elif (artifacts / "retrieval_replica_v2/smoke/finetuned_report.json").exists():
        smoke = json.loads((artifacts / "retrieval_replica_v2/smoke/finetuned_report.json").read_text())
        retrieval_gate = {
            "status": "smoke_failed",
            "gate_pass": bool(smoke.get("gate_pass", False)),
            "smoke_val_recall_at_100": smoke.get("finetuned", {}).get("val", {}).get("recall_at_k"),
            "required_recall_at_100": smoke.get("recall_gate"),
        }

    gpt_dir = artifacts / "llm_student_cascade/votes/gpt_oss_20b"
    gpt_parts = sorted(gpt_dir.glob("part_*.parquet"))
    report = {
        "schema_version": 1,
        "date": "2026-07-17",
        "kaggle_submission_called": False,
        "data": {
            "submission_rows": int(len(sub)),
            "terms": int(sub["term_id"].nunique()),
            "exactly_100_queries": int((sub.groupby("term_id").size() == 100).sum()),
            "over_100_queries": int((sub.groupby("term_id").size() > 100).sum()),
            "anchor_rows": int(len(anchor)),
            "anchor_positive_count": int(anchor["prediction"].sum()),
            "anchor_positive_rate": float(anchor["prediction"].mean()),
        },
        "gpt_oss": {
            "status": "available" if gpt_parts else "missing",
            "part_count": len(gpt_parts),
            "expected_dir": str(gpt_dir.relative_to(ROOT)),
            "audit": json.loads((artifacts / "llm_student_cascade/gpt_oss_audit.json").read_text())
            if (artifacts / "llm_student_cascade/gpt_oss_audit.json").exists() else None,
        },
        "retrieval_replica_v2": {
            "full_prepare": {
                "status": "complete" if queries is not None else "missing",
                "queries": int(len(queries)) if queries is not None else None,
                "train_queries": int((queries["split"] == "train").sum()) if queries is not None else None,
                "validation_queries": int((queries["split"] == "val").sum()) if queries is not None else None,
                "catalog_rows": int(len(pd.read_parquet(full_prep / "catalog.parquet")))
                if (full_prep / "catalog.parquet").exists() else None,
            },
            "gate": retrieval_gate,
            "wide_scale_residual_allowed": bool(retrieval_gate.get("gate_pass", False)),
        },
        "candidates": candidate_checks,
        "selection_policy": {
            "triple_candidates_allowed": bool(gpt_parts) and merge.get("gpt_oss_present", False),
            "retrieval_residual_allowed": bool(retrieval_gate.get("gate_pass", False)),
            "submission_upload_performed": False,
        },
    }
    out = artifacts / "pipeline_state_audit_20260717.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
