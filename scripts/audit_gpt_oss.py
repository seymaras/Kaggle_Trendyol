#!/usr/bin/env python3
"""Audit the GPT-OSS third-referee run once its Colab votes are downloaded.

Run AFTER copying the GPT-OSS cascade votes to:
    artifacts/llm_student_cascade/votes/gpt_oss_20b/part_*.parquet
(and optionally model_run.json).  Reports exactly what the user asked for:
  - rows GPT-OSS processed
  - GPT-OSS positive rate + single-class-collapse check
  - 0->1 and 1->0 relative to the anchor (net effect of the triple candidate)
  - how many qwen_mistral consensus flips GPT-OSS VETOED
  - triple-model agreement rate
  - triple-strict vs triple-medium comparison

NO Kaggle submission. This script only reads votes and prints/writes a report.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/seyma/Documents/Kaggle_Trendyol")
IN = ROOT / "artifacts/llm_student_cascade/package/trendyol_llm_student_cascade/input"
CASCADE = ROOT / "artifacts/llm_student_cascade/votes"
SECOND_PREFILTER, THIRD_PREFILTER = 0.58, 0.65
STRICT, MEDIUM = 0.80, 0.65


def load(vote_dir: Path):
    fs = sorted(glob.glob(str(vote_dir / "part_*.parquet")))
    if not fs:
        return None
    return pd.concat([pd.read_parquet(p, columns=["id", "label", "p_relevant", "confidence"])
                      for p in fs], ignore_index=True)


def main() -> None:
    gpt = load(CASCADE / "gpt_oss_20b")
    if gpt is None:
        report = {
            "status": "missing",
            "expected_path": str(CASCADE / "gpt_oss_20b" / "part_*.parquet"),
            "message": "Run the Colab Harmony repair cell before selecting triple candidates.",
            "kaggle_submission_called": False,
        }
        out = CASCADE.parent / "gpt_oss_audit.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nreport -> {out}")
        return

    pool = pd.read_parquet(IN / "llm_judge_pool.parquet")
    pid = pool["id"].astype("string").to_numpy()
    alt = pool["alternative_prediction"].to_numpy(np.int8)
    ancp = pool["anchor_prediction"].to_numpy(np.int8)

    q = load(CASCADE / "qwen")
    q = q.set_index(q["id"].astype("string")).reindex(pid)
    lab0, c0 = q["label"].to_numpy(np.int8), q["confidence"].to_numpy(np.float32)
    sm = (lab0 == alt) & (c0 >= SECOND_PREFILTER)
    m = load(CASCADE / "mistral")
    m = m.set_index(m["id"].astype("string")).reindex(pid[sm])
    lab1 = np.full(len(pool), -1, np.int8); c1 = np.zeros(len(pool), np.float32)
    lab1[sm] = m["label"].to_numpy(np.int8); c1[sm] = m["confidence"].to_numpy(np.float32)
    joint2 = np.minimum(c0, c1)
    consensus = (lab0 == alt) & (lab1 == alt)          # qwen ∧ mistral pick the flip

    third_mask = consensus & (c1 >= THIRD_PREFILTER)   # rows GPT-OSS was asked to judge
    g = gpt.set_index(gpt["id"].astype("string")).reindex(pid[third_mask])
    if g["label"].isna().any():
        raise RuntimeError(
            f"GPT-OSS votes do not cover the third_mask exactly "
            f"({int(g['label'].isna().sum())} missing of {int(third_mask.sum())})."
        )
    lab2 = np.full(len(pool), -1, np.int8); c2 = np.zeros(len(pool), np.float32)
    lab2[third_mask] = g["label"].to_numpy(np.int8)
    c2[third_mask] = g["confidence"].to_numpy(np.float32)

    gpt_pos_rate = float((lab2[third_mask] == 1).mean())
    collapse = len(set(lab2[third_mask].tolist())) < 2
    triple = consensus & (lab2 == alt)                 # all three pick the flip
    triple_conf = np.minimum(joint2, c2)

    # veto = a qwen∧mistral consensus flip that GPT-OSS was asked about and REJECTED
    asked = third_mask
    vetoed = asked & (lab2 != alt)
    agreement = float((lab2[asked] == alt).mean())     # of asked rows, GPT agrees with the flip

    def bucket(mask, tag):
        fa = ancp[mask]
        return dict(name=tag, flips=int(mask.sum()),
                    zero_to_one=int((fa == 0).sum()), one_to_zero=int((fa == 1).sum()),
                    net_pos=int((fa == 0).sum() - (fa == 1).sum()))

    report = {
        "status": "complete",
        "gpt_oss_rows_processed": int(third_mask.sum()),
        "gpt_oss_positive_rate": round(gpt_pos_rate, 4),
        "single_class_collapse": collapse,
        "qwen_mistral_consensus_flips": int(consensus.sum()),
        "gpt_asked_rows": int(asked.sum()),
        "gpt_vetoed_rows": int(vetoed.sum()),
        "gpt_veto_rate_of_asked": round(float(vetoed.sum() / max(1, asked.sum())), 4),
        "triple_agreement_rate_on_asked": round(agreement, 4),
        "triple_consensus_flips": int(triple.sum()),
        "candidates": {
            "qwen_mistral_strict": bucket(consensus & (joint2 >= STRICT), "qwen_mistral_strict"),
            "qwen_mistral_medium": bucket(consensus & (joint2 >= MEDIUM), "qwen_mistral_medium"),
            "triple_strict": bucket(triple & (triple_conf >= STRICT), "triple_strict"),
            "triple_medium": bucket(triple & (triple_conf >= MEDIUM), "triple_medium"),
        },
        "kaggle_submission_called": False,
    }
    if collapse:
        report["WARNING"] = "GPT-OSS collapsed to a single class — DO NOT use this run."
    out = CASCADE.parent / "gpt_oss_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
