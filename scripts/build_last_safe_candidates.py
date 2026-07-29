#!/usr/bin/env python3
"""Build the two final, conservative upload candidates.

Candidate A is the already proven 0.915 strict file. Candidate B starts from
that exact file and adds only official-engine high-cert rows that remain 0 and
are not explicitly judged negative by either round-1 LLM. No broad consensus,
cardinality, max-stack, or retrieval residual is included.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/merged_candidates_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labels(folder: Path) -> dict[str, int]:
    frames = [pd.read_parquet(p, columns=["id", "label"]) for p in sorted(folder.glob("part_*.parquet"))]
    if not frames:
        return {}
    frame = pd.concat(frames, ignore_index=True)
    return dict(zip(frame["id"].astype(str), frame["label"].astype(int)))


def main() -> None:
    strict_path = OUT / "01_llm_qwen_mistral_strict.csv"
    strict = pd.read_csv(strict_path)
    strict_ids = strict["id"].astype(str).to_numpy()
    strict_pred = strict["prediction"].to_numpy(np.uint8)
    strict_map = dict(zip(strict_ids, strict_pred))

    floor = pd.read_csv(ROOT / "artifacts/official_engine_colab/official_v6_floor_highcert_6008.csv")
    v6 = pd.read_csv(ROOT / "artifacts/final_candidates/00_proven_anchor_v6_lb0874.csv")
    v6_map = dict(zip(v6["id"].astype(str), v6["prediction"].astype(np.uint8)))
    q1 = load_labels(ROOT / "artifacts/llm_judge_v1/drive_votes/qwen")
    m1 = load_labels(ROOT / "artifacts/llm_judge_v1/drive_votes/mistral")

    add_ids = []
    excluded_llm_conflict = 0
    already_positive = 0
    for uid, floor_pred in zip(floor["id"].astype(str), floor["prediction"].astype(np.uint8)):
        if floor_pred != 1 or v6_map.get(uid, 0) != 0:
            continue
        if strict_map.get(uid, 0) == 1:
            already_positive += 1
            continue
        votes = [q1[uid], m1[uid]] if uid in q1 and uid in m1 else [q1.get(uid), m1.get(uid)]
        votes = [v for v in votes if v is not None]
        if any(v == 0 for v in votes):
            excluded_llm_conflict += 1
            continue
        add_ids.append(uid)

    pred = strict_pred.copy()
    idx = {uid: i for i, uid in enumerate(strict_ids)}
    for uid in add_ids:
        pred[idx[uid]] = 1
    out_path = OUT / "09_strict_plus_engine_highcert_clean.csv"
    pd.DataFrame({"id": strict_ids, "prediction": pred}).to_csv(out_path, index=False)

    report = {
        "base": strict_path.name,
        "base_proven_lb": 0.915,
        "candidate": out_path.name,
        "added_0_to_1": int(len(add_ids)),
        "removed_1_to_0": 0,
        "floor_rows_already_positive_in_base": int(already_positive),
        "excluded_due_to_llm_negative": int(excluded_llm_conflict),
        "positive_count": int(pred.sum()),
        "positive_rate": round(float(pred.mean()), 6),
        "rows": int(len(pred)),
        "id_order_matches_base": bool(np.array_equal(strict_ids, pd.read_csv(out_path, usecols=["id"])["id"].astype(str).to_numpy())),
        "sha256_base": sha256(strict_path),
        "sha256_candidate": sha256(out_path),
        "kaggle_submission_called": False,
    }
    (OUT / "last_safe_candidates_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
