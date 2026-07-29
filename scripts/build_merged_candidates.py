#!/usr/bin/env python3
"""Build auditable merge candidates on the proven 0.901 anchor.

NO Kaggle submission is ever made by this script. It only writes CSVs + a report.

Signals combined (each independent):
  * LLM round-2  : qwen_mistral cascade consensus flips (both directions).
                   If GPT-OSS votes are present, an extra GPT-vetoed (triple)
                   variant is produced.
  * Engine floor : official-engine floor-deficit high-cert 0->1 flips that are
                   still negative in the 0.901 anchor.
  * Structural   : forced-membership residual still-negatives, split by how many
                   structural rankers agree it is outside top-100, with rows the
                   LLM explicitly judged negative removed (user rule: never
                   auto-add a structural decision that conflicts with the LLM).

Base anchor = anchor_v6.parquet (== llm_consensus_medium, LB 0.901, 24.72% pos).
"""
from __future__ import annotations
import glob, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/seyma/Documents/Kaggle_Trendyol")
IN = ROOT / "artifacts/llm_student_cascade/package/trendyol_llm_student_cascade/input"
CASCADE = ROOT / "artifacts/llm_student_cascade/votes"          # round-2 qwen/mistral (+gpt when ready)
DRIVE = ROOT / "artifacts/llm_judge_v1/drive_votes"            # round-1 broad LLM coverage
RESIDUAL = ROOT / "artifacts/official_engine_colab/structural_residual_candidates.parquet"
FLOOR_HC = ROOT / "artifacts/official_engine_colab/official_v6_floor_highcert_6008.csv"
V6 = ROOT / "artifacts/final_candidates/00_proven_anchor_v6_lb0874.csv"
OUT = ROOT / "artifacts/merged_candidates_v1"
SECOND_PREFILTER, THIRD_PREFILTER = 0.58, 0.65


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_votes(vote_dir: Path, cols=("id", "label", "confidence")) -> pd.DataFrame:
    fs = sorted(glob.glob(str(vote_dir / "part_*.parquet")))
    if not fs:
        return pd.DataFrame(columns=list(cols))
    df = pd.concat([pd.read_parquet(p, columns=list(cols)) for p in fs], ignore_index=True)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.read_parquet(IN / "anchor_v6.parquet")
    anchor_ids = anchor["id"].astype(str).to_numpy()
    base = anchor["prediction"].to_numpy(np.uint8)
    pos = {i: p for i, p in zip(anchor_ids, base)}
    idx = {i: k for k, i in enumerate(anchor_ids)}
    n = len(anchor)
    print(f"anchor rows={n} pos={int(base.sum())} rate={base.mean():.4f}")

    # ---------- LLM round-2 cascade consensus (qwen ∧ mistral) ----------
    pool = pd.read_parquet(IN / "llm_judge_pool.parquet")
    pid = pool["id"].astype("string").to_numpy()
    alt = pool["alternative_prediction"].to_numpy(np.int8)
    ancp = pool["anchor_prediction"].to_numpy(np.int8)
    q = load_votes(CASCADE / "qwen")
    q = q.set_index(q["id"].astype("string")).reindex(pid)
    lab0, c0 = q["label"].to_numpy(np.int8), q["confidence"].to_numpy(np.float32)
    sm = (lab0 == alt) & (c0 >= SECOND_PREFILTER)
    m = load_votes(CASCADE / "mistral")
    m = m.set_index(m["id"].astype("string")).reindex(pid[sm])
    lab1 = np.full(len(pool), -1, np.int8); c1 = np.zeros(len(pool), np.float32)
    lab1[sm] = m["label"].to_numpy(np.int8); c1[sm] = m["confidence"].to_numpy(np.float32)
    joint = np.minimum(c0, c1)
    consensus = (lab0 == alt) & (lab1 == alt)

    # optional GPT-OSS third referee
    gpt_dir = CASCADE / "gpt_oss_20b"
    gpt_present = bool(glob.glob(str(gpt_dir / "part_*.parquet")))
    triple = None
    gpt_audit = {"status": "missing"}
    if gpt_present:
        third_mask = consensus & (c1 >= THIRD_PREFILTER)
        g = load_votes(gpt_dir)
        g = g.set_index(g["id"].astype("string")).reindex(pid[third_mask])
        if g["label"].isna().any():
            raise RuntimeError(
                "GPT-OSS output does not exactly cover the third-model pool: "
                f"{int(g['label'].isna().sum())} missing rows."
            )
        lab2 = np.full(len(pool), -1, np.int8)
        lab2[third_mask] = g["label"].to_numpy(np.int8)
        gpt_pos_rate = float((lab2[third_mask] == 1).mean())
        if len(set(lab2[third_mask].tolist())) < 2:
            raise RuntimeError(
                "GPT-OSS collapsed to a single class; refusing all triple candidates."
            )
        triple = consensus & (lab2 == alt)
        gpt_audit = {
            "status": "complete",
            "rows_processed": int(third_mask.sum()),
            "positive_rate": round(gpt_pos_rate, 6),
            "qwen_mistral_consensus_flips": int(consensus.sum()),
            "vetoed": int((third_mask & (lab2 != alt)).sum()),
            "veto_rate": round(float((third_mask & (lab2 != alt)).sum() / max(1, third_mask.sum())), 6),
            "triple_agreement_rate": round(float((lab2[third_mask] == alt).mean()), 6),
            "triple_confirmed": int(triple.sum()),
        }
        print(f"GPT-OSS present: third_mask={int(third_mask.sum())} triple_confirm={int(triple.sum())}")
    else:
        print("GPT-OSS votes NOT present -> producing pre-GPT (qwen_mistral) LLM candidates only")

    def llm_flip_ids(threshold: float, use_triple: bool):
        base_mask = (triple if use_triple else consensus) & (joint >= threshold)
        ids = pid[base_mask]
        return {str(i): int(a) for i, a in zip(ids, ancp[base_mask])}  # id -> anchor(before) value

    # ---------- Engine floor-deficit high-cert (still-neg in anchor) ----------
    v6 = pd.read_csv(V6); v6m = dict(zip(v6["id"].astype(str), v6["prediction"].astype(np.int8)))
    hc = pd.read_csv(FLOOR_HC); hcm = dict(zip(hc["id"].astype(str), hc["prediction"].astype(np.int8)))
    floor_ids = [i for i in hc["id"].astype(str)
                 if hcm[i] == 1 and v6m.get(i, 0) == 0 and pos.get(i, 0) == 0]
    print(f"engine floor high-cert still-neg in anchor: {len(floor_ids)}")

    # ---------- Structural forced-membership residual (clean, LLM-conflict excluded) ----------
    q1 = load_votes(DRIVE / "qwen"); m1 = load_votes(DRIVE / "mistral")
    qL = dict(zip(q1["id"].astype(str), q1["label"].astype(np.int8)))
    mL = dict(zip(m1["id"].astype(str), m1["label"].astype(np.int8)))
    res = pd.read_parquet(RESIDUAL, columns=["id", "outside_agreement", "forced_confidence"])
    res["id"] = res["id"].astype(str)
    res = res[res["id"].map(lambda i: pos.get(i, 1) == 0)]  # still negative in anchor

    def not_llm_negative(i):
        votes = [v for v in (qL.get(i), mL.get(i)) if v is not None]
        return not votes or all(v == 1 for v in votes)  # unjudged or unanimous-positive
    res["clean"] = res["id"].map(not_llm_negative)
    struct_a3 = res[(res.outside_agreement == 3) & res.clean]["id"].tolist()
    struct_a2 = res[(res.outside_agreement >= 2) & res.clean]["id"].tolist()
    print(f"structural residual clean: a3={len(struct_a3)} a>=2={len(struct_a2)}")

    # ---------- candidate assembly ----------
    def apply_flips(name, llm_map=None, add_ids=None, note=""):
        pred = base.copy()
        z2o = o2z = 0
        if llm_map:
            for i, before in llm_map.items():
                k = idx[i]; after = 1 - before
                pred[k] = after
                z2o += (before == 0); o2z += (before == 1)
        if add_ids:
            for i in add_ids:
                k = idx[i]
                if pred[k] == 0:
                    pred[k] = 1; z2o += 1
        assert set(np.unique(pred).tolist()) <= {0, 1}
        path = OUT / f"{name}.csv"
        pd.DataFrame({"id": anchor_ids, "prediction": pred}).to_csv(path, index=False)
        rep = dict(file=path.name, note=note, flips=int(z2o + o2z), zero_to_one=int(z2o),
                   one_to_zero=int(o2z), positive_count=int(pred.sum()),
                   positive_rate=round(float(pred.mean()), 6), sha256=sha256(path))
        print(f"  {name:38s} flips={rep['flips']:6d} 0->1={z2o:6d} 1->0={o2z:5d} "
              f"rate={rep['positive_rate']:.4f}")
        return rep

    reports = {}
    use_triple = gpt_present
    tag = "gpt_triple" if use_triple else "qwen_mistral"
    print("\n=== candidates ===")
    # Reference
    reports["00_anchor_0901"] = apply_flips("00_anchor_0901", note="proven LB 0.901 baseline")
    # C1: LLM round-2 only (isolate LLM signal)
    reports[f"01_llm_{tag}_strict"] = apply_flips(
        f"01_llm_{tag}_strict", llm_map=llm_flip_ids(0.80, use_triple),
        note=f"LLM round-2 {tag} consensus, strict conf>=0.80")
    reports[f"02_llm_{tag}_medium"] = apply_flips(
        f"02_llm_{tag}_medium", llm_map=llm_flip_ids(0.65, use_triple),
        note=f"LLM round-2 {tag} consensus, medium conf>=0.65")
    # C2: engine structural only (isolate structural signal), no LLM round-2
    reports["03_engine_floor_highcert"] = apply_flips(
        "03_engine_floor_highcert", add_ids=floor_ids,
        note="0.901 + floor-deficit high-cert (structural, LLM-independent)")
    reports["04_engine_struct_3signal"] = apply_flips(
        "04_engine_struct_3signal", add_ids=set(floor_ids) | set(struct_a3),
        note="0.901 + floor high-cert + forced-membership agreement-3 (LLM-conflict excluded)")
    reports["05_engine_struct_2signal"] = apply_flips(
        "05_engine_struct_2signal", add_ids=set(floor_ids) | set(struct_a2),
        note="0.901 + floor high-cert + forced-membership agreement>=2 (LLM-conflict excluded)")
    # C3: combined LLM round-2 + engine structural (full stack)
    reports[f"06_combined_{tag}_medium_plus_3signal"] = apply_flips(
        f"06_combined_{tag}_medium_plus_3signal",
        llm_map=llm_flip_ids(0.65, use_triple),
        add_ids=set(floor_ids) | set(struct_a3),
        note="LLM round-2 medium + floor high-cert + forced-membership agreement-3")
    if gpt_present:
        # GPT-vetted candidates are intentionally kept separate from the
        # pre-GPT variants so no one can mistake a Qwen/Mistral candidate for
        # a triple-referee result.  Engine additions are considered only on
        # the current LLM anchor and never override a judged negative.
        reports["07_gpt_triple_strict_only"] = apply_flips(
            "07_gpt_triple_strict_only",
            llm_map=llm_flip_ids(0.80, True),
            note="GPT-vetted triple consensus, strict confidence>=0.80")
        reports["08_gpt_triple_medium_only"] = apply_flips(
            "08_gpt_triple_medium_only",
            llm_map=llm_flip_ids(0.65, True),
            note="GPT-vetted triple consensus, medium confidence>=0.65")
        reports["09_gpt_engine_3signal"] = apply_flips(
            "09_gpt_engine_3signal",
            llm_map=llm_flip_ids(0.65, True),
            add_ids=set(floor_ids) | set(struct_a3),
            note="GPT triple medium + official engine high-cert + clean structural 3-signal")
        reports["10_gpt_engine_2signal"] = apply_flips(
            "10_gpt_engine_2signal",
            llm_map=llm_flip_ids(0.65, True),
            add_ids=set(floor_ids) | set(struct_a2),
            note="GPT triple medium + official engine high-cert + clean structural >=2-signal")

    manifest = dict(
        base_anchor="anchor_v6.parquet (== llm_consensus_medium, LB 0.901)",
        gpt_oss_present=gpt_present, llm_tag=tag,
        gpt_oss_audit=gpt_audit,
        floor_highcert_new=len(floor_ids),
        structural_clean_a3=len(struct_a3), structural_clean_a2=len(struct_a2),
        kaggle_submission_called=False, candidates=reports)
    (OUT / "merge_report.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nreport -> {OUT/'merge_report.json'}")
    print("Kaggle submission called: False")


if __name__ == "__main__":
    main()
