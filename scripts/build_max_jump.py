#!/usr/bin/env python3
"""Max-jump stack: every validated signal on top of the proven 0.915 base.

Layers (in order, all auditable; LLM-contradicted rows are never auto-added):
  L0  base           = 0.901 anchor + FULL qwen∧mistral consensus (broad, conf>=0.50,
                       both directions) — supersets the proven strict 0.915 flips
  L1  structural     = forced-membership residual still-neg, agreement>=2,
                       excluding rows round-1 LLM judged negative/mixed
  L2  floor midcert  = official floor-deficit mid-cert 0->1 still-neg,
                       same LLM-conflict exclusion
  L3  k=0 floor      = queries left with zero positives get their single best
                       student-prob candidate set to 1

No Kaggle submission is made. Output: 08_max_jump_stack.csv + audit JSON.
"""
from __future__ import annotations
import glob, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/seyma/Documents/Kaggle_Trendyol")
IN = ROOT / "artifacts/llm_student_cascade/package/trendyol_llm_student_cascade/input"
CASCADE = ROOT / "artifacts/llm_student_cascade/votes"
DRIVE = ROOT / "artifacts/llm_judge_v1/drive_votes"
RESIDUAL = ROOT / "artifacts/official_engine_colab/structural_residual_candidates.parquet"
FLOOR_MC = ROOT / "artifacts/official_engine_colab/official_v6_floor_midcert_8304.csv"
V6 = ROOT / "artifacts/final_candidates/00_proven_anchor_v6_lb0874.csv"
STUDENT = ROOT / "artifacts/llm_student_v1/student_scores.parquet"
OUT = ROOT / "artifacts/merged_candidates_v1"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_votes(d: Path, cols=("id", "label", "confidence")):
    fs = sorted(glob.glob(str(d / "part_*.parquet")))
    return pd.concat([pd.read_parquet(p, columns=list(cols)) for p in fs], ignore_index=True)


def main() -> None:
    anchor = pd.read_parquet(IN / "anchor_v6.parquet")
    ids = anchor["id"].astype(str).to_numpy()
    pred = anchor["prediction"].to_numpy(np.uint8).copy()
    idx = {i: k for k, i in enumerate(ids)}
    anc0 = pred.copy()
    audit = {"base_anchor_pos": int(pred.sum())}

    # ---- L0: full qwen∧mistral consensus (broad, conf>=0.50), both directions ----
    pool = pd.read_parquet(IN / "llm_judge_pool.parquet")
    pid = pool["id"].astype("string").to_numpy()
    alt = pool["alternative_prediction"].to_numpy(np.int8)
    q = load_votes(CASCADE / "qwen"); q = q.set_index(q["id"].astype("string")).reindex(pid)
    lab0, c0 = q["label"].to_numpy(np.int8), q["confidence"].to_numpy(np.float32)
    sm = (lab0 == alt) & (c0 >= 0.58)
    m = load_votes(CASCADE / "mistral"); m = m.set_index(m["id"].astype("string")).reindex(pid[sm])
    lab1 = np.full(len(pool), -1, np.int8); c1 = np.zeros(len(pool), np.float32)
    lab1[sm] = m["label"].to_numpy(np.int8); c1[sm] = m["confidence"].to_numpy(np.float32)
    joint = np.minimum(c0, c1)
    consensus = (lab0 == alt) & (lab1 == alt) & (joint >= 0.50)
    z2o = o2z = 0
    for i, a in zip(pid[consensus], alt[consensus]):
        k = idx[str(i)]
        if pred[k] != a:
            z2o += (a == 1); o2z += (a == 0)
            pred[k] = a
    audit["L0_llm_broad_consensus"] = dict(flips=int(z2o + o2z), zero_to_one=int(z2o), one_to_zero=int(o2z))

    # round-1 LLM verdict map for the conflict rule
    q1 = load_votes(DRIVE / "qwen", ("id", "label")); m1 = load_votes(DRIVE / "mistral", ("id", "label"))
    qL = dict(zip(q1["id"].astype(str), q1["label"].astype(np.int8)))
    mL = dict(zip(m1["id"].astype(str), m1["label"].astype(np.int8)))

    def llm_clean(i: str) -> bool:
        v = [x for x in (qL.get(i), mL.get(i)) if x is not None]
        return not v or all(x == 1 for x in v)

    def add_layer(name, cand_ids):
        added = 0
        for i in cand_ids:
            k = idx.get(i)
            if k is not None and pred[k] == 0 and llm_clean(i):
                pred[k] = 1; added += 1
        audit[name] = dict(added=int(added), offered=int(len(cand_ids)))
        return added

    # ---- L1: forced-membership residual, agreement>=2, still-neg in 0.901 anchor ----
    res = pd.read_parquet(RESIDUAL, columns=["id", "outside_agreement"])
    res["id"] = res["id"].astype(str)
    res = res[res["id"].map(lambda i: anc0[idx[i]] == 0)]
    add_layer("L1_struct_a2_clean", res[res.outside_agreement >= 2]["id"].tolist())

    # ---- L2: floor mid-cert (0->1 vs v6), still-neg here ----
    v6 = pd.read_csv(V6); v6m = dict(zip(v6["id"].astype(str), v6["prediction"].astype(np.int8)))
    mc = pd.read_csv(FLOOR_MC)
    floor = [i for i, p in zip(mc["id"].astype(str), mc["prediction"].astype(np.int8))
             if p == 1 and v6m.get(i, 0) == 0]
    add_layer("L2_floor_midcert", floor)

    # ---- L3: k=0 floor via best student prob ----
    sp = pd.read_csv(ROOT / "data/submission_pairs.csv")[["id", "term_id"]]
    st = pd.read_parquet(STUDENT)
    sp = sp.merge(st, on="id")
    sp["pred"] = pred[[idx[i] for i in sp["id"].astype(str)]]
    kcount = sp.groupby("term_id")["pred"].sum()
    zero_terms = set(kcount[kcount == 0].index)
    fixed = 0
    zsub = sp[sp["term_id"].isin(zero_terms)]
    for t, g in zsub.groupby("term_id"):
        best = g.loc[g["student_prob"].idxmax(), "id"]
        k = idx[str(best)]
        if pred[k] == 0:
            pred[k] = 1; fixed += 1
    audit["L3_k0_floor"] = dict(zero_k_queries=int(len(zero_terms)), fixed=int(fixed))

    # ---- write + audit ----
    path = OUT / "08_max_jump_stack.csv"
    pd.DataFrame({"id": ids, "prediction": pred}).to_csv(path, index=False)
    fa = anc0[pred != anc0]
    audit["total_vs_0901_anchor"] = dict(
        flips=int((pred != anc0).sum()),
        zero_to_one=int((fa == 0).sum()), one_to_zero=int((fa == 1).sum()))
    # delta vs proven 0.915 (strict candidate)
    strict = pd.read_csv(OUT / "01_llm_qwen_mistral_strict.csv")["prediction"].to_numpy(np.uint8)
    sa = strict[pred != strict]
    audit["delta_vs_proven_0915"] = dict(
        flips=int((pred != strict).sum()),
        zero_to_one=int((sa == 0).sum()), one_to_zero=int((sa == 1).sum()))
    audit["positive_count"] = int(pred.sum())
    audit["positive_rate"] = round(float(pred.mean()), 6)
    audit["sha256"] = sha256(path)
    audit["kaggle_submission_called"] = False
    (OUT / "max_jump_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))

    sample = pd.read_csv(ROOT / "data/sample_submission.csv")["id"].astype(str).to_numpy()
    ok = np.array_equal(pd.read_csv(path)["id"].astype(str).to_numpy(), sample)
    print(f"\nvalid_id_order={ok}  file={path}")


if __name__ == "__main__":
    main()
