#!/usr/bin/env python3
"""V2 blend: average multiple cross-encoder score files, calibrate flip thresholds
to PRECISION TARGETS on the real-teacher query-disjoint holdout, and assemble
guarded candidates vs the anchor.

Thresholds are not hand-picked numbers: for each precision target p we find the
smallest hi with Prec(teacher==1 | score>=hi) >= p on held-out queries, and the
largest lo with Prec(teacher==0 | score<=lo) >= p. Same per-query caps and
strong-vote locks as v1 assembly.

NO Kaggle submission is ever made.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd


def log(m): print(m, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--score-files", nargs="+", required=True, help="parquets with id, ce_score")
    p.add_argument("--teacher", required=True, help="teacher_labels.parquet WITH pseudo column")
    p.add_argument("--anchor", required=True, help="anchor parquet or csv (id, prediction)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--precision-targets", nargs="+", type=float, default=[0.98, 0.96, 0.94])
    p.add_argument("--min-flip-support", type=int, default=300, help="min val rows above threshold")
    return p.parse_args()


def main():
    a = parse_args()
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    scs = [pd.read_parquet(f) for f in a.score_files]
    df = scs[0].rename(columns={"ce_score": "s0"})
    for k, s in enumerate(scs[1:], 1):
        df = df.merge(s.rename(columns={"ce_score": f"s{k}"}), on="id")
    cols = [c for c in df.columns if c.startswith("s")]
    df["score"] = df[cols].mean(axis=1)
    log(f"blended {len(cols)} score files over {len(df)} rows")

    teacher = pd.read_parquet(a.teacher)
    real_val = teacher[(~teacher.get("pseudo", pd.Series(False, index=teacher.index)))
                       & (teacher["split"] == "val")]
    val = real_val.merge(df[["id", "score"]], on="id")
    y = val["label"].to_numpy(); s = val["score"].to_numpy()
    log(f"calibration holdout: {len(val)} real-teacher val rows")

    def hi_for_precision(p):
        best = None
        for t in np.arange(0.50, 0.9999, 0.005):
            m = s >= t
            if m.sum() < a.min_flip_support: break
            if (y[m] == 1).mean() >= p: best = float(t); break
        return best

    def lo_for_precision(p):
        best = None
        for t in np.arange(0.50, 0.0001, -0.005):
            m = s <= t
            if m.sum() < a.min_flip_support: break
            if (y[m] == 0).mean() >= p: best = float(t); break
        return best

    anchor = (pd.read_parquet(a.anchor) if str(a.anchor).endswith(".parquet")
              else pd.read_csv(a.anchor))
    pairs = pd.read_parquet(Path(a.input_dir) / "pairs.parquet", columns=["id", "term_id"])
    base_df = anchor.merge(df[["id", "score"]], on="id", how="left").merge(pairs, on="id")
    base_df["score"] = base_df["score"].fillna(0.5)
    strong = teacher[(teacher["weight"] >= 0.9) & (~teacher["pseudo"])]
    lock_pos = set(strong[strong["label"] == 1]["id"])
    lock_neg = set(strong[strong["label"] == 0]["id"])
    ids = base_df["id"].to_numpy(); base = base_df["prediction"].to_numpy(np.uint8)
    scr = base_df["score"].to_numpy(np.float32); term = base_df["term_id"].to_numpy()
    in_lockpos = pd.Series(ids).isin(lock_pos).to_numpy()
    in_lockneg = pd.Series(ids).isin(lock_neg).to_numpy()

    audit = {"score_files": a.score_files, "calibration_rows": int(len(val))}
    for p in a.precision_targets:
        hi, lo = hi_for_precision(p), lo_for_precision(p)
        name = f"ce2_p{int(p*100)}"
        if hi is None or lo is None:
            audit[name] = {"skipped": f"no threshold reaches precision {p}"}
            log(f"{name}: SKIP (precision {p} unreachable)"); continue
        pred = base.copy()
        want_up = (base == 0) & (scr >= hi) & ~in_lockneg
        want_dn = (base == 1) & (scr <= lo) & ~in_lockpos
        z2o = o2z = 0
        gdf = pd.DataFrame({"i": np.arange(len(ids)), "term": term,
                            "up": want_up, "dn": want_dn, "s": scr, "p": base})
        for t, g in gdf.groupby("term", sort=False):
            n = len(g); cap = max(3, int(math.ceil(0.10 * n)))
            ups = g[g["up"]].sort_values("s", ascending=False).head(cap)
            k_now = int(g["p"].sum())
            dn_cap = min(cap, max(0, k_now + len(ups) - 1))
            dns = g[g["dn"]].sort_values("s").head(dn_cap)
            pred[ups["i"].to_numpy()] = 1; z2o += len(ups)
            pred[dns["i"].to_numpy()] = 0; o2z += len(dns)
        path = OUT / f"{name}.csv"
        pd.DataFrame({"id": ids, "prediction": pred}).to_csv(path, index=False)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        audit[name] = dict(precision_target=p, hi=hi, lo=lo, flips=int(z2o + o2z),
                           zero_to_one=int(z2o), one_to_zero=int(o2z),
                           positive_rate=round(float(pred.mean()), 6), sha256=h)
        log(f"{name}: hi={hi:.3f} lo={lo:.3f} +{z2o} -{o2z} rate={pred.mean():.4f}")
    audit["kaggle_submission_called"] = False
    (OUT / "blend_report.json").write_text(json.dumps(audit, indent=2))
    log("blend assemble done")


if __name__ == "__main__":
    main()
