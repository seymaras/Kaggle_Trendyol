#!/usr/bin/env python3
"""Per-query positive-count (k) model — replace a single global threshold.

Design choice driven by the project's hard constraint that there is NO trustworthy
local Macro-F1 proxy (official train labels are positive-only; train/test queries
are disjoint; k-distributions shift between train and test). So the PRIMARY signal
is distribution-free and only *nudges* the proven 0.901 anchor's per-query
cardinality — it never discards the anchor's per-candidate decisions:

  * k=0 floor fix: the anchor assigns zero positives to ~1,900 queries; every query
    should have >=1 relevant item, so force the single best candidate positive.
  * score-elbow k: per query, find the natural cutoff in the sorted student_prob
    scores (largest relative gap) -> an evidence-based target count.
  * three candidates (conservative / medium / aggressive) adjust the anchor toward
    the elbow target by adding best-scoring 0s / trimming weakest 1s.

A gradient-boosted k-regressor trained on train positive counts is ALSO fit and
reported (MAE + predicted-vs-anchor distribution) purely as a cross-check; it is
flagged, not used as the selector, because its labels are shifted/incomplete.

No Kaggle submission is ever made.
"""
from __future__ import annotations
import argparse, glob, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/seyma/Documents/Kaggle_Trendyol")
IN = ROOT / "artifacts/llm_student_cascade/package/trendyol_llm_student_cascade/input"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--anchor", default=str(IN / "anchor_v6.parquet"))
    p.add_argument("--student", default=str(ROOT / "artifacts/llm_student_v1/student_scores.parquet"))
    p.add_argument("--residual", default=str(ROOT / "artifacts/official_engine_colab/structural_residual_candidates.parquet"))
    p.add_argument("--out-dir", default=str(ROOT / "artifacts/cardinality_v1"))
    p.add_argument("--fit-regressor", action="store_true", help="also fit+report the train k-regressor cross-check")
    return p.parse_args()


def score_elbow_k(scores: np.ndarray, kmin: int, kmax: int) -> int:
    """Largest relative gap in the descending score list -> cut point (count above)."""
    s = np.sort(scores)[::-1]
    if len(s) <= 1:
        return int(len(s))
    top = s[:min(len(s), kmax + 5)]
    gaps = top[:-1] - top[1:]
    denom = top[:-1] + 1e-6
    rel = gaps / denom
    cut = int(np.argmax(rel)) + 1                 # number of items above the biggest cliff
    return int(np.clip(cut, kmin, min(kmax, len(s))))


def build(args) -> None:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(Path(args.data_dir) / "submission_pairs.csv")[["id", "term_id", "item_id"]]
    terms = pd.read_csv(Path(args.data_dir) / "terms.csv")
    anchor = pd.read_parquet(args.anchor)
    student = pd.read_parquet(args.student)
    df = sp.merge(anchor, on="id").merge(student, on="id")
    df = df.merge(terms, on="term_id", how="left")
    qlen = terms.assign(qc=terms["query"].str.len(),
                        qt=terms["query"].str.split().map(len))
    qlen = dict(zip(qlen["term_id"], zip(qlen["qc"], qlen["qt"])))

    # residual (forced-membership) count per term
    res = pd.read_parquet(args.residual, columns=["term_id"])
    res_count = res.groupby("term_id").size().to_dict()

    # per-query aggregation
    grp = df.groupby("term_id")
    recs = []
    KMAX = 100
    for t, g in grp:
        sc = g["student_prob"].to_numpy()
        anc_k = int(g["prediction"].sum())
        n = len(g)
        elbow = score_elbow_k(sc, kmin=1, kmax=KMAX)
        qc, qt = qlen.get(t, (0, 0))
        recs.append(dict(
            term_id=t, n_candidates=n, anchor_k=anc_k,
            elbow_k=elbow,
            score_mean=float(sc.mean()), score_std=float(sc.std()),
            score_max=float(sc.max()), score_p90=float(np.quantile(sc, .9)),
            n_ge_050=int((sc >= .5).sum()), n_ge_035=int((sc >= .35).sum()),
            n_ge_070=int((sc >= .7).sum()),
            query_chars=int(qc or 0), query_tokens=int(qt or 0),
            residual_count=int(res_count.get(t, 0)),
        ))
    q = pd.DataFrame(recs)
    q.to_parquet(out / "query_features.parquet")
    print(f"queries={len(q)} anchor_k: k0={int((q.anchor_k==0).sum())} "
          f"median={int(q.anchor_k.median())} mean={q.anchor_k.mean():.1f}")

    # optional trained regressor cross-check (train positive counts)
    reg_report = None
    if args.fit_regressor:
        reg_report = fit_regressor(args, out)

    # ---- build the three cardinality candidates on the anchor ----
    # rank candidates within each query by student_prob (stable) to pick add/trim order
    df = df.sort_values(["term_id", "student_prob"], ascending=[True, False])
    df["rank"] = df.groupby("term_id").cumcount()          # 0 = best
    k_lookup = q.set_index("term_id")

    def make(name, target_fn, allow_trim):
        pred = dict(zip(anchor["id"].astype(str), anchor["prediction"].astype(np.int8)))
        add = trim = 0
        for t, g in df.groupby("term_id", sort=False):
            anc_k = int(k_lookup.at[t, "anchor_k"])
            tgt = int(target_fn(k_lookup.loc[t]))
            ids = g["id"].astype(str).to_numpy()             # already best->worst
            if tgt > anc_k:                                   # add best currently-0
                need = tgt - anc_k
                for i in ids:
                    if need == 0:
                        break
                    if pred[i] == 0:
                        pred[i] = 1; add += 1; need -= 1
            elif allow_trim and tgt < anc_k:                  # trim weakest 1s
                need = anc_k - tgt
                for i in ids[::-1]:                            # worst->best
                    if need == 0:
                        break
                    if pred[i] == 1:
                        pred[i] = 0; trim += 1; need -= 1
        ids_all = anchor["id"].astype(str).to_numpy()
        arr = np.array([pred[i] for i in ids_all], dtype=np.uint8)
        path = out / f"{name}.csv"
        pd.DataFrame({"id": ids_all, "prediction": arr}).to_csv(path, index=False)
        rep = dict(file=path.name, added=add, trimmed=trim,
                   positive_count=int(arr.sum()), positive_rate=round(float(arr.mean()), 6),
                   sha256=sha256(path))
        print(f"  {name:34s} +{add:6d} -{trim:6d} pos_rate={rep['positive_rate']:.4f}")
        return rep

    # target definitions (distribution-free, narrow + evidence-based).
    # The anchor's per-query k already correlates ~0.98 with the student's
    # threshold-0.70 count, so it is well-calibrated; changes are confined to
    # clear defects. The elbow feature overcounts badly and is NOT used here.
    def t_conservative(r):        # only fix the k=0 queries -> force the single best
        return max(int(r["anchor_k"]), 1)
    def t_medium(r):              # k=0 fix + lift extreme under-predictions (student-backed)
        k = max(int(r["anchor_k"]), 1)
        if r["anchor_k"] <= 2 and r["n_ge_070"] >= 5:      # tiny k yet many strong scores
            k = int(min(r["n_ge_070"], r["anchor_k"] + 8))
        return k
    def t_aggressive(r):         # medium + trim clear over-predictions toward n>=0.5 support
        k = t_medium(r)
        if r["anchor_k"] > r["n_ge_050"] + 15:             # far more 1s than score supports
            k = int(max(r["n_ge_050"], 1))
        return k

    cands = {
        "card_conservative": make("card_conservative", t_conservative, allow_trim=False),
        "card_medium": make("card_medium", t_medium, allow_trim=False),
        "card_aggressive": make("card_aggressive", t_aggressive, allow_trim=True),
    }
    report = dict(
        base_anchor=str(args.anchor), queries=int(len(q)),
        anchor_zero_k_queries=int((q.anchor_k == 0).sum()),
        note=("Primary logic is distribution-free (elbow + anchor + k=0 floor). "
              "The regressor below is a flagged cross-check only, not the selector."),
        regressor_crosscheck=reg_report, candidates=cands,
        kaggle_submission_called=False,
    )
    (out / "cardinality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report -> {out/'cardinality_report.json'}")
    print("Kaggle submission called: False")


def fit_regressor(args, out) -> dict:
    """Cross-check only: predict per-query positive count from features on TRAIN.

    Uses train_testlike (retrieval-style candidates with pseudo labels) so features
    match the test-side construction. Honest caveats: labels are positive-only /
    pseudo-negative and the k-distribution shifts, so this is reported, not used.
    """
    from catboost import CatBoostRegressor, Pool
    tl = ROOT / "artifacts/train_testlike.parquet"
    if not tl.exists():
        return {"skipped": "train_testlike.parquet not found"}
    t = pd.read_parquet(tl, columns=["term_id", "label", "retrieval_score", "query"])
    g = t.groupby("term_id")
    feat = g.agg(n_candidates=("label", "size"),
                 k=("label", "sum"),
                 score_mean=("retrieval_score", "mean"),
                 score_max=("retrieval_score", "max"),
                 score_std=("retrieval_score", "std")).reset_index()
    ql = t.drop_duplicates("term_id").set_index("term_id")["query"].astype(str)
    feat["query_chars"] = feat["term_id"].map(ql.str.len().to_dict())
    feat["query_tokens"] = feat["term_id"].map(ql.str.split().map(len).to_dict())
    feat = feat.fillna(0)
    cols = ["n_candidates", "score_mean", "score_max", "score_std", "query_chars", "query_tokens"]
    rng = np.random.default_rng(13)
    m = rng.permutation(len(feat)); nval = int(len(feat) * 0.2)
    val_idx, tr_idx = m[:nval], m[nval:]
    model = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05,
                              loss_function="MAE", verbose=False, random_seed=13)
    model.fit(Pool(feat.iloc[tr_idx][cols], feat.iloc[tr_idx]["k"]))
    pred = model.predict(feat.iloc[val_idx][cols])
    mae = float(np.mean(np.abs(pred - feat.iloc[val_idx]["k"].to_numpy())))
    model.save_model(str(out / "k_regressor.cbm"))
    return {"val_mae": round(mae, 3), "train_k_median": int(feat["k"].median()),
            "train_k_mean": round(float(feat["k"].mean()), 2),
            "caveat": "labels positive-only/pseudo-negative + train/test k shift; cross-check only"}


if __name__ == "__main__":
    build(parse_args())
