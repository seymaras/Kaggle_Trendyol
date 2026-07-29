#!/usr/bin/env python3
"""LLM-teacher cross-encoder distillation — scores ALL 3,359,679 test pairs.

The only in-hand approach whose theoretical ceiling exceeds the ~0.92 band:
511k LLM referee decisions (Qwen3-30B + Mistral-24B, rounds 1+2) become teacher
labels; a Turkish cross-encoder is fine-tuned on them (query-disjoint split) and
then scores every test pair. Candidates are assembled CONSERVATIVELY on top of
the proven anchor: the student may only flip rows where it is very confident and
never overrides a strong LLM or train-transfer verdict. Per-query flip caps
prevent cardinality explosions.

Stages (checkpointed to --work-dir; rerun skips finished stages):
  teacher    build teacher_labels.parquet from vote parquets (+ transfer golds)
  train      fine-tune the cross-encoder (bf16, AdamW) on teacher labels
  score      score all test pairs -> student_scores_ce.parquet
  assemble   build conservative/medium/aggressive CSVs vs the anchor + audit

NO Kaggle submission is ever made.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["teacher", "train", "score", "assemble", "all"])
    p.add_argument("--input-dir", required=True, help="dir with pairs.parquet, items.parquet, terms.parquet, votes/, anchor.parquet, transfer.parquet")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--model", default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-len", type=int, default=160)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-frac", type=float, default=0.06)
    p.add_argument("--val-pct", type=int, default=10, help="held-out terms, stable hash split")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--max-steps", type=int, default=0, help="cap steps (smoke)")
    p.add_argument("--limit-pairs", type=int, default=0, help="score only first N pairs (smoke)")
    p.add_argument("--hi", type=float, default=0.92, help="0->1 flip needs score >= hi (medium tier)")
    p.add_argument("--lo", type=float, default=0.08, help="1->0 flip needs score <= lo (medium tier)")
    # noisy-student v2: feed a previous student's confident scores back as pseudo-labels
    p.add_argument("--pseudo-scores", default=None, help="parquet with id, ce_score from a prior run")
    p.add_argument("--pseudo-hi", type=float, default=0.985)
    p.add_argument("--pseudo-lo", type=float, default=0.015)
    p.add_argument("--pseudo-weight", type=float, default=0.35)
    p.add_argument("--pseudo-max", type=int, default=1_500_000)
    return p.parse_args()

def term_split(term_id: str, val_pct: int) -> str:
    h = int(hashlib.sha1(term_id.encode()).hexdigest()[:8], 16) % 100
    return "val" if h < val_pct else "train"

def load_votes(d, cols=("id", "label", "confidence")):
    fs = sorted(glob.glob(str(Path(d) / "part_*.parquet")))
    if not fs: return None
    return pd.concat([pd.read_parquet(p, columns=list(cols)) for p in fs], ignore_index=True)

# ---------------------------------------------------------------- teacher ----
def build_teacher(args):
    IN, W = Path(args.input_dir), Path(args.work_dir)
    W.mkdir(parents=True, exist_ok=True)
    out = W / "teacher_labels.parquet"
    if out.exists(): log("teacher: checkpoint exists, skip"); return
    votes = {}
    for tag in ["r1_qwen", "r1_mistral", "r2_qwen", "r2_mistral", "r2_gpt"]:
        v = load_votes(IN / "votes" / tag)
        if v is not None:
            votes[tag] = v
            log(f"teacher: {tag} {len(v)} votes")
    frames = []
    for tag, v in votes.items():
        f = v[["id", "label", "confidence"]].copy(); f["src"] = tag
        frames.append(f)
    allv = pd.concat(frames, ignore_index=True)
    g = allv.groupby("id").agg(n=("label", "size"), pos=("label", "sum"),
                               conf=("confidence", "mean"))
    unanimous = g[(g["pos"] == 0) | (g["pos"] == g["n"])].copy()
    unanimous["label"] = (unanimous["pos"] > 0).astype(np.int8)
    unanimous["weight"] = unanimous["conf"].clip(0.5, 1.0) * (1.0 + 0.25 * (unanimous["n"] - 1))
    teacher = unanimous.reset_index()[["id", "label", "weight"]]
    dropped = len(g) - len(unanimous)
    # train-transfer golds (label=1, strong weight)
    tf = IN / "transfer.parquet"
    if tf.exists():
        t = pd.read_parquet(tf)[["id"]].assign(label=np.int8(1), weight=np.float32(2.5))
        teacher = pd.concat([teacher[~teacher["id"].isin(set(t["id"]))], t], ignore_index=True)
        log(f"teacher: +{len(t)} transfer golds")
    teacher["pseudo"] = False
    # noisy-student: confident scores from a prior run become weak extra labels
    if args.pseudo_scores:
        ps = pd.read_parquet(args.pseudo_scores)
        ps = ps[~ps["id"].isin(set(teacher["id"]))]
        conf = ps[(ps["ce_score"] >= args.pseudo_hi) | (ps["ce_score"] <= args.pseudo_lo)].copy()
        if len(conf) > args.pseudo_max:
            conf = conf.sample(n=args.pseudo_max, random_state=13)
        conf["label"] = (conf["ce_score"] >= 0.5).astype(np.int8)
        conf["weight"] = np.float32(args.pseudo_weight)
        conf["pseudo"] = True
        teacher = pd.concat([teacher, conf[["id", "label", "weight", "pseudo"]]], ignore_index=True)
        log(f"teacher: +{len(conf)} pseudo-labels (|hi>={args.pseudo_hi} lo<={args.pseudo_lo}| "
            f"w={args.pseudo_weight}) pos_rate={conf['label'].mean():.3f}")
    pairs = pd.read_parquet(IN / "pairs.parquet", columns=["id", "term_id"])
    teacher = teacher.merge(pairs, on="id", how="inner")
    teacher["split"] = teacher["term_id"].map(lambda t: term_split(str(t), args.val_pct))
    # pseudo rows never enter validation (they are not ground truth)
    teacher.loc[teacher["pseudo"] & (teacher["split"] == "val"), "split"] = "train"
    teacher.to_parquet(out)
    real = teacher[~teacher["pseudo"]]
    log(f"teacher: {len(teacher)} rows ({len(real)} real, dropped {dropped} conflicted) "
        f"pos_rate={teacher['label'].mean():.3f} val={int((teacher.split=='val').sum())}")

# ------------------------------------------------------------------ texts ----
def build_pair_text(IN, ids=None):
    pairs = pd.read_parquet(IN / "pairs.parquet")
    if ids is not None:
        pairs = pairs[pairs["id"].isin(set(ids))]
    items = pd.read_parquet(IN / "items.parquet")   # item_id, text
    terms = pd.read_parquet(IN / "terms.parquet")   # term_id, query
    df = pairs.merge(terms, on="term_id").merge(items, on="item_id")
    return df  # id, term_id, item_id, query, text

# ------------------------------------------------------------------ train ----
def train(args):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    IN, W = Path(args.input_dir), Path(args.work_dir)
    model_dir = W / "ce_model"
    if (model_dir / "config.json").exists(): log("train: checkpoint exists, skip"); return
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    teacher = pd.read_parquet(W / "teacher_labels.parquet")
    txt = build_pair_text(IN, ids=teacher["id"])
    df = teacher.merge(txt[["id", "query", "text"]], on="id")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    log(f"train: {len(tr)} train / {len(va)} val (query-disjoint)")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    dev = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    model.to(dev)
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()

    class DS(Dataset):
        def __init__(s, d): s.q = d["query"].tolist(); s.t = d["text"].tolist(); \
                            s.y = d["label"].to_numpy(np.float32); s.w = d["weight"].to_numpy(np.float32)
        def __len__(s): return len(s.q)
        def __getitem__(s, i): return s.q[i], s.t[i], s.y[i], s.w[i]

    def collate(b):
        q, t, y, w = zip(*b)
        enc = tok(list(q), list(t), truncation=True, max_length=args.max_len,
                  padding=True, return_tensors="pt")
        return enc, torch.tensor(y), torch.tensor(w)

    dl = DataLoader(DS(tr), batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                    num_workers=0, drop_last=True)
    steps = args.max_steps or len(dl) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(steps * args.warmup_frac), steps)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    model.train(); step = 0; t0 = time.time()
    for ep in range(args.epochs):
        for enc, y, w in dl:
            enc = {k: v.to(dev) for k, v in enc.items()}; y, w = y.to(dev), w.to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**enc).logits.squeeze(-1)
                loss = (lossf(logits.float(), y) * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(); step += 1
            if step % 200 == 0:
                log(f"  ep{ep} step {step}/{steps} loss={loss.item():.4f} "
                    f"({step/(time.time()-t0):.1f} it/s)")
            if args.max_steps and step >= args.max_steps: break
        if args.max_steps and step >= args.max_steps: break
    # query-disjoint validation
    model.eval(); preds = []
    vdl = DataLoader(DS(va), batch_size=args.eval_batch_size, collate_fn=collate)
    with torch.no_grad():
        for enc, y, w in vdl:
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                preds.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy())
    pv = np.concatenate(preds) if preds else np.zeros(0)
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    yv = va["label"].to_numpy()
    metrics = dict(rows=int(len(va)),
                   auc=float(roc_auc_score(yv, pv)) if len(set(yv)) > 1 else None,
                   ap=float(average_precision_score(yv, pv)) if len(set(yv)) > 1 else None,
                   f1_at_05=float(f1_score(yv, pv >= 0.5)) if len(set(yv)) > 1 else None)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir); tok.save_pretrained(model_dir)
    (W / "train_report.json").write_text(json.dumps(metrics, indent=2))
    log(f"train done: {metrics}")

# ------------------------------------------------------------------ score ----
def score(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    IN, W = Path(args.input_dir), Path(args.work_dir)
    out = W / "student_scores_ce.parquet"
    if out.exists(): log("score: checkpoint exists, skip"); return
    tok = AutoTokenizer.from_pretrained(W / "ce_model")
    model = AutoModelForSequenceClassification.from_pretrained(W / "ce_model")
    dev = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    df = build_pair_text(IN)
    if args.limit_pairs: df = df.head(args.limit_pairs)
    log(f"score: {len(df)} pairs")
    scores = np.zeros(len(df), dtype=np.float32); B = args.eval_batch_size
    part = W / "score_partial.npy"; start = 0
    if part.exists():
        prev = np.load(part); scores[:len(prev)] = prev; start = len(prev)
        start = (start // B) * B
        log(f"score: resuming at {start}")
    q = df["query"].tolist(); t = df["text"].tolist(); t0 = time.time()
    with torch.no_grad():
        for s in range(start, len(df), B):
            enc = tok(q[s:s+B], t[s:s+B], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                scores[s:s+B] = torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy()
            if (s // B) % 200 == 0 and s > start:
                rate = (s - start) / max(1e-9, time.time() - t0)
                log(f"  {s}/{len(df)} ({rate:.0f} rows/s, ETA {(len(df)-s)/max(rate,1)/60:.0f} min)")
                np.save(part, scores[:s+B])
    pd.DataFrame({"id": df["id"].to_numpy(), "ce_score": scores}).to_parquet(out)
    if part.exists(): part.unlink()
    log(f"score done -> {out}")

# --------------------------------------------------------------- assemble ----
def assemble(args):
    IN, W = Path(args.input_dir), Path(args.work_dir)
    anchor = pd.read_parquet(IN / "anchor.parquet")            # id, prediction (proven best)
    sc = pd.read_parquet(W / "student_scores_ce.parquet")
    teacher = pd.read_parquet(W / "teacher_labels.parquet")
    pairs = pd.read_parquet(IN / "pairs.parquet", columns=["id", "term_id"])
    df = anchor.merge(sc, on="id", how="left").merge(pairs, on="id")
    df["ce_score"] = df["ce_score"].fillna(0.5)
    strong = teacher[teacher["weight"] >= 0.9]
    lock_pos = set(strong[strong["label"] == 1]["id"])          # never 1->0
    lock_neg = set(strong[strong["label"] == 0]["id"])          # never 0->1
    tiers = {"ce_conservative": (0.96, 0.04), "ce_medium": (args.hi, args.lo),
             "ce_aggressive": (0.88, 0.12)}
    ids = df["id"].to_numpy(); base = df["prediction"].to_numpy(np.uint8)
    scr = df["ce_score"].to_numpy(np.float32); term = df["term_id"].to_numpy()
    audit = {}
    for name, (hi, lo) in tiers.items():
        pred = base.copy()
        want_up = (base == 0) & (scr >= hi) & ~pd.Series(ids).isin(lock_neg).to_numpy()
        want_dn = (base == 1) & (scr <= lo) & ~pd.Series(ids).isin(lock_pos).to_numpy()
        z2o = o2z = 0
        gdf = pd.DataFrame({"i": np.arange(len(ids)), "term": term,
                            "up": want_up, "dn": want_dn, "s": scr, "p": base})
        for t, g in gdf.groupby("term", sort=False):
            n = len(g); cap = max(3, int(math.ceil(0.10 * n)))
            ups = g[g["up"]].sort_values("s", ascending=False).head(cap)
            k_now = int(g["p"].sum())
            dn_cap = min(cap, max(0, k_now + len(ups) - 1))     # keep k >= 1
            dns = g[g["dn"]].sort_values("s").head(dn_cap)
            pred[ups["i"].to_numpy()] = 1; z2o += len(ups)
            pred[dns["i"].to_numpy()] = 0; o2z += len(dns)
        path = W / f"{name}.csv"
        pd.DataFrame({"id": ids, "prediction": pred}).to_csv(path, index=False)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        audit[name] = dict(hi=hi, lo=lo, flips=int(z2o + o2z), zero_to_one=int(z2o),
                           one_to_zero=int(o2z), positive_rate=round(float(pred.mean()), 6),
                           sha256=h)
        log(f"{name}: +{z2o} -{o2z} rate={pred.mean():.4f}")
    audit["kaggle_submission_called"] = False
    tr = W / "train_report.json"
    audit["validation"] = json.loads(tr.read_text()) if tr.exists() else None
    (W / "assemble_report.json").write_text(json.dumps(audit, indent=2))
    log("assemble done")

def main():
    a = parse_args()
    for st in (["teacher", "train", "score", "assemble"] if a.stage == "all" else [a.stage]):
        log(f"=== {st} ==="); {"teacher": build_teacher, "train": train,
                              "score": score, "assemble": assemble}[st](a)
    print("Kaggle submission called: False")

if __name__ == "__main__":
    main()
