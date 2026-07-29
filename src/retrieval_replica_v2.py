#!/usr/bin/env python3
"""Retrieval Replica V2 — learn to reproduce Trendyol's candidate membership.

Target is the OBSERVED retrieval membership (the items that appear as candidates
for a query), NOT relevance. This keeps the label fully observable and turns the
model into an independent, verifiable signal:

  * a good replica lets us, for the 1,807 queries with >100 candidates, isolate
    `observed - predicted_top100` = the items Trendyol force-added beyond its own
    retrieval, i.e. the probable "known positive that retrieval missed" set;
  * it also gives a calibrated retrieval-floor feature for the cardinality model.

Pipeline stages (each checkpointed; run with --stage):
  prepare       build query/item texts + observed membership + query-disjoint split
  embed         encode queries and the full catalog with the base model
  baseline      dense (pretrained) + lexical Recall@100 / Jaccard@100 / p10 vs observed
  mine          3-strategy hard-negative mining for contrastive training
  train         contrastive bi-encoder fine-tune (MultipleNegativesRanking + hard negs)
  eval          re-embed with the fine-tuned model; gate Recall@100 >= 0.85
  residual      n>100 groups: observed - predicted_top100 -> forced-positive signal

Colab: --model Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 --device cuda
Local smoke: --smoke --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

No Kaggle submission is ever made.
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_MODEL = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
SMOKE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["prepare", "embed", "baseline", "mine", "train", "eval", "residual", "all"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--work-dir", default="artifacts/retrieval_replica_v2")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--finetuned-model", default=None, help="path to fine-tuned model for eval/residual")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train-lr-batch", type=int, default=128)
    p.add_argument("--hard-negs-per-query", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--recall-gate", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--smoke", action="store_true",
                   help="tiny local run: subsample queries + catalog to prove correctness")
    p.add_argument("--smoke-queries", type=int, default=300)
    p.add_argument("--smoke-catalog", type=int, default=40000)
    p.add_argument("--max-train-examples", type=int, default=0,
                   help="cap contrastive examples (0 = no cap); use a small value for a fast smoke")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# text construction
# --------------------------------------------------------------------------- #
def item_text(row) -> str:
    parts = [str(row.get("title", "")), str(row.get("category", "")),
             str(row.get("brand", ""))]
    attrs = str(row.get("attributes", ""))
    if attrs and attrs != "nan":
        parts.append(attrs[:200])
    return " | ".join(p for p in parts if p and p != "nan")


def prepare(args) -> None:
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    terms = pd.read_csv(data / "terms.csv")
    sub = pd.read_csv(data / "submission_pairs.csv")
    log(f"terms={len(terms)} submission_pairs={len(sub)}")

    # observed membership per test term
    membership = sub.groupby("term_id")["item_id"].agg(list)
    counts = membership.map(len)
    test_terms = membership.index.to_numpy()
    log(f"test terms={len(test_terms)} exactly100={int((counts==100).sum())} gt100={int((counts>100).sum())}")

    items = pd.read_csv(data / "items.csv")
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        keep_terms = rng.choice(test_terms, size=min(args.smoke_queries, len(test_terms)), replace=False)
        keep_terms = set(keep_terms.tolist())
        membership = membership[membership.index.isin(keep_terms)]
        test_terms = membership.index.to_numpy()
        # catalog = all observed items of kept terms + random distractors (bounded)
        obs_items = set(i for lst in membership for i in lst)
        pool = items[~items["item_id"].isin(obs_items)]
        n_extra = max(0, args.smoke_catalog - len(obs_items))
        extra = pool.sample(n=min(n_extra, len(pool)), random_state=args.seed)["item_id"]
        keep_items = obs_items | set(extra.tolist())
        items = items[items["item_id"].isin(keep_items)].reset_index(drop=True)
        log(f"SMOKE: terms={len(test_terms)} catalog={len(items)} observed_items={len(obs_items)}")

    items = items.reset_index(drop=True)
    items["text"] = items.apply(item_text, axis=1)
    item_id_to_row = {i: k for k, i in enumerate(items["item_id"].to_numpy())}

    qmap = dict(zip(terms["term_id"], terms["query"]))
    membership_rows = {t: [item_id_to_row[i] for i in membership[t] if i in item_id_to_row]
                       for t in test_terms}
    # query-disjoint split for contrastive training
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(test_terms))
    n_val = int(len(test_terms) * args.val_frac)
    val_terms = set(test_terms[perm[:n_val]].tolist())

    items[["item_id", "text"]].to_parquet(work / "catalog.parquet")
    q_df = pd.DataFrame({
        "term_id": test_terms,
        "query": [str(qmap.get(t, "")) for t in test_terms],
        "n_candidates": [len(membership[t]) for t in test_terms],
        "split": ["val" if t in val_terms else "train" for t in test_terms],
    })
    q_df.to_parquet(work / "queries.parquet")
    with open(work / "membership_rows.json", "w") as f:
        json.dump({t: membership_rows[t] for t in test_terms.tolist()}, f)
    log(f"prepared: catalog={len(items)} queries={len(q_df)} "
        f"(train={int((q_df.split=='train').sum())} val={n_val})")


# --------------------------------------------------------------------------- #
# embedding + retrieval helpers
# --------------------------------------------------------------------------- #
def _load_model(name: str, device: str, max_seq: int):
    from sentence_transformers import SentenceTransformer
    kw = dict(device=device)
    try:
        m = SentenceTransformer(name, trust_remote_code=True, **kw)
    except TypeError:
        m = SentenceTransformer(name, **kw)
    m.max_seq_length = max_seq
    return m


def _encode(model, texts, batch_size):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float16)


def embed(args, model_name=None, suffix="base") -> None:
    work = Path(args.work_dir)
    model = _load_model(model_name or args.model, args.device, args.max_seq_length)
    cat = pd.read_parquet(work / "catalog.parquet")
    q = pd.read_parquet(work / "queries.parquet")
    log(f"encoding {len(q)} queries + {len(cat)} catalog items with {model_name or args.model}")
    np.save(work / f"emb_query_{suffix}.npy", _encode(model, q["query"].tolist(), args.batch_size))
    np.save(work / f"emb_item_{suffix}.npy", _encode(model, cat["text"].tolist(), args.batch_size))
    log(f"saved embeddings ({suffix})")


def topk_search(qemb: np.ndarray, iemb: np.ndarray, k: int, device: str, chunk: int = 512):
    """Return top-k item-row indices per query via chunked cosine (normalized inputs)."""
    import torch
    dev = device if (device == "cuda" and torch.cuda.is_available()) else (
        "mps" if (device == "mps" and torch.backends.mps.is_available()) else "cpu")
    I = torch.from_numpy(iemb.astype(np.float32)).to(dev)
    out = np.empty((len(qemb), k), dtype=np.int32)
    for s in range(0, len(qemb), chunk):
        Q = torch.from_numpy(qemb[s:s + chunk].astype(np.float32)).to(dev)
        sims = Q @ I.T
        idx = torch.topk(sims, k=min(k, I.shape[0]), dim=1).indices.cpu().numpy()
        out[s:s + chunk, :idx.shape[1]] = idx
    return out


def _metrics(pred_topk, membership_rows, term_ids, k):
    recalls, jacc = [], []
    for t in term_ids:
        obs = set(membership_rows[t])
        if not obs:
            continue
        pred = set(pred_topk[t][:k])
        inter = len(obs & pred)
        recalls.append(inter / len(obs))
        jacc.append(inter / len(obs | pred))
    recalls = np.array(recalls)
    return dict(recall_at_k=float(recalls.mean()),
                jaccard_at_k=float(np.mean(jacc)),
                p10_recall_at_k=float(np.percentile(recalls, 10)),
                n=len(recalls))


def _eval_embeddings(args, suffix, tag) -> dict:
    work = Path(args.work_dir)
    q = pd.read_parquet(work / "queries.parquet")
    membership_rows = {k: v for k, v in json.load(open(work / "membership_rows.json")).items()}
    qemb = np.load(work / f"emb_query_{suffix}.npy")
    iemb = np.load(work / f"emb_item_{suffix}.npy")
    pred = topk_search(qemb, iemb, args.topk, args.device)
    pred_by_term = {t: pred[i].tolist() for i, t in enumerate(q["term_id"])}
    val_terms = q[q.split == "val"]["term_id"].tolist()
    all_terms = q["term_id"].tolist()
    res = {"tag": tag,
           "val": _metrics(pred_by_term, membership_rows, val_terms, args.topk),
           "all": _metrics(pred_by_term, membership_rows, all_terms, args.topk)}
    log(f"{tag}: val Recall@{args.topk}={res['val']['recall_at_k']:.4f} "
        f"Jaccard={res['val']['jaccard_at_k']:.4f} p10={res['val']['p10_recall_at_k']:.4f}")
    return res


def lexical_baseline(args) -> dict:
    """Simple token-Jaccard lexical retrieval over the catalog (BM25-lite proxy)."""
    work = Path(args.work_dir)
    cat = pd.read_parquet(work / "catalog.parquet")
    q = pd.read_parquet(work / "queries.parquet")
    membership_rows = json.load(open(work / "membership_rows.json"))
    from collections import defaultdict
    # inverted index on catalog title tokens
    postings = defaultdict(list)
    cat_tokens = [set(str(t).lower().split()) for t in cat["text"]]
    for row, toks in enumerate(cat_tokens):
        for tok in toks:
            postings[tok].append(row)
    def retrieve(query):
        qt = set(str(query).lower().split())
        scores = defaultdict(int)
        for tok in qt:
            for row in postings.get(tok, ())[:20000]:
                scores[row] += 1
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda x: -x[1])[:args.topk]
        return [r for r, _ in top]
    pred_by_term = {t: retrieve(query) for t, query in zip(q["term_id"], q["query"])}
    val_terms = q[q.split == "val"]["term_id"].tolist()
    res = {"tag": "lexical",
           "val": _metrics(pred_by_term, membership_rows, val_terms, args.topk),
           "all": _metrics(pred_by_term, membership_rows, q["term_id"].tolist(), args.topk)}
    log(f"lexical: val Recall@{args.topk}={res['val']['recall_at_k']:.4f} "
        f"p10={res['val']['p10_recall_at_k']:.4f}")
    return res


def baseline(args) -> None:
    work = Path(args.work_dir)
    reports = {"dense_pretrained": _eval_embeddings(args, "base", "dense_pretrained"),
               "lexical": lexical_baseline(args)}
    (work / "baseline_report.json").write_text(json.dumps(reports, indent=2))
    log("baseline report written")


# --------------------------------------------------------------------------- #
# hard-negative mining (3 independent strategies)
# --------------------------------------------------------------------------- #
def mine(args) -> None:
    work = Path(args.work_dir)
    q = pd.read_parquet(work / "queries.parquet")
    cat = pd.read_parquet(work / "catalog.parquet")
    membership_rows = json.load(open(work / "membership_rows.json"))
    qemb = np.load(work / "emb_query_base.npy")
    iemb = np.load(work / "emb_item_base.npy")

    # near-miss negatives: catalog items ranked high but NOT observed for the query
    dense_top = topk_search(qemb, iemb, args.topk * 3, args.device)
    train_q = q[q.split == "train"].reset_index(drop=True)
    qi = {t: i for i, t in enumerate(q["term_id"])}

    # cross-query negatives: observed items of the nearest OTHER query
    qq_top = topk_search(qemb, qemb, 6, args.device)  # nearest queries (incl self)
    triples = []  # (query_text, pos_item_row, [hard_neg_item_rows])
    cat_cat = cat["text"].tolist()
    for _, r in train_q.iterrows():
        t = r["term_id"]; qrow = qi[t]
        obs = set(membership_rows[t])
        if not obs:
            continue
        near = [c for c in dense_top[qrow].tolist() if c not in obs][:args.hard_negs_per_query]
        # cross-query: pull observed items of the nearest different query
        cross = []
        for nq in qq_top[qrow].tolist():
            nt = q["term_id"].iloc[nq]
            if nt == t:
                continue
            cross = [c for c in membership_rows[nt] if c not in obs][:args.hard_negs_per_query]
            break
        hard = list(dict.fromkeys(near + cross))[:args.hard_negs_per_query]
        for pos in list(obs)[:args.topk]:
            triples.append((r["query"], pos, hard))
    with open(work / "train_triples.json", "w") as f:
        json.dump({"n": len(triples)}, f)
    np.save(work / "train_triples.npy", np.array(triples, dtype=object), allow_pickle=True)
    log(f"mined {len(triples)} (query,pos,hard_negs) training rows")


# --------------------------------------------------------------------------- #
# contrastive fine-tune
# --------------------------------------------------------------------------- #
def train(args) -> None:
    work = Path(args.work_dir)
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    cat = pd.read_parquet(work / "catalog.parquet")
    cat_text = cat["text"].tolist()
    triples = np.load(work / "train_triples.npy", allow_pickle=True)
    examples = []
    for qtext, pos_row, hard_rows in triples:
        # MultipleNegativesRankingLoss: (anchor, positive[, hard_negative])
        hn = cat_text[hard_rows[0]] if len(hard_rows) else None
        if hn is not None:
            examples.append(InputExample(texts=[qtext, cat_text[pos_row], hn]))
        else:
            examples.append(InputExample(texts=[qtext, cat_text[pos_row]]))
    if args.max_train_examples and len(examples) > args.max_train_examples:
        examples = examples[:args.max_train_examples]
    log(f"training on {len(examples)} examples")
    model = _load_model(args.model, args.device, args.max_seq_length)
    loader = DataLoader(examples, shuffle=True, batch_size=args.train_lr_batch)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(len(loader) * args.epochs * 0.1)
    out = str(work / "finetuned")
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
              warmup_steps=warmup, optimizer_params={"lr": args.lr},
              output_path=out, show_progress_bar=True)
    log(f"fine-tuned model saved -> {out}")


def eval_finetuned(args) -> None:
    work = Path(args.work_dir)
    ft = args.finetuned_model or str(work / "finetuned")
    embed(args, model_name=ft, suffix="ft")
    res = _eval_embeddings(args, "ft", "dense_finetuned")
    base = json.loads((work / "baseline_report.json").read_text()) if (work / "baseline_report.json").exists() else {}
    gate_pass = res["val"]["recall_at_k"] >= args.recall_gate
    report = {"finetuned": res, "baseline": base, "recall_gate": args.recall_gate,
              "gate_pass": gate_pass,
              "gain_vs_dense_pretrained": (res["val"]["recall_at_k"]
                  - base.get("dense_pretrained", {}).get("val", {}).get("recall_at_k", float("nan")))
                  if base else None}
    (work / "finetuned_report.json").write_text(json.dumps(report, indent=2))
    log(f"GATE Recall@{args.topk}>={args.recall_gate}: {'PASS' if gate_pass else 'FAIL'} "
        f"(val={res['val']['recall_at_k']:.4f})")
    if not gate_pass:
        log("Recall gate not met -> do NOT use the retrieval residual at scale (per project rule).")


def residual(args) -> None:
    """For n>100 queries, observed - predicted_top100 = forced-positive signal."""
    work = Path(args.work_dir)
    suffix = "ft" if (work / "emb_query_ft.npy").exists() else "base"
    gate_path = work / "finetuned_report.json"
    if suffix == "ft" and gate_path.exists():
        gate = json.loads(gate_path.read_text())
        if not bool(gate.get("gate_pass", False)):
            raise RuntimeError(
                f"Recall@{args.topk} gate failed; refusing to emit retrieval residual. "
                f"See {gate_path}."
            )
    elif suffix == "base":
        raise RuntimeError(
            "No gated fine-tuned retrieval model is available; base residual is "
            "not allowed for wide-scale candidate merging."
        )
    q = pd.read_parquet(work / "queries.parquet")
    cat = pd.read_parquet(work / "catalog.parquet")
    membership_rows = json.load(open(work / "membership_rows.json"))
    sub = pd.read_csv(Path(args.data_dir) / "submission_pairs.csv", usecols=["id", "term_id", "item_id"])
    id_lookup = {(str(t), str(i)): str(uid) for uid, t, i in sub.itertuples(index=False, name=None)}
    qemb = np.load(work / f"emb_query_{suffix}.npy")
    iemb = np.load(work / f"emb_item_{suffix}.npy")
    pred = topk_search(qemb, iemb, args.topk, args.device)
    item_ids = cat["item_id"].to_numpy()
    rows = []
    for i, t in enumerate(q["term_id"]):
        if q["n_candidates"].iloc[i] <= args.topk:
            continue
        obs = set(membership_rows[t]); pset = set(pred[i].tolist())
        for r in obs - pset:                       # observed but NOT in our top-100 = forced
            item_id = str(item_ids[r])
            rows.append((id_lookup.get((str(t), item_id), ""), t, item_id))
    out = pd.DataFrame(rows, columns=["id", "term_id", "item_id"])
    out = out[out["id"] != ""].reset_index(drop=True)
    out.to_parquet(work / f"forced_positive_residual_{suffix}.parquet")
    log(f"forced-positive residual ({suffix}): {len(out)} rows over "
        f"{out['term_id'].nunique() if len(out) else 0} n>100 queries")


def main() -> None:
    args = parse_args()
    if args.smoke and args.model == DEFAULT_MODEL:
        args.model = SMOKE_MODEL
    stages = ["prepare", "embed", "baseline", "mine", "train", "eval", "residual"] \
        if args.stage == "all" else [args.stage]
    for st in stages:
        log(f"=== stage: {st} ===")
        {"prepare": prepare, "embed": embed, "baseline": baseline, "mine": mine,
         "train": train, "eval": eval_finetuned, "residual": residual}[st](args)
    print("Kaggle submission called: False")


if __name__ == "__main__":
    main()
