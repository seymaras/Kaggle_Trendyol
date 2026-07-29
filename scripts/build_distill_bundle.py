#!/usr/bin/env python3
"""Package all inputs for the cross-encoder distillation Colab run.

Produces artifacts/distill_bundle/trendyol_distill_bundle.zip containing:
  pairs.parquet     id, term_id, item_id (all 3,359,679 test pairs)
  items.parquet     item_id, text (only items used by test pairs; compact text)
  terms.parquet     term_id, query
  anchor.parquet    id, prediction  == best proven submission (UPDATE AS LB IMPROVES)
  transfer.parquet  train-transfer gold positives
  votes/r1_qwen, r1_mistral, r2_qwen, r2_mistral (+ r2_gpt when available)
  src/distill_cross_encoder.py + manifest.json (SHA-256 of every file)
"""
from __future__ import annotations
import glob, hashlib, json, shutil, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/seyma/Documents/Kaggle_Trendyol")
OUT = ROOT / "artifacts/distill_bundle"
STAGE = OUT / "stage"
# The only locally proven 0.915 anchor. Do not point distillation at a
# max-jump/transfer candidate: its 0.884 leaderboard result showed that those
# broad stacks can poison both assembly and any downstream interpretation.
ANCHOR_CSV = ROOT / "artifacts/merged_candidates_v1/01_llm_qwen_mistral_strict.csv"

VOTE_SRC = {
    "r1_qwen": ROOT / "artifacts/llm_judge_v1/drive_votes/qwen",
    "r1_mistral": ROOT / "artifacts/llm_judge_v1/drive_votes/mistral",
    "r2_qwen": ROOT / "artifacts/llm_student_cascade/votes/qwen",
    "r2_mistral": ROOT / "artifacts/llm_student_cascade/votes/mistral",
    "r2_gpt": ROOT / "artifacts/llm_student_cascade/votes/gpt_oss_20b",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "votes").mkdir(parents=True)
    (STAGE / "src").mkdir(parents=True)

    sp = pd.read_csv(ROOT / "data/submission_pairs.csv")
    sp.to_parquet(STAGE / "pairs.parquet")
    print(f"pairs: {len(sp)}")

    items = pd.read_csv(ROOT / "data/items.csv")
    items = items[items["item_id"].isin(set(sp["item_id"]))].reset_index(drop=True)
    def text(r):
        parts = [str(r["title"]), str(r["category"]), str(r["brand"])]
        a = str(r["attributes"])
        if a and a != "nan":
            parts.append(a[:180])
        return " | ".join(x for x in parts if x and x != "nan")
    items["text"] = items.apply(text, axis=1)
    items[["item_id", "text"]].to_parquet(STAGE / "items.parquet")
    print(f"items: {len(items)}")

    terms = pd.read_csv(ROOT / "data/terms.csv")
    terms = terms[terms["term_id"].isin(set(sp["term_id"]))]
    terms.to_parquet(STAGE / "terms.parquet")
    print(f"terms: {len(terms)}")

    anc = pd.read_csv(ANCHOR_CSV)
    assert len(anc) == len(sp)
    anc.to_parquet(STAGE / "anchor.parquet")
    print(f"anchor: {ANCHOR_CSV.name} pos_rate={anc['prediction'].mean():.4f}")

    tf = ROOT / "artifacts/train_transfer_hits.parquet"
    if tf.exists():
        pd.read_parquet(tf)[["id"]].to_parquet(STAGE / "transfer.parquet")
        print("transfer golds copied")

    for tag, src in VOTE_SRC.items():
        parts = sorted(glob.glob(str(src / "part_*.parquet")))
        if not parts:
            print(f"votes {tag}: MISSING (ok if r2_gpt)")
            continue
        d = STAGE / "votes" / tag
        d.mkdir(parents=True)
        for p in parts:
            shutil.copy2(p, d / Path(p).name)
        print(f"votes {tag}: {len(parts)} parts")

    shutil.copy2(ROOT / "src/distill_cross_encoder.py", STAGE / "src/distill_cross_encoder.py")

    manifest = {"anchor_source": ANCHOR_CSV.name,
                "files": {str(p.relative_to(STAGE)): sha256(p)
                          for p in sorted(STAGE.rglob("*")) if p.is_file()}}
    (STAGE / "manifest.json").write_text(json.dumps(manifest, indent=2))

    zpath = OUT / "trendyol_distill_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE))
    print(f"\nbundle -> {zpath} ({zpath.stat().st_size/2**20:.0f} MiB)")
    print("Kaggle submission called: False")


if __name__ == "__main__":
    main()
