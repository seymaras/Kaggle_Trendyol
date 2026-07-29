#!/usr/bin/env python3
"""Önceden eğitilmiş MiniLM ile 3 cosine-similarity feature üretir (fine-tune YOK).

Cold-start'ta lexical (TF-IDF) sinyal zayıfken semantik sinyal sağlar. Her
(term_id, item_id) çifti için query embedding'i ile şu 3 metin embedding'i
arasındaki cosine benzerliği hesaplanır: title, category, item_text (tam
ürün metni). Çıktı, mevcut feature setine (term_id, item_id) üzerinden
join edilebilecek bir parquet dosyasıdır.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_features import make_item_text, normalize_category_path, normalize_text  # noqa: E402

ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
ROOT = SRC_DIR.parent
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--train-pairs", type=Path, default=None,
        help="term_id,item_id kolonlu train parquet (varsayılan: artifacts/train_testlike.parquet)",
    )
    parser.add_argument(
        "--test-pairs", type=Path, default=None,
        help="term_id,item_id kolonlu test csv/parquet (varsayılan: data/submission_pairs.csv)",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Cosine hesaplama için satır grup boyutu")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_pairs(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=["term_id", "item_id"])
    else:
        frame = pd.read_csv(path, usecols=["term_id", "item_id"], dtype={"term_id": "string", "item_id": "string"})
    frame["term_id"] = frame["term_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    return frame.drop_duplicates().reset_index(drop=True)


def encode_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32)


def build_features(pairs: pd.DataFrame, terms: pd.DataFrame, items: pd.DataFrame, model, args) -> pd.DataFrame:
    term_ids = sorted(pairs["term_id"].unique())
    term_pos = {t: i for i, t in enumerate(term_ids)}
    query_texts = terms.set_index("term_id").reindex(term_ids)["query"].fillna("").tolist()
    print(f"  Query embedding: {len(query_texts):,} benzersiz sorgu")
    query_emb = encode_texts(model, query_texts, args.batch_size)

    item_ids = sorted(pairs["item_id"].unique())
    item_pos = {it: i for i, it in enumerate(item_ids)}
    item_meta = items.set_index("item_id").reindex(item_ids)
    if item_meta[["title", "category", "item_text"]].isna().any().any():
        raise RuntimeError("Bazı item_id'ler için katalog eşleşmesi bulunamadı")

    print(f"  Title embedding: {len(item_ids):,} benzersiz ürün")
    title_emb = encode_texts(model, item_meta["title"].tolist(), args.batch_size)
    print(f"  Category embedding: {len(item_ids):,} benzersiz ürün")
    category_texts = item_meta["category"].str.replace("/", " ", regex=False).tolist()
    category_emb = encode_texts(model, category_texts, args.batch_size)
    print(f"  Item-text embedding: {len(item_ids):,} benzersiz ürün")
    item_text_emb = encode_texts(model, item_meta["item_text"].tolist(), args.batch_size)

    term_idx = pairs["term_id"].map(term_pos).to_numpy()
    item_idx = pairs["item_id"].map(item_pos).to_numpy()

    cos_title = np.empty(len(pairs), dtype=np.float32)
    cos_category = np.empty(len(pairs), dtype=np.float32)
    cos_item_text = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), args.chunk_size):
        end = min(start + args.chunk_size, len(pairs))
        q = query_emb[term_idx[start:end]]
        cos_title[start:end] = np.einsum("ij,ij->i", q, title_emb[item_idx[start:end]])
        cos_category[start:end] = np.einsum("ij,ij->i", q, category_emb[item_idx[start:end]])
        cos_item_text[start:end] = np.einsum("ij,ij->i", q, item_text_emb[item_idx[start:end]])
        print(f"  Cosine: {end:,}/{len(pairs):,}")

    return pd.DataFrame({
        "term_id": pairs["term_id"].to_numpy(),
        "item_id": pairs["item_id"].to_numpy(),
        "emb_cos_title": cos_title,
        "emb_cos_category": cos_category,
        "emb_cos_item_text": cos_item_text,
    })


def main() -> None:
    args = parse_args()
    train_pairs_path = args.train_pairs or (args.artifacts_dir / "train_testlike.parquet")
    test_pairs_path = args.test_pairs or (args.data_dir / "submission_pairs.csv")
    output_dir = args.output_dir or (args.artifacts_dir / "embedding_features")

    if not train_pairs_path.exists():
        raise FileNotFoundError(
            f"Train pair dosyası yok: {train_pairs_path}\n"
            "Önce build_testlike_training.py çalıştırın veya --train-pairs ile yol verin."
        )
    if not test_pairs_path.exists():
        raise FileNotFoundError(f"Test pair dosyası yok: {test_pairs_path}")

    print("=== Katalog ===")
    items = pd.read_csv(args.data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype="string").drop_duplicates("item_id")
    terms = pd.read_csv(args.data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string").drop_duplicates("term_id")
    items["category"] = items["category"].fillna("").astype("string").map(normalize_category_path)
    for column in ["title", "brand", "gender", "age_group", "attributes"]:
        items[column] = items[column].fillna("").astype("string").map(normalize_text)
    items["item_text"] = make_item_text(items)
    items["item_id"] = items["item_id"].astype(str)
    terms["term_id"] = terms["term_id"].astype(str)
    terms["query"] = terms["query"].fillna("").astype("string").map(normalize_text)
    print(f"items: {len(items):,} | terms: {len(terms):,}")

    print(f"\n=== Model yükleniyor: {args.model_name} (yalnızca inference, fine-tune yok) ===")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, pairs_path in (("train", train_pairs_path), ("test", test_pairs_path)):
        print(f"\n=== {split} pair'leri: {pairs_path} ===")
        pairs = load_pairs(pairs_path)
        print(f"Benzersiz çift: {len(pairs):,}")
        features = build_features(pairs, terms, items, model, args)
        out_path = output_dir / f"{split}_embedding_features.parquet"
        features.to_parquet(out_path, index=False)
        print(f"Çıktı: {out_path} ({len(features):,} satır)")

    print("\nTamamlandı.")


if __name__ == "__main__":
    main()
