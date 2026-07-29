#!/usr/bin/env python3
"""Test-benzeri eğitim verisi: train sorguları için retrieval'dan top-N aday + bilinen pozitifler.

submission_pairs.csv incelendiğinde sorgu başına medyan tam olarak top-100 aday olduğu
(uzun kuyruktaki 101-116 gibi sayılar da top-100 dışında kalan bilinen pozitiflerin zorla
eklendiğini düşündürüyor) görülüyor. Bu script aynı süreci train sorgularına uygulayarak
train/test aday dağılımını hizalar: mevcut train_with_negatives.parquet'teki rastgele/kolay
negatifler yerine, gerçek retrieval'dan gelen -ve test'te karşılaşılacak- adaylar kullanılır.
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

from retrieval_utils import RetrievalConfig, SparseCatalogRetriever  # noqa: E402
from text_features import make_item_text, normalize_category_path, normalize_text  # noqa: E402

ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
ROOT = SRC_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--top-n", type=int, default=100, help="Sorgu başına retrieval'dan alınacak aday sayısı")
    parser.add_argument(
        "--retrieval-top-n", type=int, default=None,
        help="Pozitif recall için arama havuzu; varsayılan max(top-n, 300). Çıktı yine top-n adaydır.",
    )
    parser.add_argument("--sample-queries", type=int, default=None, help="Smoke test için sorgu (term) sayısını sınırla")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    items = pd.read_csv(data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype="string").drop_duplicates("item_id")
    terms = pd.read_csv(data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string").drop_duplicates("term_id")
    positives = pd.read_csv(
        data_dir / "training_pairs.csv",
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"},
    )
    items["category"] = items["category"].fillna("").astype("string").map(normalize_category_path)
    for column in ["title", "brand", "gender", "age_group", "attributes"]:
        items[column] = items[column].fillna("").astype("string").map(normalize_text)
    items["item_text"] = make_item_text(items)
    terms["query"] = terms["query"].fillna("").astype("string").map(normalize_text)
    return items.reset_index(drop=True), terms.reset_index(drop=True), positives.reset_index(drop=True)


def merged_search(title_retriever: SparseCatalogRetriever, text_retriever: SparseCatalogRetriever, terms: pd.DataFrame, top_n: int):
    """Başlık-only ve tam item_text retrieval'larını birleştirir.

    Tam item_text (title+category+brand+attributes) tek başına kullanıldığında uzun
    attribute metni cosine normalizasyonunu seyreltip temiz başlık eşleşmelerini
    top-N dışına itebiliyor. Başlık-only kanalı bu sinyali saf tutar; item_text
    kanalı ise yalnızca kategori/marka/öznitelik bağlamından yakalanabilecek
    eşleşmeleri sağlar. İkisinin en iyisini (max skor) alıp yeniden sıralıyoruz.
    """
    for title_batch, text_batch in zip(title_retriever.iter_search(terms), text_retriever.iter_search(terms)):
        title_batch = title_batch[["term_id", "query", "item_id", "retrieval_score"]].rename(
            columns={"retrieval_score": "title_score"}
        )
        text_batch = text_batch[["term_id", "query", "item_id", "retrieval_score"]].rename(
            columns={"retrieval_score": "text_score"}
        )
        merged = title_batch.merge(text_batch, on=["term_id", "query", "item_id"], how="outer")
        merged["title_score"] = merged["title_score"].fillna(0.0)
        merged["text_score"] = merged["text_score"].fillna(0.0)
        merged["retrieval_score"] = merged[["title_score", "text_score"]].max(axis=1)
        merged["retrieval_source"] = np.where(merged["title_score"] >= merged["text_score"], "title", "item_text")
        merged = merged.sort_values(["term_id", "retrieval_score"], ascending=[True, False])
        merged["retrieval_rank"] = merged.groupby("term_id").cumcount() + 1
        merged = merged[merged["retrieval_rank"] <= top_n]
        yield merged[["term_id", "query", "item_id", "retrieval_rank", "retrieval_score", "retrieval_source"]]


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        suffix = f"_sample{args.sample_queries}" if args.sample_queries else ""
        output = args.artifacts_dir / f"train_testlike{suffix}.parquet"

    print("=== Katalog ve eşleşmeler ===")
    items, terms, positives = load_data(args.data_dir)
    print(f"items: {len(items):,} | terms: {len(terms):,} | pozitif çift: {len(positives):,}")

    train_terms = terms[terms["term_id"].isin(positives["term_id"])].copy()
    if args.sample_queries and len(train_terms) > args.sample_queries:
        train_terms = train_terms.sample(args.sample_queries, random_state=args.seed).sort_values("term_id")
        positives = positives[positives["term_id"].isin(train_terms["term_id"])]
        print(f"Smoke test: {len(train_terms):,} sorguya örneklendi")

    known = positives.groupby("term_id")["item_id"].agg(lambda s: set(s.astype(str))).to_dict()

    retrieval_top_n = max(args.top_n, args.retrieval_top_n or 300)
    print(f"\n=== Retrieval havuzu (top-{retrieval_top_n}), çıktı adayları (top-{args.top_n}), tam katalog: {len(items):,} ürün ===")
    config = RetrievalConfig(top_k=retrieval_top_n, batch_size=args.batch_size)
    title_retriever = SparseCatalogRetriever(config).fit(items["item_id"], items["title"])
    text_retriever = SparseCatalogRetriever(config).fit(items["item_id"], items["item_text"])

    frames = []
    total_positive_pairs = 0
    found_positive_pairs = 0
    processed_terms = 0
    for batch in merged_search(title_retriever, text_retriever, train_terms, retrieval_top_n):
        for term_id, group in batch.groupby("term_id", sort=False):
            term_known = known[str(term_id)]
            retrieved_ids = set(group["item_id"].astype(str))
            total_positive_pairs += len(term_known)
            found_positive_pairs += len(retrieved_ids & term_known)

            group = group.copy()
            group["label"] = group["item_id"].astype(str).isin(term_known).astype(np.int8)
            # Retrieval'ın ilk sıraları gerçek pozitifleri kaçırabileceği için
            # bilinmeyen adayları kesin negatif değil, confidence-weighted negatif
            # olarak taşırız. Ranker yalnızca açıkça bilinen pozitifleri weight=1
            # ile görür; top-rank false-negative riski düşük weight ile kalır.
            group["label_confidence"] = np.where(
                group["label"].eq(1), "known_positive",
                np.where(group["retrieval_rank"].le(10), "ambiguous",
                         np.where(group["retrieval_rank"].le(50), "hard", "medium")),
            )
            group["sample_weight"] = np.where(
                group["label"].eq(1), 1.0,
                np.where(group["retrieval_rank"].le(10), 0.15,
                         np.where(group["retrieval_rank"].le(50), 0.50, 0.85)),
            ).astype(np.float32)

            missing = term_known - retrieved_ids
            if missing:
                extra = pd.DataFrame({
                    "term_id": str(term_id),
                    "query": group["query"].iloc[0],
                    "item_id": sorted(missing),
                    "retrieval_rank": 0,
                    "retrieval_score": 0.0,
                    "retrieval_source": "forced_positive",
                    "label": np.int8(1),
                    "label_confidence": "known_positive",
                    "sample_weight": np.float32(1.0),
                })
                group = pd.concat([group, extra], ignore_index=True)
            # Retrieval havuzu geniş, fakat ranker'ın train/test grup boyutu
            # testteki yaklaşık 100 adaya yakın kalsın: bütün bilinen
            # pozitifleri koru, kalan slotları en yüksek retrieval negatifleriyle doldur.
            known_rows = group[group["item_id"].astype(str).isin(term_known)]
            unknown_rows = group[~group["item_id"].astype(str).isin(term_known)]
            keep_unknown = max(0, args.top_n - len(known_rows))
            group = pd.concat([
                unknown_rows.sort_values("retrieval_rank").head(keep_unknown),
                known_rows,
            ], ignore_index=True)
            frames.append(group)
        processed_terms += batch["term_id"].nunique()
        print(f"  Retrieval: {processed_terms:,}/{len(train_terms):,} sorgu")

    candidates = pd.concat(frames, ignore_index=True)
    candidates["term_id"] = candidates["term_id"].astype(str)
    candidates["item_id"] = candidates["item_id"].astype(str)
    candidates = candidates.drop_duplicates(["term_id", "item_id"]).reset_index(drop=True)

    print("\n=== Zenginleştirme ===")
    train = candidates.merge(items[ITEM_COLUMNS], on="item_id", how="left", validate="many_to_one")
    if train[ITEM_COLUMNS[1:]].isna().any().any():
        raise RuntimeError("Bazı item_id'ler için katalog eşleşmesi bulunamadı")

    recall = found_positive_pairs / total_positive_pairs if total_positive_pairs else float("nan")
    print(f"\nToplam bilinen pozitif: {total_positive_pairs:,}")
    print(f"Retrieval'da doğal olarak bulunan: {found_positive_pairs:,}")
    print(f"Top-{args.top_n} pozitif recall: {recall:.4f} ({recall * 100:.2f}%)")
    if recall < 0.85:
        print(f"UYARI: recall %85 altında. --retrieval-top-n değerini artırmayı (ör. 500) düşünün (şu an {retrieval_top_n}).")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output, index=False)
    print(
        f"\nSatır: {len(train):,} | Pozitif: {int(train['label'].sum()):,} "
        f"({train['label'].mean() * 100:.2f}%) | Term: {train['term_id'].nunique():,}"
    )
    print(f"Çıktı: {output}")


if __name__ == "__main__":
    main()
