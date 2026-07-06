#!/usr/bin/env python3
"""Mine query-driven negatives from the full Trendyol catalog."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import experiment_dir, write_config
from retrieval_utils import RetrievalConfig, SparseCatalogRetriever
from text_features import (
    extract_age_terms, extract_colors, extract_gender_terms, extract_numbers,
    make_item_text, normalize_category_path, normalize_text, select_attributes,
    is_generic_query,
)

ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]


def contradiction(query: str, item_text: str) -> bool:
    extractors = (extract_colors, extract_gender_terms, extract_age_terms, extract_numbers)
    return any((q := fn(query)) and (v := fn(item_text)) and not q & v for fn in extractors)


def load_data(data_dir: Path, max_terms: int | None, seed: int):
    items = pd.read_csv(data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype="string").drop_duplicates("item_id")
    terms = pd.read_csv(data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string").drop_duplicates("term_id")
    positives = pd.read_csv(data_dir / "training_pairs.csv", dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"})
    train_terms = terms[terms["term_id"].isin(positives["term_id"])].copy()
    if max_terms and len(train_terms) > max_terms:
        train_terms = train_terms.sample(max_terms, random_state=seed).sort_values("term_id")
        positives = positives[positives["term_id"].isin(train_terms["term_id"])]
    items["category"] = items["category"].map(normalize_category_path)
    for col in ["title", "brand", "gender", "age_group", "attributes"]:
        items[col] = items[col].fillna("").map(normalize_text)
    terms["query"] = terms["query"].fillna("").map(normalize_text)
    train_terms["query"] = train_terms["query"].fillna("").map(normalize_text)
    items["selected_attributes"] = [select_attributes(a) for a in items["attributes"]]
    items["item_text"] = (
        "[TITLE] " + items["title"] + " [CATEGORY] " + items["category"].str.replace("/", " ", regex=False)
        + " [BRAND] " + items["brand"] + " [ATTR] " + items["selected_attributes"]
    )
    return items.reset_index(drop=True), terms, positives.reset_index(drop=True), train_terms.reset_index(drop=True)


def target_negative_count(
    positive_count: int,
    negatives_per_positive: float = 2.0,
    min_negatives: int = 12,
    max_negatives: int = 40,
) -> int:
    return int(np.clip(math.ceil(positive_count * negatives_per_positive), min_negatives, max_negatives))


def choose_negatives(
    candidates: pd.DataFrame,
    known: set[str],
    positive_count: int = 1,
    negatives_per_positive: float = 2.0,
    min_negatives: int = 12,
    max_negatives: int = 40,
    easy_fraction: float = 0.10,
) -> tuple[pd.DataFrame, int]:
    """Choose rank-bucket negatives and return them with the required easy count."""
    available = candidates[~candidates["item_id"].isin(known)].copy()
    target = target_negative_count(
        positive_count, negatives_per_positive, min_negatives, max_negatives
    )
    easy_count = min(4, max(1, int(round(target * easy_fraction))))
    retrieval_target = max(0, target - easy_count)
    generic = bool(len(candidates) and is_generic_query(candidates["query"].iloc[0]))
    ambiguous_count = 0 if generic else min(4, max(1, int(round(target * 0.10))))
    hard_count = int(round(target * 0.40)) + (min(4, max(1, int(round(target * 0.10)))) if generic else 0)
    medium_count = max(0, retrieval_target - ambiguous_count - hard_count)
    buckets = [
        (1, 10, ambiguous_count, "ambiguous", 0.2),
        (11, 75, hard_count, "hard", 0.6),
        (76, 250, medium_count, "medium", 0.8),
    ]
    selected = []
    for lower, upper, count, confidence, weight in buckets:
        part = available[available["retrieval_rank"].between(lower, upper)].head(count).copy()
        part["negative_confidence"] = confidence
        part["sample_weight"] = weight
        selected.append(part)
    result = pd.concat(selected, ignore_index=True) if selected else available.iloc[0:0]
    if len(result) < retrieval_target:
        remaining = available[
            available["retrieval_rank"].ge(11)
            & ~available["item_id"].isin(result["item_id"])
        ].head(retrieval_target - len(result)).copy()
        remaining["negative_confidence"] = "medium_fill"
        remaining["sample_weight"] = 0.8
        result = pd.concat([result, remaining], ignore_index=True)

    positive_scores = candidates.loc[candidates["item_id"].isin(known), "retrieval_score"]
    if len(positive_scores):
        near_positive_cutoff = float(positive_scores.median())
        near_positive = result["retrieval_score"].ge(near_positive_cutoff)
        result.loc[near_positive, "negative_confidence"] = "near_positive"
        result.loc[near_positive, "sample_weight"] = 0.1
    return result.head(retrieval_target), easy_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--experiment-id", default="query_mining_v2")
    parser.add_argument("--max-terms", type=int)
    parser.add_argument("--catalog-size", type=int)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negatives-per-positive", type=float, default=2.0)
    parser.add_argument("--min-negatives-per-query", type=int, default=12)
    parser.add_argument("--max-negatives-per-query", type=int, default=40)
    parser.add_argument("--easy-fraction", type=float, default=0.10)
    parser.add_argument("--target-effective-negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = experiment_dir(args.artifacts_dir, args.experiment_id)
    write_config(out / "config.json", vars(args))

    items, terms, positives, train_terms = load_data(args.data_dir, args.max_terms, args.seed)
    if args.catalog_size and len(items) > args.catalog_size:
        required = items[items["item_id"].isin(positives["item_id"])]
        remaining = items[~items["item_id"].isin(required["item_id"])]
        sampled = remaining.sample(max(0, args.catalog_size - len(required)), random_state=args.seed)
        items = pd.concat([required, sampled]).drop_duplicates("item_id").reset_index(drop=True)

    config = RetrievalConfig(top_k=args.top_k, batch_size=args.batch_size)
    retriever = SparseCatalogRetriever(config).fit(items["item_id"], items["item_text"])
    known = positives.groupby("term_id")["item_id"].agg(lambda s: set(s.astype(str))).to_dict()
    rng = np.random.default_rng(args.seed)
    negatives = []
    retrieval_frames = []
    for batch in retriever.iter_search(train_terms):
        retrieval_frames.append(batch.drop(columns="item_position"))
        for term_id, group in batch.groupby("term_id", sort=False):
            term_known = known.get(str(term_id), set())
            chosen, easy_count = choose_negatives(
                group, term_known, positive_count=len(term_known),
                negatives_per_positive=args.negatives_per_positive,
                min_negatives=args.min_negatives_per_query,
                max_negatives=args.max_negatives_per_query,
                easy_fraction=args.easy_fraction,
            )
            used = set(chosen["item_id"].astype(str)) | term_known
            easy_records = []
            attempts = 0
            while len(easy_records) < easy_count and attempts < easy_count * 100:
                attempts += 1
                easy_id = str(items.iloc[int(rng.integers(len(items)))]["item_id"])
                if easy_id in used:
                    continue
                used.add(easy_id)
                easy_records.append({
                    "term_id": term_id, "query": group["query"].iloc[0], "item_id": easy_id,
                    "retrieval_rank": 0, "retrieval_score": 0.0, "retrieval_source": "easy_random",
                    "negative_confidence": "easy", "sample_weight": 1.0,
                })
            if easy_records:
                chosen = pd.concat([
                    chosen.drop(columns="item_position", errors="ignore"),
                    pd.DataFrame(easy_records),
                ], ignore_index=True)
            negatives.append(chosen)

    negative_pairs = pd.concat(negatives, ignore_index=True).drop_duplicates(["term_id", "item_id"])
    negative_pairs["label"] = np.int8(0)
    negative_pairs["negative_type"] = "query_retrieval"
    enriched_neg = negative_pairs.merge(items[ITEM_COLUMNS], on="item_id", how="left", validate="many_to_one")
    item_text = make_item_text(enriched_neg)
    clear = [contradiction(q, t) for q, t in zip(enriched_neg["query"], item_text)]
    enriched_neg.loc[clear, ["negative_confidence", "sample_weight"]] = ["clear_contradiction", 1.0]
    enriched_neg["base_sample_weight"] = enriched_neg["sample_weight"].astype(np.float32)

    positive_weight_total = float(len(positives))
    raw_negative_weight_total = float(enriched_neg["sample_weight"].sum())
    negative_weight_multiplier = (
        args.target_effective_negative_ratio * positive_weight_total
        / max(raw_negative_weight_total, 1e-12)
    )
    enriched_neg["sample_weight"] = (
        enriched_neg["sample_weight"] * negative_weight_multiplier
    ).astype(np.float32)

    positive = positives.merge(terms, on="term_id", how="left").merge(items[ITEM_COLUMNS], on="item_id", how="left")
    positive["negative_type"] = "positive"
    positive["negative_confidence"] = "positive"
    positive["retrieval_source"] = "known_positive"
    positive["retrieval_rank"] = 0
    positive["retrieval_score"] = 1.0
    positive["sample_weight"] = 1.0
    positive["base_sample_weight"] = 1.0
    columns = ["term_id", "item_id", "query", *ITEM_COLUMNS[1:], "label", "negative_type", "negative_confidence", "retrieval_source", "retrieval_rank", "retrieval_score", "base_sample_weight", "sample_weight"]
    train = pd.concat([positive[columns], enriched_neg[columns]], ignore_index=True)
    if train.duplicated(["term_id", "item_id"]).any():
        raise RuntimeError("Duplicate term-item üretildi")
    collision = train[train["label"].eq(0)].merge(positives[["term_id", "item_id"]], on=["term_id", "item_id"])
    if len(collision):
        raise RuntimeError("Bilinen pozitif collision bulundu")
    train.sample(frac=1, random_state=args.seed).to_parquet(out / "train_query_negatives.parquet", index=False)
    pd.concat(retrieval_frames, ignore_index=True).to_parquet(out / "retrieval_candidates.parquet", index=False)
    train.groupby(["label", "negative_confidence"]).agg(rows=("item_id", "size"), mean_weight=("sample_weight", "mean")).reset_index().to_csv(out / "mining_stats.csv", index=False)
    summary = {
        "rows": int(len(train)),
        "positive_rows": int(train["label"].sum()),
        "negative_rows": int(train["label"].eq(0).sum()),
        "positive_weight_total": float(train.loc[train["label"].eq(1), "sample_weight"].sum()),
        "negative_weight_total": float(train.loc[train["label"].eq(0), "sample_weight"].sum()),
        "negative_weight_multiplier": float(negative_weight_multiplier),
        "duplicate_pairs": int(train.duplicated(["term_id", "item_id"]).sum()),
    }
    (out / "weight_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
