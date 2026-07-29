#!/usr/bin/env python3
"""Build reusable Trendyol-domain semantic and pair-context features.

The expensive operation is encoding each unique query/item once.  Candidate
features are then produced with vector lookups, so the same cache can score
train-like pairs, submission pairs, and engine-replica decoys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODEL = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).translate(TR_LOWER).lower()
    text = re.sub(r"[^0-9a-zçğıöşü]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(value: object) -> frozenset[str]:
    return frozenset(normalize_text(value).split())


def make_short_item_text(items: pd.DataFrame, attribute_chars: int = 420) -> pd.Series:
    """Keep the high-value fields early so 384-token truncation is predictable."""

    category = items["category"].fillna("").str.replace("/", " > ", regex=False)
    attributes = items["attributes"].fillna("").str.slice(0, attribute_chars)
    return (
        "başlık: " + items["title"].fillna("")
        + " | kategori: " + category
        + " | marka: " + items["brand"].fillna("")
        + " | cinsiyet: " + items["gender"].fillna("")
        + " | yaş: " + items["age_group"].fillna("")
        + " | özellikler: " + attributes
    ).astype("string")


def ordered_hash(values: pd.Series) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def encode_catalog(
    texts: pd.Series,
    *,
    model_name: str,
    batch_size: int,
    dimension: int | None,
    device: str,
    trust_remote_code: bool,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, object] = {
        "device": device,
        "trust_remote_code": trust_remote_code,
    }
    if dimension:
        kwargs["truncate_dim"] = dimension
    model = SentenceTransformer(model_name, **kwargs)
    embedding = model.encode(
        texts.fillna("").astype(str).tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(embedding, dtype=np.float16)


def build_embedding_cache(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    terms = pd.read_csv(args.data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string")
    items = pd.read_csv(args.data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype="string")
    terms = terms.drop_duplicates("term_id").sort_values("term_id").reset_index(drop=True)
    items = items.drop_duplicates("item_id").sort_values("item_id").reset_index(drop=True)
    if args.max_queries:
        terms = terms.head(args.max_queries).copy()
    if args.max_items:
        items = items.head(args.max_items).copy()
    terms["embedding_index"] = np.arange(len(terms), dtype=np.int32)
    items["embedding_index"] = np.arange(len(items), dtype=np.int32)
    item_text = make_short_item_text(items, args.attribute_chars)
    device = choose_device(args.device)
    print(f"device={device} model={args.model_name} queries={len(terms):,} items={len(items):,}")

    query_path = args.output_dir / "query_embeddings.f16.npy"
    item_path = args.output_dir / "item_embeddings.f16.npy"
    if not (args.reuse and query_path.exists()):
        query_embedding = encode_catalog(
            terms["query"], model_name=args.model_name, batch_size=args.batch_size,
            dimension=args.dimension, device=device, trust_remote_code=args.trust_remote_code,
        )
        np.save(query_path, query_embedding)
    if not (args.reuse and item_path.exists()):
        item_embedding = encode_catalog(
            item_text, model_name=args.model_name, batch_size=args.batch_size,
            dimension=args.dimension, device=device, trust_remote_code=args.trust_remote_code,
        )
        np.save(item_path, item_embedding)

    terms.to_parquet(args.output_dir / "query_catalog.parquet", index=False)
    items.assign(short_item_text=item_text).to_parquet(args.output_dir / "item_catalog.parquet", index=False)
    q_shape = list(np.load(query_path, mmap_mode="r").shape)
    i_shape = list(np.load(item_path, mmap_mode="r").shape)
    metadata = {
        "model_name": args.model_name,
        "dimension": q_shape[1],
        "device_used": device,
        "query_shape": q_shape,
        "item_shape": i_shape,
        "query_id_hash": ordered_hash(terms["term_id"]),
        "item_id_hash": ordered_hash(items["item_id"]),
        "attribute_chars": args.attribute_chars,
    }
    (args.output_dir / "cache_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def pair_text_features(
    query_indices: np.ndarray,
    item_indices: np.ndarray,
    query_text: list[str],
    title_text: list[str],
    category_text: list[str],
    brand_text: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cheap lexical channels for observed and decoy pairs."""

    q_tokens = [token_set(value) for value in query_text]
    title_tokens = [token_set(value) for value in title_text]
    category_tokens = [token_set(value.replace("/", " ")) for value in category_text]
    normalized_query = [normalize_text(value) for value in query_text]
    normalized_title = [normalize_text(value) for value in title_text]
    normalized_brand = [normalize_text(value) for value in brand_text]
    recall = np.zeros(len(query_indices), dtype=np.float32)
    jaccard = np.zeros(len(query_indices), dtype=np.float32)
    query_in_title = np.zeros(len(query_indices), dtype=np.float32)
    taxonomy_overlap = np.zeros(len(query_indices), dtype=np.float32)
    for row, (q_idx, i_idx) in enumerate(zip(query_indices, item_indices)):
        query = q_tokens[int(q_idx)]
        title = title_tokens[int(i_idx)]
        if query:
            intersection = len(query & title)
            recall[row] = intersection / len(query)
            union = query | title
            jaccard[row] = intersection / len(union) if union else 0.0
            taxonomy = category_tokens[int(i_idx)]
            brand = normalized_brand[int(i_idx)]
            taxonomy_overlap[row] = float(bool(query & taxonomy) or bool(brand and brand in query))
        raw_query = normalized_query[int(q_idx)]
        query_in_title[row] = float(bool(raw_query and raw_query in normalized_title[int(i_idx)]))
    return recall, jaccard, query_in_title, taxonomy_overlap


def score_pairs(args: argparse.Namespace) -> None:
    query_catalog = pd.read_parquet(args.output_dir / "query_catalog.parquet")
    item_catalog = pd.read_parquet(args.output_dir / "item_catalog.parquet")
    query_embedding = np.load(args.output_dir / "query_embeddings.f16.npy", mmap_mode="r")
    item_embedding = np.load(args.output_dir / "item_embeddings.f16.npy", mmap_mode="r")
    q_map = query_catalog.set_index("term_id")["embedding_index"]
    i_map = item_catalog.set_index("item_id")["embedding_index"]
    if args.pairs.suffix == ".parquet":
        pairs = pd.read_parquet(args.pairs)
    else:
        pairs = pd.read_csv(args.pairs, dtype="string")
    required = {"term_id", "item_id"}
    if missing := required - set(pairs.columns):
        raise ValueError(f"pair columns missing: {sorted(missing)}")
    pairs["term_id"] = pairs["term_id"].astype(str)
    pairs["item_id"] = pairs["item_id"].astype(str)
    query_indices = pairs["term_id"].map(q_map).to_numpy()
    item_indices = pairs["item_id"].map(i_map).to_numpy()
    valid = ~pd.isna(query_indices) & ~pd.isna(item_indices)
    if not valid.all():
        raise ValueError(f"embedding cache misses {int((~valid).sum()):,} pairs; rebuild full cache")
    query_indices = query_indices.astype(np.int32)
    item_indices = item_indices.astype(np.int32)
    dense = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), args.chunk_size):
        end = min(start + args.chunk_size, len(pairs))
        q = np.asarray(query_embedding[query_indices[start:end]], dtype=np.float32)
        i = np.asarray(item_embedding[item_indices[start:end]], dtype=np.float32)
        dense[start:end] = np.einsum("ij,ij->i", q, i)
        print(f"dense pairs {end:,}/{len(pairs):,}")

    lexical = pair_text_features(
        query_indices, item_indices,
        query_catalog["query"].fillna("").astype(str).tolist(),
        item_catalog["title"].fillna("").astype(str).tolist(),
        item_catalog["category"].fillna("").astype(str).tolist(),
        item_catalog["brand"].fillna("").astype(str).tolist(),
    )
    result_columns = [column for column in ("id", "term_id", "item_id", "label") if column in pairs]
    result = pairs[result_columns].copy()
    result["query_embedding_index"] = query_indices
    result["item_embedding_index"] = item_indices
    result["ty_cosine"] = dense
    result["token_recall"] = lexical[0]
    result["token_jaccard"] = lexical[1]
    result["query_in_title"] = lexical[2]
    result["taxonomy_overlap"] = lexical[3]
    query_lengths = query_catalog["query"].fillna("").str.len().to_numpy(dtype=np.float32)
    title_lengths = item_catalog["title"].fillna("").str.len().to_numpy(dtype=np.float32)
    result["query_char_count"] = query_lengths[query_indices]
    result["title_char_count"] = title_lengths[item_indices]
    group = result.groupby("term_id", sort=False)
    result["candidate_count"] = group["item_id"].transform("size").astype(np.int32)
    result["candidate_excess"] = (result["candidate_count"] - 100).clip(lower=0).astype(np.int32)
    result["ty_rank"] = group["ty_cosine"].rank(method="first", ascending=False).astype(np.int32)
    denom = (result["candidate_count"] - 1).clip(lower=1)
    result["ty_percentile"] = (1.0 - (result["ty_rank"] - 1) / denom).astype(np.float32)
    result["ty_gap_to_top"] = (group["ty_cosine"].transform("max") - result["ty_cosine"]).astype(np.float32)
    mean = group["ty_cosine"].transform("mean")
    std = group["ty_cosine"].transform("std").fillna(0.0).clip(lower=1e-6)
    result["ty_zscore"] = ((result["ty_cosine"] - mean) / std).astype(np.float32)
    output = args.feature_output or (args.output_dir / f"{args.pairs.stem}_domain_features.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f"wrote {output}: {len(result):,} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cache", "score", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/trendyol_domain"))
    parser.add_argument("--pairs", type=Path, default=Path("data/submission_pairs.csv"))
    parser.add_argument("--feature-output", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--attribute-chars", type=int, default=420)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"cache", "all"}:
        build_embedding_cache(args)
    if args.stage in {"score", "all"}:
        score_pairs(args)


if __name__ == "__main__":
    main()
