#!/usr/bin/env python3
"""Build the same title+item-text retrieval features for test candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_testlike_training import ITEM_COLUMNS, load_data, merged_search
from retrieval_utils import RetrievalConfig, SparseCatalogRetriever
from text_features import make_item_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/test_retrieval.parquet"))
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    items, terms, positives = load_data(args.data_dir)
    submission = pd.read_csv(args.data_dir / "submission_pairs.csv", usecols=["term_id", "item_id"], dtype="string")
    test_terms = terms[terms["term_id"].isin(submission["term_id"].unique())].copy()
    items["item_text"] = make_item_text(items)
    config = RetrievalConfig(top_k=args.top_n, batch_size=args.batch_size)
    title_retriever = SparseCatalogRetriever(config).fit(items["item_id"], items["title"])
    text_retriever = SparseCatalogRetriever(config).fit(items["item_id"], items["item_text"])
    frames = []
    for batch in merged_search(title_retriever, text_retriever, test_terms, args.top_n):
        frames.append(batch)
        print(f"retrieval: {sum(len(frame) for frame in frames):,} candidate rows")
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["term_id", "item_id"])
    result = result.merge(submission.drop_duplicates(), on=["term_id", "item_id"], how="inner")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"{args.output}: {len(result):,} rows, {result.term_id.nunique():,} terms")


if __name__ == "__main__":
    main()
