"""Sparse full-catalog retrieval and positive-rank evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # sklearn brute force remains a dependency-free fallback.
    sp_matmul_topn = None


@dataclass(frozen=True)
class RetrievalConfig:
    word_max_features: int = 150_000
    char_max_features: int = 100_000
    top_k: int = 300
    batch_size: int = 128


class SparseCatalogRetriever:
    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self.word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            max_features=config.word_max_features, sublinear_tf=True, dtype=np.float32,
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=config.char_max_features, sublinear_tf=True, dtype=np.float32,
        )

    def fit(self, item_ids: pd.Series, item_texts: pd.Series) -> "SparseCatalogRetriever":
        self.item_ids = item_ids.astype(str).to_numpy()
        texts = item_texts.fillna("").astype(str).tolist()
        self.word_matrix = self.word.fit_transform(texts)
        self.char_matrix = self.char.fit_transform(texts)
        n_neighbors = min(self.config.top_k, len(texts))
        self.word_nn = self.char_nn = None
        if sp_matmul_topn is None:
            self.word_nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1).fit(self.word_matrix)
            self.char_nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1).fit(self.char_matrix)
        return self

    @staticmethod
    def _topk(query_matrix, item_matrix, fallback, top_k):
        if sp_matmul_topn is not None:
            similarities = sp_matmul_topn(query_matrix, item_matrix.T, top_n=top_k, threshold=0.0, sort=True)
            rows = []
            for row in range(similarities.shape[0]):
                start, end = similarities.indptr[row], similarities.indptr[row + 1]
                rows.append((similarities.indices[start:end], similarities.data[start:end]))
            return rows
        distances, indices = fallback.kneighbors(query_matrix, n_neighbors=top_k)
        return [(idx, 1.0 - dist) for idx, dist in zip(indices, distances)]

    def iter_search(self, terms: pd.DataFrame):
        """Yield one merged word/char top-k frame per query batch."""
        top_k = min(self.config.top_k, len(self.item_ids))
        for start in range(0, len(terms), self.config.batch_size):
            batch = terms.iloc[start:start + self.config.batch_size]
            word_q = self.word.transform(batch["query"].fillna("").astype(str))
            char_q = self.char.transform(batch["query"].fillna("").astype(str))
            word_results = self._topk(word_q, self.word_matrix, self.word_nn, top_k)
            char_results = self._topk(char_q, self.char_matrix, self.char_nn, top_k)
            records = []
            for row_no, term in enumerate(batch.itertuples(index=False)):
                merged: dict[int, tuple[float, str]] = {}
                for pos, score in zip(*word_results[row_no]):
                    merged[int(pos)] = (float(score), "word")
                for pos, score in zip(*char_results[row_no]):
                    score = float(score)
                    old = merged.get(int(pos))
                    merged[int(pos)] = (max(score, old[0]) if old else score, "word+char" if old else "char")
                ranked = sorted(merged.items(), key=lambda item: (-item[1][0], self.item_ids[item[0]]))[:top_k]
                for rank, (pos, (score, source)) in enumerate(ranked, start=1):
                    records.append((str(term.term_id), str(term.query), self.item_ids[pos], rank, score, source, pos))
            yield pd.DataFrame(records, columns=[
                "term_id", "query", "item_id", "retrieval_rank", "retrieval_score", "retrieval_source", "item_position"
            ])


def retrieval_metrics(candidates: pd.DataFrame, positives: pd.DataFrame) -> pd.DataFrame:
    positive_pairs = positives[["term_id", "item_id"]].drop_duplicates().astype(str)
    ranked = candidates.merge(positive_pairs.assign(is_positive=1), on=["term_id", "item_id"], how="left")
    hits = ranked[ranked["is_positive"].eq(1)]
    total_by_term = positive_pairs.groupby("term_id").size()
    rows = {}
    for k in (10, 25, 50):
        found = hits[hits["retrieval_rank"].le(k)].groupby("term_id")["item_id"].nunique()
        rows[f"recall@{k}"] = float((found.reindex(total_by_term.index, fill_value=0) / total_by_term).mean())
    best_rank = hits.groupby("term_id")["retrieval_rank"].min()
    rows["mrr"] = float((1.0 / best_rank).reindex(total_by_term.index, fill_value=0).mean())
    rows["mean_positive_rank"] = float(hits["retrieval_rank"].mean()) if len(hits) else float("nan")
    rows["query_coverage"] = float(best_rank.index.nunique() / max(1, total_by_term.index.nunique()))
    return pd.DataFrame([rows])
