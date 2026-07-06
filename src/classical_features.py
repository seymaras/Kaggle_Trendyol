"""Fielded TF-IDF and query-level features for the classical v2 model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from text_features import FEATURE_NAMES, build_lexical_features


FIELD_SPECS = {
    "title": ((1, 2), (3, 5)),
    "category": ((1, 2), (3, 5)),
    "attributes": ((1, 2), (3, 5)),
}
QUERY_FEATURES = [
    "score_rank", "score_percentile", "score_gap_to_top", "score_minus_mean",
    "query_score_std", "candidate_top_category_frequency", "candidate_leaf_category_frequency",
]


class FieldedTfidf:
    def __init__(self, max_features: int = 40_000) -> None:
        self.vectorizers = {}
        for field, (word_range, char_range) in FIELD_SPECS.items():
            self.vectorizers[(field, "word")] = TfidfVectorizer(
                analyzer="word", ngram_range=word_range, min_df=2,
                max_features=max_features, sublinear_tf=True, dtype=np.float32,
            )
            self.vectorizers[(field, "char")] = TfidfVectorizer(
                analyzer="char_wb", ngram_range=char_range, min_df=2,
                max_features=max_features, sublinear_tf=True, dtype=np.float32,
            )

    def fit(self, catalog: pd.DataFrame) -> "FieldedTfidf":
        for (field, _), vectorizer in self.vectorizers.items():
            vectorizer.fit(catalog[field].fillna("").astype(str))
        return self

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
        result = {}
        query = pairs["query"].fillna("").astype(str)
        for (field, kind), vectorizer in self.vectorizers.items():
            q = vectorizer.transform(query)
            item = vectorizer.transform(pairs[field].fillna("").astype(str))
            result[f"{kind}_tfidf_{field}_sim"] = np.asarray(q.multiply(item).sum(axis=1)).ravel()
        return pd.DataFrame(result, index=pairs.index, dtype=np.float32)


def add_query_level_features(frame: pd.DataFrame, score_col: str = "semantic_score") -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby("term_id", sort=False)[score_col]
    out["score_rank"] = grouped.rank(method="first", ascending=False)
    group_size = grouped.transform("size").clip(lower=1)
    out["score_percentile"] = 1.0 - (out["score_rank"] - 1.0) / group_size
    top = grouped.transform("max")
    mean = grouped.transform("mean")
    out["score_gap_to_top"] = top - out[score_col]
    out["score_minus_mean"] = out[score_col] - mean
    out["query_score_std"] = grouped.transform("std").fillna(0.0)
    for field in ("top_category", "leaf_category"):
        counts = out.groupby(["term_id", field], sort=False)["item_id"].transform("size")
        out[f"candidate_{field}_frequency"] = counts / group_size
    return out


def build_classical_features(pairs: pd.DataFrame, fielded: FieldedTfidf) -> pd.DataFrame:
    lexical = build_lexical_features(pairs)
    lexical = lexical.drop(columns=["tfidf_title_sim", "tfidf_item_sim"], errors="ignore")
    sparse = fielded.transform(pairs)
    result = lexical.join(sparse)
    result["semantic_score"] = sparse.mean(axis=1)
    return result.astype(np.float32)


def model_feature_names(frame: pd.DataFrame) -> list[str]:
    excluded = {"id", "term_id", "item_id", "label", "sample_weight", "top_category", "leaf_category"}
    return [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]

