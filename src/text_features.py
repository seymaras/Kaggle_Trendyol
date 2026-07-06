"""Query–item feature helpers shared by training and negative sampling."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

TURKISH_ASCII_TABLE = str.maketrans(
    {"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"}
)

STOPWORDS = {
    "ve", "ile", "icin", "için", "bir", "the", "of", "adet", "urun", "ürün"
}

COLORS = {
    "siyah", "beyaz", "kırmızı", "kirmizi", "mavi", "lacivert", "bej", "krem",
    "haki", "bordo", "gri", "yeşil", "yesil", "pembe", "mor", "turuncu",
    "kahverengi", "ekru", "vizon", "taş", "tas", "füme", "fume", "antrasit",
}

GENERIC_PRODUCT_QUERIES = {
    "elbise", "çanta", "canta", "bot", "ayakkabı", "ayakkabi", "gömlek",
    "gomlek", "mont", "telefon", "kulaklık", "kulaklik", "kitaplık", "kitaplik",
    "tabak", "termos", "tişört", "tisort", "kazak", "pantolon",
}

UNKNOWN_VALUES = {"", "unknown", "bilinmiyor", "none", "nan", "null", "-"}

ATTRIBUTE_KEYS = {
    "renk", "color detail", "materyal", "materyal bileşeni", "beden",
    "boyut/ebat", "kapasite", "hacim", "yaş", "cinsiyet", "model",
    "araç marka ve model", "uyumlu marka", "uyumlu model", "numara",
    "güç (watt)", "paket içeriği", "parça sayısı",
}
UNIT_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:gb|tb|mb|ml|cl|l|lt|kg|gr|g|cm|mm|m|w|watt|v|mah|inç|inc)\b",
    re.IGNORECASE,
)
MODEL_PATTERN = re.compile(r"\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]{2,}\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower().strip()
    text = re.sub(r"[_\W]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def split_category(category: object) -> list[str]:
    """Split a raw category path before punctuation normalization."""
    if pd.isna(category):
        return []
    return [part for raw in str(category).split("/") if (part := normalize_text(raw))]


def normalize_category_path(category: object) -> str:
    return "/".join(split_category(category))


def category_parts(category: object) -> dict[str, str | int]:
    parts = split_category(category)
    return {
        "top_category": parts[0] if parts else "",
        "second_category": parts[1] if len(parts) > 1 else "",
        "parent_category": "/".join(parts[:-1]) if len(parts) > 1 else "",
        "leaf_category": parts[-1] if parts else "",
        "category_depth": len(parts),
    }


def parse_attributes(value: object, selected_keys: Iterable[str] | None = None) -> dict[str, str]:
    """Parse Trendyol's comma-separated ``key: value`` attributes."""
    if pd.isna(value):
        return {}
    allowed = {normalize_text(key) for key in selected_keys} if selected_keys else None
    parsed: dict[str, str] = {}
    for segment in str(value).split(","):
        if ":" not in segment:
            continue
        raw_key, raw_value = segment.split(":", 1)
        key, item_value = normalize_text(raw_key), normalize_text(raw_value)
        if key and item_value and (allowed is None or key in allowed):
            parsed[key] = item_value
    return parsed


def select_attributes(value: object, query: object = "") -> str:
    parsed = parse_attributes(value)
    query_tokens = token_set(query)
    selected = []
    for key, item_value in parsed.items():
        if key in ATTRIBUTE_KEYS or query_tokens & token_set(f"{key} {item_value}"):
            selected.append(f"{key}: {item_value}")
    return " | ".join(selected)


def extract_numbers(text: object) -> set[str]:
    return {match.replace(",", ".") for match in NUMBER_PATTERN.findall(normalize_text(text))}


def extract_units(text: object) -> set[str]:
    return {normalize_text(match).replace(" ", "") for match in UNIT_PATTERN.findall(str(text))}


def extract_model_codes(text: object) -> set[str]:
    return {normalize_text(match) for match in MODEL_PATTERN.findall(normalize_text(text))}


def token_set(value: object) -> set[str]:
    normalized = normalize_text(value)
    folded = normalized.translate(TURKISH_ASCII_TABLE)
    tokens = set(normalized.split()) | set(folded.split())
    return {token for token in tokens if token and token not in STOPWORDS}


def overlap_ratio(left: object, right: object) -> float:
    left_tokens = token_set(left)
    if not left_tokens:
        return 0.0
    return len(left_tokens & token_set(right)) / len(left_tokens)


def unmatched_ratio(query: object, text: object) -> float:
    query_tokens = token_set(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens - token_set(text)) / len(query_tokens)


def extract_colors(text: object) -> set[str]:
    return token_set(text) & COLORS


def extract_gender_terms(text: object) -> set[str]:
    tokens = token_set(text)
    result: set[str] = set()
    if tokens & {"kadın", "kadin", "bayan", "kız", "kiz"}:
        result.add("kadın")
    if "erkek" in tokens:
        result.add("erkek")
    if "unisex" in tokens:
        result.add("unisex")
    return result


def extract_age_terms(text: object) -> set[str]:
    tokens = token_set(text)
    result: set[str] = set()
    if tokens & {"bebek", "yenidoğan", "yenidogan"}:
        result.add("bebek")
    if tokens & {"çocuk", "cocuk", "kız", "kiz", "oğlan", "oglan"}:
        result.add("çocuk")
    if tokens & {"yetişkin", "yetiskin"}:
        return result | {"yetişkin"}
    return result


def is_generic_query(query: object) -> bool:
    tokens = token_set(query)
    return len(tokens) <= 2 and bool(tokens) and tokens.issubset(GENERIC_PRODUCT_QUERIES)


def valid_brand(brand: object) -> bool:
    return normalize_text(brand) not in UNKNOWN_VALUES


def make_item_text(frame: pd.DataFrame) -> pd.Series:
    columns = ["title", "category", "brand", "gender", "age_group", "attributes"]
    result = pd.Series("", index=frame.index, dtype="string")
    for column in columns:
        if column in frame.columns:
            result = result.str.cat(frame[column].fillna("").astype("string"), sep=" ")
    return result.map(normalize_text).astype("string")


def query_token_count(query: object) -> int:
    return len(token_set(query))


def category_depth(category: object) -> int:
    return len(split_category(category))


def jaccard(left: object, right: object) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def set_overlap_size(left: object, right: object) -> int:
    return len(extract_colors(left) & extract_colors(right))


def gender_overlap(query: object, item_gender: object, item_text: object) -> int:
    query_gender = extract_gender_terms(query)
    if not query_gender:
        return 0
    item_gender_terms = extract_gender_terms(item_gender) | extract_gender_terms(item_text)
    return int(bool(query_gender & item_gender_terms))


def age_overlap(query: object, item_age: object, item_text: object) -> int:
    query_age = extract_age_terms(query)
    if not query_age:
        return 0
    item_age_terms = extract_age_terms(item_age) | extract_age_terms(item_text)
    return int(bool(query_age & item_age_terms))


def brand_in_query(query: object, brand: object) -> int:
    if not valid_brand(brand):
        return 0
    brand_norm = normalize_text(brand)
    return int(brand_norm in normalize_text(query) or brand_norm in token_set(query))


def top_category_match(query: object, category: object) -> int:
    parts = split_category(category)
    top = parts[0] if parts else ""
    if not top:
        return 0
    return int(top in token_set(query) or top in normalize_text(query))


def contradiction(query: object, item_text: object, extractor) -> int:
    query_values = extractor(query)
    item_values = extractor(item_text)
    return int(bool(query_values and item_values and not (query_values & item_values)))


FEATURE_NAMES = [
    "overlap_title",
    "overlap_category",
    "overlap_attributes",
    "overlap_item_text",
    "unmatched_title",
    "query_token_count",
    "category_depth",
    "top_category_match",
    "overlap_top_category",
    "overlap_second_category",
    "overlap_leaf_category",
    "brand_in_query",
    "gender_overlap",
    "age_overlap",
    "color_overlap",
    "color_contradiction",
    "gender_contradiction",
    "age_contradiction",
    "number_match",
    "number_contradiction",
    "unit_match",
    "model_code_match",
    "jaccard_title",
    "is_generic_query",
    "tfidf_title_sim",
    "tfidf_item_sim",
]


def build_lexical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Lexical and metadata overlap features (TF-IDF added separately)."""
    item_text = make_item_text(frame)

    def row_features(idx: int) -> dict[str, float]:
        row = frame.iloc[idx]
        text = item_text.iloc[idx]
        category = category_parts(row["category"])
        query_numbers, item_numbers = extract_numbers(row["query"]), extract_numbers(text)
        query_units, item_units = extract_units(row["query"]), extract_units(text)
        query_models, item_models = extract_model_codes(row["query"]), extract_model_codes(text)
        return {
            "overlap_title": overlap_ratio(row["query"], row["title"]),
            "overlap_category": overlap_ratio(row["query"], row["category"]),
            "overlap_attributes": overlap_ratio(row["query"], row["attributes"]),
            "overlap_item_text": overlap_ratio(row["query"], text),
            "unmatched_title": unmatched_ratio(row["query"], row["title"]),
            "query_token_count": float(query_token_count(row["query"])),
            "category_depth": float(category_depth(row["category"])),
            "top_category_match": float(top_category_match(row["query"], row["category"])),
            "overlap_top_category": overlap_ratio(row["query"], category["top_category"]),
            "overlap_second_category": overlap_ratio(row["query"], category["second_category"]),
            "overlap_leaf_category": overlap_ratio(row["query"], category["leaf_category"]),
            "brand_in_query": float(brand_in_query(row["query"], row["brand"])),
            "gender_overlap": float(
                gender_overlap(row["query"], row["gender"], text)
            ),
            "age_overlap": float(age_overlap(row["query"], row["age_group"], text)),
            "color_overlap": float(set_overlap_size(row["query"], text)),
            "color_contradiction": float(contradiction(row["query"], text, extract_colors)),
            "gender_contradiction": float(contradiction(row["query"], text, extract_gender_terms)),
            "age_contradiction": float(contradiction(row["query"], text, extract_age_terms)),
            "number_match": float(bool(query_numbers & item_numbers)),
            "number_contradiction": float(bool(query_numbers and item_numbers and not query_numbers & item_numbers)),
            "unit_match": float(bool(query_units & item_units)),
            "model_code_match": float(bool(query_models & item_models)),
            "jaccard_title": jaccard(row["query"], row["title"]),
            "is_generic_query": float(is_generic_query(row["query"])),
        }

    records = [row_features(i) for i in range(len(frame))]
    return pd.DataFrame(records, index=frame.index)


def sparse_cosine(row_a, row_b) -> float:
    if row_a is None or row_b is None:
        return 0.0
    return float((row_a @ row_b.T)[0, 0])


def attach_tfidf_similarity(
    frame: pd.DataFrame,
    query_vectors: dict[str, np.ndarray],
    title_vectors: dict[str, np.ndarray],
    item_vectors: dict[str, np.ndarray],
) -> pd.DataFrame:
    tfidf_title = []
    tfidf_item = []
    for row in frame.itertuples(index=False):
        qv = query_vectors.get(str(row.term_id))
        tv = title_vectors.get(str(row.item_id))
        iv = item_vectors.get(str(row.item_id))
        tfidf_title.append(sparse_cosine(qv, tv))
        tfidf_item.append(sparse_cosine(qv, iv))
    out = frame.copy()
    out["tfidf_title_sim"] = tfidf_title
    out["tfidf_item_sim"] = tfidf_item
    return out
