# %% [markdown]
# # Trendyol Datathon — Gerçek Katalogdan Negatif Örnek Üretimi
#
# Bu notebook/script hiçbir hayali ürün üretmez. Bütün negatif `item_id` değerleri
# doğrudan `items.csv` kataloğundan seçilir. Önce küçük örnekle deneyin; kontroller
# geçtikten sonra `USE_FULL_DATA = True` yapın.

# %%
# 1. LIBRARY IMPORTLARI
from __future__ import annotations

import gc
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=FutureWarning)


# %% [markdown]
# ## 2. Config
#
# Kaggle'da `DATA_DIR` yarışma dataset klasörünü göstermelidir. Yol yanlışsa
# `AUTO_FIND_FILES=True` gerekli CSV'leri `/kaggle/input` altında arar.
# İlk çalıştırmada küçük sample kullanın.

# %%
DATA_DIR = Path("/kaggle/input/trendyol-e-ticaret-yarismasi-2026")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
AUTO_FIND_FILES = True

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
USE_FULL_DATA = os.getenv("USE_FULL_DATA", "false").lower() == "true"
POS_SAMPLE_SIZE = int(os.getenv("POS_SAMPLE_SIZE", "50000"))
NEGATIVE_RATIO = float(os.getenv("NEGATIVE_RATIO", "3.0"))

# Her stratejinin kaç pozitif satır üzerinde çalışacağı. Toplam yaklaşık 3 negatif/pozitif.
STRATEGY_RATES = {
    "easy_random": 0.55,
    "different_top_category": 0.55,
    "same_top_category_hard": 0.45,
    "same_gender_age_hard": 0.40,
    "brand_hard": 0.25,
    "title_similar_hard": 0.40,
    "attribute_mismatch": 0.40,
}

# TF-IDF retrieval ayarları. Büyük katalogda önce sınırlı gerçek ürün havuzuyla deneyin.
TFIDF_ITEM_POOL_SIZE = int(os.getenv("TFIDF_ITEM_POOL_SIZE", "75000"))
USE_FULL_TFIDF_CATALOG = os.getenv("USE_FULL_TFIDF_CATALOG", "false").lower() == "true"
TFIDF_ANALYZER = os.getenv("TFIDF_ANALYZER", "word")  # word daha az RAM; char_wb daha esnek.
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "50000"))
TFIDF_TOP_K = 30
TFIDF_QUERY_BATCH_SIZE = 128
TFIDF_MIN_SIMILARITY = 0.12
TFIDF_MAX_SAFE_SIMILARITY = 0.94

MAX_SAMPLE_ATTEMPTS = 30
STRONG_MATCH_RATIO = 0.75
ADD_ASCII_TEXT_COLUMNS = False  # True daha çok RAM kullanır.

if Path("/kaggle/working").exists():
    OUTPUT_DIR = Path("/kaggle/working")
elif (Path.cwd() / "artifacts").exists():
    OUTPUT_DIR = Path.cwd() / "artifacts"
elif (Path.cwd().parent / "artifacts").exists():
    OUTPUT_DIR = Path.cwd().parent / "artifacts"
else:
    OUTPUT_DIR = Path.cwd()
OUTPUT_CSV = OUTPUT_DIR / "train_with_negatives.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "train_with_negatives.parquet"
OUTPUT_AUDIT = OUTPUT_DIR / "negative_audit_sample.csv"
OUTPUT_STATS = OUTPUT_DIR / "negative_sampling_stats.csv"


# %% [markdown]
# ## 3. CSV dosyalarını okuma
#
# `locate_csv` Kaggle dataset klasör adı değişse bile dosyaları bulmaya çalışır.

# %%
REQUIRED_FILES = {
    "items": "items.csv",
    "terms": "terms.csv",
    "training": "training_pairs.csv",
}


def locate_csv(filename: str) -> Path:
    direct = DATA_DIR / filename
    if direct.exists():
        return direct

    local_candidates = [
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        Path.cwd().parent / "data" / filename,
    ]
    for local in local_candidates:
        if local.exists():
            return local

    if AUTO_FIND_FILES and KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
        if matches:
            print(f"{filename} bulundu: {matches[0]}")
            return matches[0]

    raise FileNotFoundError(
        f"{filename} bulunamadı. DATA_DIR değerini yarışma dataset klasörüne ayarlayın."
    )


ITEM_COLUMNS = [
    "item_id", "title", "category", "brand", "gender", "age_group", "attributes"
]

items = pd.read_csv(
    locate_csv(REQUIRED_FILES["items"]),
    usecols=ITEM_COLUMNS,
    dtype={column: "string" for column in ITEM_COLUMNS},
)
terms = pd.read_csv(
    locate_csv(REQUIRED_FILES["terms"]),
    usecols=["term_id", "query"],
    dtype={"term_id": "string", "query": "string"},
)
training_all = pd.read_csv(
    locate_csv(REQUIRED_FILES["training"]),
    usecols=["id", "term_id", "item_id", "label"],
    dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"},
)

print(f"items: {len(items):,}")
print(f"terms: {len(terms):,}")
print(f"training positives: {len(training_all):,}")
assert training_all["label"].eq(1).all(), "training_pairs.csv içinde 1 dışında label bulundu."
training_all = training_all.drop_duplicates(["term_id", "item_id"]).reset_index(drop=True)


# %% [markdown]
# ## 4. Veri tiplerini ve örneklemeyi optimize etme
#
# Bilinen pozitif kontrolü tüm eğitim dosyasına dayanır. Modelleme/negatif üretme
# içinse ilk denemede yalnızca `POS_SAMPLE_SIZE` kadar pozitif kullanılır.

# %%
items = items.drop_duplicates("item_id").reset_index(drop=True)
terms = terms.drop_duplicates("term_id").reset_index(drop=True)

if USE_FULL_DATA or POS_SAMPLE_SIZE is None or len(training_all) <= POS_SAMPLE_SIZE:
    positive_pairs = training_all.copy()
else:
    positive_pairs = training_all.sample(
        n=POS_SAMPLE_SIZE, random_state=RANDOM_STATE
    ).reset_index(drop=True)

sample_term_ids = positive_pairs["term_id"].drop_duplicates()

# Yalnızca örneklemdeki term'lerin tüm bilinen pozitiflerini bellekte set olarak tut.
known_subset = training_all[
    training_all["term_id"].isin(sample_term_ids)
][["term_id", "item_id"]].drop_duplicates()

known_positive_by_term: dict[str, set[str]] = {
    str(term_id): set(group["item_id"].astype(str))
    for term_id, group in known_subset.groupby("term_id", sort=False)
}

del known_subset
gc.collect()

print(f"Negatif üretilecek pozitif sayısı: {len(positive_pairs):,}")
print(f"Örneklemdeki benzersiz term sayısı: {positive_pairs['term_id'].nunique():,}")


# %% [markdown]
# ## 5. Metin temizleme fonksiyonları
#
# Türkçe karakterler korunur. Ayrıca karşılaştırmalarda `kadın/kadin`,
# `çocuk/cocuk` gibi yazımların eşleşmesi için ASCII-fold tokenları da kullanılır.

# %%
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


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower().strip()
    text = re.sub(r"[_\W]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype("string")
        .str.lower()
        .str.replace(r"[_\W]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def ascii_fold(value: object) -> str:
    return normalize_text(value).translate(TURKISH_ASCII_TABLE)


def token_set(value: object) -> set[str]:
    normalized = normalize_text(value)
    folded = normalized.translate(TURKISH_ASCII_TABLE)
    tokens = set(normalized.split()) | set(folded.split())
    return {token for token in tokens if token and token not in STOPWORDS}


def make_item_text(frame: pd.DataFrame) -> pd.Series:
    columns = ["title", "category", "brand", "gender", "age_group", "attributes"]
    result = pd.Series("", index=frame.index, dtype="string")
    for column in columns:
        result = result.str.cat(frame[column].fillna("").astype("string"), sep=" ")
    return normalize_series(result)


def extract_numbers(text: object) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\b", normalize_text(text)))


def extract_primary_number(text: object) -> str:
    match = re.search(r"\b\d+(?:[.,]\d+)?\b", normalize_text(text))
    return match.group(0) if match else ""


# %% [markdown]
# ## 6. Kategori ve ürün metni hazırlığı
#
# `top_category`, `/` ile ayrılmış kategori yolunun ilk; `leaf_category` son parçasıdır.

# %%
raw_category = items["category"].fillna("").astype("string")
items["top_category"] = raw_category.str.split("/").str[0].fillna("")
items["second_category"] = raw_category.str.split("/").str[1].fillna("")
items["parent_category"] = raw_category.str.rsplit("/", n=1).str[0].fillna("")
items["leaf_category"] = raw_category.str.split("/").str[-1].fillna("")

# attributes çoğunlukla en büyük metin alanıdır. 966 bin üründe ikinci bir tam kopya
# oluşturmamak için katalog genelinde normalize edilmez; yalnızca seçilen eğitim
# satırlarında normalize edilir.
for column in ["title", "category", "brand", "gender", "age_group",
               "top_category", "second_category", "parent_category", "leaf_category"]:
    items[column] = normalize_series(items[column])

terms["query"] = normalize_series(terms["query"])

if ADD_ASCII_TEXT_COLUMNS:
    terms["query_ascii"] = terms["query"].map(ascii_fold).astype("string")

# Attribute mismatch için katalogda ilk görülen rengi tek bir hafif sütunda tut.
COLOR_PATTERN = r"\b(" + "|".join(sorted(COLORS, key=len, reverse=True)) + r")\b"
NUMBER_PATTERN = r"\b(\d+(?:[.,]\d+)?)\b"
title_color = items["title"].str.extract(COLOR_PATTERN, expand=False)
missing_color = title_color.isna()
attribute_color = pd.Series(pd.NA, index=items.index, dtype="string")
attribute_color.loc[missing_color] = (
    items.loc[missing_color, "attributes"]
    .fillna("")
    .astype("string")
    .str.lower()
    .str.extract(COLOR_PATTERN, expand=False)
)
items["primary_color"] = title_color.fillna(attribute_color).fillna("").astype("string")
title_number = items["title"].str.extract(NUMBER_PATTERN, expand=False)
missing_number = title_number.isna()
attribute_number = pd.Series(pd.NA, index=items.index, dtype="string")
attribute_number.loc[missing_number] = (
    items.loc[missing_number, "attributes"]
    .fillna("")
    .astype("string")
    .str.extract(NUMBER_PATTERN, expand=False)
)
items["primary_number"] = title_number.fillna(attribute_number).fillna("").astype("string")
del title_color, missing_color, attribute_color, title_number, missing_number, attribute_number, raw_category

# Sık filtrelenen düşük kardinaliteli alanlarda category dtype RAM'i azaltır.
for column in ["top_category", "second_category", "parent_category", "leaf_category",
               "gender", "age_group", "primary_color", "primary_number"]:
    items[column] = items[column].astype("category")

gc.collect()


# %% [markdown]
# ## 7. Pozitif dataframe ve indeksler

# %%
positive_df = (
    positive_pairs
    .merge(terms, on="term_id", how="left", validate="many_to_one")
    .merge(items, on="item_id", how="left", validate="many_to_one")
)

missing_products = positive_df["title"].isna().sum()
missing_terms = positive_df["query"].isna().sum()
assert missing_products == 0, f"Pozitiflerde katalogda bulunmayan {missing_products} item_id var."
assert missing_terms == 0, f"Pozitiflerde terms.csv'de bulunmayan {missing_terms} term_id var."

positive_df["label"] = np.int8(1)
positive_df["negative_type"] = "positive"
positive_df["negative_confidence"] = "not_applicable"
positive_df["attributes"] = normalize_series(positive_df["attributes"])

ALL_ITEM_POSITIONS = np.arange(len(items), dtype=np.int64)
ITEM_IDS = items["item_id"].astype(str).to_numpy()
ITEM_TITLES = items["title"].astype(str).to_numpy()
ITEM_CATEGORIES = items["category"].astype(str).to_numpy()
ITEM_ATTRIBUTES = items["attributes"].fillna("").astype(str).to_numpy()
ITEM_TOPS = items["top_category"].astype(str).to_numpy()
ITEM_SECONDS = items["second_category"].astype(str).to_numpy()
ITEM_PARENTS = items["parent_category"].astype(str).to_numpy()
ITEM_LEAVES = items["leaf_category"].astype(str).to_numpy()
ITEM_GENDERS = items["gender"].astype(str).to_numpy()
ITEM_AGES = items["age_group"].astype(str).to_numpy()
ITEM_BRANDS = items["brand"].astype(str).to_numpy()
ITEM_COLORS = items["primary_color"].astype(str).to_numpy()
ITEM_NUMBERS = items["primary_number"].astype(str).to_numpy()


def item_text_at(item_position: int) -> str:
    return " ".join([
        ITEM_TITLES[item_position], ITEM_CATEGORIES[item_position], ITEM_BRANDS[item_position],
        ITEM_GENDERS[item_position], ITEM_AGES[item_position], ITEM_ATTRIBUTES[item_position],
    ])

top_to_indices = {
    str(key): np.asarray(index_values, dtype=np.int64)
    for key, index_values in items.groupby("top_category", observed=True).indices.items()
}

second_to_indices = {
    str(key): np.asarray(index_values, dtype=np.int64)
    for key, index_values in items.groupby("second_category", observed=True).indices.items()
}

parent_to_indices = {
    str(key): np.asarray(index_values, dtype=np.int64)
    for key, index_values in items.groupby("parent_category", observed=True).indices.items()
}

leaf_to_indices = {
    str(key): np.asarray(index_values, dtype=np.int64)
    for key, index_values in items.groupby("leaf_category", observed=True).indices.items()
}

parent_gender_age_to_indices = {
    tuple(map(str, key)): np.asarray(index_values, dtype=np.int64)
    for key, index_values in items.groupby(
        ["parent_category", "gender", "age_group"], observed=True
    ).indices.items()
}

second_gender_age_to_indices = {
    tuple(map(str, key)): np.asarray(index_values, dtype=np.int64)
    for key, index_values in items.groupby(
        ["second_category", "gender", "age_group"], observed=True
    ).indices.items()
}

top_keys = [key for key in top_to_indices if key not in UNKNOWN_VALUES]
used_negative_by_term: dict[str, set[str]] = defaultdict(set)


# %% [markdown]
# ## 8–9. Güvenlik ve örnekleme yardımcıları

# %%
def is_known_positive(term_id: object, item_id: object) -> bool:
    return str(item_id) in known_positive_by_term.get(str(term_id), set())


def extract_colors(text: object) -> set[str]:
    return token_set(text) & COLORS


def extract_gender_terms(text: object) -> set[str]:
    tokens = token_set(text)
    result = set()
    if tokens & {"kadın", "kadin", "bayan", "kız", "kiz"}:
        result.add("kadın")
    if "erkek" in tokens:
        result.add("erkek")
    if "unisex" in tokens:
        result.add("unisex")
    return result


def extract_age_terms(text: object) -> set[str]:
    tokens = token_set(text)
    result = set()
    if tokens & {"bebek", "yenidoğan", "yenidogan"}:
        result.add("bebek")
    if tokens & {"çocuk", "cocuk", "kız", "kiz", "oğlan", "oglan"}:
        result.add("çocuk")
    if tokens & {"yetişkin", "yetiskin"}:
        result.add("yetişkin")
    return result


def is_generic_query(query: object) -> bool:
    tokens = token_set(query)
    return len(tokens) <= 2 and bool(tokens) and tokens.issubset(GENERIC_PRODUCT_QUERIES)


def query_match_ratio(query: object, item_text: object) -> float:
    query_tokens = token_set(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & token_set(item_text)) / len(query_tokens)


def has_strong_query_match(query: object, item_text: object) -> bool:
    query_tokens = token_set(query)
    if not query_tokens:
        return False
    ratio = query_match_ratio(query, item_text)
    if is_generic_query(query):
        return ratio >= 0.60
    return len(query_tokens) >= 2 and ratio >= STRONG_MATCH_RATIO


def valid_brand(brand: object) -> bool:
    return normalize_text(brand) not in UNKNOWN_VALUES


def select_source_rows(frame: pd.DataFrame, rate: float, seed_offset: int) -> pd.DataFrame:
    if frame.empty or rate <= 0:
        return frame.iloc[0:0]
    count = min(len(frame), max(1, int(round(len(frame) * rate))))
    if count == len(frame):
        return frame
    return frame.sample(n=count, random_state=RANDOM_STATE + seed_offset)


def safe_sample_item(
    candidate_positions: Iterable[int],
    term_id: object,
    source_item_id: object,
    query: object,
    rng: np.random.Generator,
    apply_strong_match_filter: bool = True,
    max_attempts: int = MAX_SAMPLE_ATTEMPTS,
) -> Optional[int]:
    candidates = np.asarray(candidate_positions, dtype=np.int64)
    if candidates.size == 0:
        return None

    term_key = str(term_id)
    forbidden = known_positive_by_term.get(term_key, set()) | {str(source_item_id)}
    forbidden = forbidden | used_negative_by_term.get(term_key, set())

    attempt_positions = rng.choice(
        candidates,
        size=min(max_attempts, candidates.size),
        replace=False,
    )

    for item_position in np.atleast_1d(attempt_positions):
        candidate_id = ITEM_IDS[int(item_position)]
        if candidate_id in forbidden:
            continue
        if apply_strong_match_filter and has_strong_query_match(
            query, item_text_at(int(item_position))
        ):
            continue
        return int(item_position)
    return None


def make_negative_record(
    source_row,
    item_position: int,
    negative_type: str,
    confidence: str,
) -> dict:
    term_id = str(source_row.term_id)
    item_id = ITEM_IDS[item_position]
    used_negative_by_term[term_id].add(item_id)
    return {
        "term_id": term_id,
        "item_id": item_id,
        "label": np.int8(0),
        "negative_type": negative_type,
        "negative_confidence": confidence,
    }


def choose_other_top(source_top: str, rng: np.random.Generator) -> Optional[str]:
    candidates = [top for top in top_keys if top != source_top]
    return str(rng.choice(candidates)) if candidates else None


# %% [markdown]
# ## 10. Negatif üretim fonksiyonları

# %%
def make_easy_negatives(source: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 101)
    records = []
    rows = select_source_rows(source, STRATEGY_RATES["easy_random"], 101)

    for row in rows.itertuples(index=False):
        other_top = choose_other_top(str(row.top_category), rng)
        candidates = top_to_indices.get(other_top, ALL_ITEM_POSITIONS) if other_top else ALL_ITEM_POSITIONS
        position = safe_sample_item(candidates, row.term_id, row.item_id, row.query, rng, False)
        if position is None:  # Fallback: bütün katalogdan dene.
            position = safe_sample_item(ALL_ITEM_POSITIONS, row.term_id, row.item_id, row.query, rng, True)
        if position is not None:
            records.append(make_negative_record(row, position, "easy_random", "high"))
    return pd.DataFrame.from_records(records)


def make_different_top_category_negatives(source: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 202)
    records = []
    rows = select_source_rows(source, STRATEGY_RATES["different_top_category"], 202)

    for row in rows.itertuples(index=False):
        other_top = choose_other_top(str(row.top_category), rng)
        if other_top is None:
            continue
        position = safe_sample_item(
            top_to_indices[other_top], row.term_id, row.item_id, row.query, rng, False
        )
        if position is not None:
            records.append(
                make_negative_record(row, position, "different_top_category", "high")
            )
    return pd.DataFrame.from_records(records)


def make_same_top_category_hard_negatives(source: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 303)
    records = []
    rows = select_source_rows(source, STRATEGY_RATES["same_top_category_hard"], 303)

    for row in rows.itertuples(index=False):
        if is_generic_query(row.query):
            continue
        source_leaf = str(row.leaf_category)
        source_second = str(row.second_category)
        candidate_pools = []

        parent_candidates = parent_to_indices.get(
            str(row.parent_category), np.array([], dtype=np.int64)
        )
        if parent_candidates.size:
            candidate_pools.append(parent_candidates[ITEM_LEAVES[parent_candidates] != source_leaf])

        second_candidates = second_to_indices.get(
            source_second, np.array([], dtype=np.int64)
        )
        if second_candidates.size:
            candidate_pools.append(second_candidates[ITEM_LEAVES[second_candidates] != source_leaf])

        position = None
        for candidates in candidate_pools:
            position = safe_sample_item(
                candidates, row.term_id, row.item_id, row.query, rng, True
            )
            if position is not None:
                break
        if position is not None:
            records.append(
                make_negative_record(row, position, "same_top_category_hard", "medium")
            )
    return pd.DataFrame.from_records(records)


def make_same_gender_age_hard_negatives(source: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 404)
    records = []
    rows = select_source_rows(source, STRATEGY_RATES["same_gender_age_hard"], 404)

    for row in rows.itertuples(index=False):
        if is_generic_query(row.query):
            continue
        source_leaf = str(row.leaf_category)
        gender = str(row.gender)
        age = str(row.age_group)
        candidate_pools = [
            parent_gender_age_to_indices.get(
                (str(row.parent_category), gender, age), np.array([], dtype=np.int64)
            ),
            second_gender_age_to_indices.get(
                (str(row.second_category), gender, age), np.array([], dtype=np.int64)
            ),
        ]
        position = None
        for candidates in candidate_pools:
            if candidates.size:
                candidates = candidates[ITEM_LEAVES[candidates] != source_leaf]
            position = safe_sample_item(
                candidates, row.term_id, row.item_id, row.query, rng, True
            )
            if position is not None:
                break
        if position is not None:
            records.append(
                make_negative_record(row, position, "same_gender_age_hard", "medium")
            )
    return pd.DataFrame.from_records(records)


def make_brand_hard_negatives(source: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 505)
    records = []
    rows = select_source_rows(source, STRATEGY_RATES["brand_hard"], 505)

    for row in rows.itertuples(index=False):
        source_brand = normalize_text(row.brand)
        source_top = str(row.top_category)
        brand_tokens = token_set(source_brand)
        query_tokens = token_set(row.query)
        # Query markayı söylemiyorsa farklı marka, otomatik olarak negatif değildir.
        if (
            is_generic_query(row.query)
            or not valid_brand(source_brand)
            or not brand_tokens
            or not brand_tokens.issubset(query_tokens)
        ):
            continue

        candidate_pools = [
            leaf_to_indices.get(str(row.leaf_category), np.array([], dtype=np.int64)),
            parent_to_indices.get(str(row.parent_category), np.array([], dtype=np.int64)),
            top_to_indices.get(source_top, np.array([], dtype=np.int64)),
        ]
        position = None
        for candidates in candidate_pools:
            if candidates.size:
                candidates = candidates[
                    (ITEM_BRANDS[candidates] != source_brand)
                    & ~np.isin(ITEM_BRANDS[candidates], list(UNKNOWN_VALUES))
                ]
            position = safe_sample_item(
                candidates, row.term_id, row.item_id, row.query, rng, True
            )
            if position is not None:
                break
        if position is not None:
            records.append(make_negative_record(row, position, "brand_hard", "medium"))
    return pd.DataFrame.from_records(records)


def make_title_similar_hard_negatives(source: pd.DataFrame) -> pd.DataFrame:
    rows = select_source_rows(source, STRATEGY_RATES["title_similar_hard"], 606)
    rows = rows[~rows["query"].map(is_generic_query)].copy()
    if rows.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(RANDOM_STATE + 606)
    if USE_FULL_TFIDF_CATALOG or len(items) <= TFIDF_ITEM_POOL_SIZE:
        pool_positions = ALL_ITEM_POSITIONS
    else:
        pool_positions = np.sort(
            rng.choice(ALL_ITEM_POSITIONS, size=TFIDF_ITEM_POOL_SIZE, replace=False)
        )

    pool_frame = items.iloc[pool_positions]
    pool_text = normalize_series(
        pool_frame["title"].astype("string")
        .str.cat(pool_frame["category"].astype("string"), sep=" ")
        .str.cat(pool_frame["brand"].astype("string"), sep=" ")
    )
    min_df = 1 if len(pool_text) < 10_000 else 2
    ngram_range = (1, 2) if TFIDF_ANALYZER == "word" else (3, 5)
    vectorizer = TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=TFIDF_MAX_FEATURES,
        dtype=np.float32,
        sublinear_tf=True,
    )
    item_matrix = vectorizer.fit_transform(pool_text)
    neighbor_count = min(TFIDF_TOP_K, len(pool_positions))
    search = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit(item_matrix)

    records = []
    row_list = list(rows.itertuples(index=False))
    for batch_start in range(0, len(row_list), TFIDF_QUERY_BATCH_SIZE):
        batch_rows = row_list[batch_start: batch_start + TFIDF_QUERY_BATCH_SIZE]
        query_matrix = vectorizer.transform([str(row.query) for row in batch_rows])
        distances, neighbor_indices = search.kneighbors(query_matrix, return_distance=True)

        for row, row_distances, row_neighbors in zip(batch_rows, distances, neighbor_indices):
            chosen_position = None
            for distance, neighbor_index in zip(row_distances, row_neighbors):
                similarity = 1.0 - float(distance)
                if similarity < TFIDF_MIN_SIMILARITY:
                    break
                if similarity > TFIDF_MAX_SAFE_SIMILARITY:
                    continue
                item_position = int(pool_positions[int(neighbor_index)])
                candidate_id = ITEM_IDS[item_position]
                if ITEM_TOPS[item_position] != str(row.top_category):
                    continue
                if ITEM_SECONDS[item_position] != str(row.second_category):
                    continue
                if is_known_positive(row.term_id, candidate_id):
                    continue
                if candidate_id == str(row.item_id):
                    continue
                if candidate_id in used_negative_by_term.get(str(row.term_id), set()):
                    continue
                if has_strong_query_match(row.query, item_text_at(item_position)):
                    continue
                chosen_position = item_position
                break

            if chosen_position is not None:
                records.append(
                    make_negative_record(
                        row, chosen_position, "title_similar_hard", "low_medium"
                    )
                )

        del query_matrix, distances, neighbor_indices
        gc.collect()

    del item_matrix, search, vectorizer, pool_text, pool_frame
    gc.collect()
    return pd.DataFrame.from_records(records)


def make_attribute_mismatch_negatives(source: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 707)
    records = []
    rows = select_source_rows(source, STRATEGY_RATES["attribute_mismatch"], 707)

    for row in rows.itertuples(index=False):
        if is_generic_query(row.query):
            continue
        query_colors = extract_colors(row.query)
        query_genders = extract_gender_terms(row.query)
        query_ages = extract_age_terms(row.query)
        query_number = extract_primary_number(row.query)
        if not (query_colors or query_genders or query_ages or query_number):
            continue

        # Attribute farkını ürün tipi farkıyla karıştırmamak için en dar kategori
        # havuzu tercih edilir; yalnızca aday yoksa hiyerarşide yukarı çıkılır.
        candidate_pools = [
            leaf_to_indices.get(str(row.leaf_category), np.array([], dtype=np.int64)),
            parent_to_indices.get(str(row.parent_category), np.array([], dtype=np.int64)),
            second_to_indices.get(str(row.second_category), np.array([], dtype=np.int64)),
            top_to_indices.get(str(row.top_category), np.array([], dtype=np.int64)),
        ]
        base_candidates = next(
            (pool for pool in candidate_pools if pool.size > 1),
            np.array([], dtype=np.int64),
        )
        if base_candidates.size == 0:
            continue

        mismatch_mask = np.zeros(base_candidates.size, dtype=bool)

        if query_colors:
            known_color = ~np.isin(ITEM_COLORS[base_candidates], list(UNKNOWN_VALUES))
            color_mismatch = ~np.isin(ITEM_COLORS[base_candidates], list(query_colors))
            mismatch_mask |= known_color & color_mismatch

        if query_genders:
            known_gender = ~np.isin(ITEM_GENDERS[base_candidates], list(UNKNOWN_VALUES))
            gender_mismatch = ~np.isin(ITEM_GENDERS[base_candidates], list(query_genders))
            mismatch_mask |= known_gender & gender_mismatch

        if query_ages:
            known_age = ~np.isin(ITEM_AGES[base_candidates], list(UNKNOWN_VALUES))
            age_mismatch = ~np.isin(ITEM_AGES[base_candidates], list(query_ages))
            mismatch_mask |= known_age & age_mismatch

        if query_number:
            known_number = ~np.isin(ITEM_NUMBERS[base_candidates], list(UNKNOWN_VALUES))
            number_mismatch = ITEM_NUMBERS[base_candidates] != query_number
            mismatch_mask |= known_number & number_mismatch

        mismatch_candidates = base_candidates[mismatch_mask]
        position = safe_sample_item(
            mismatch_candidates, row.term_id, row.item_id, row.query, rng, False
        )
        if position is not None:
            records.append(
                make_negative_record(row, position, "attribute_mismatch", "medium_high")
            )
    return pd.DataFrame.from_records(records)


# %% [markdown]
# ## 11. Bütün negatif stratejilerini çalıştırma

# %%
negative_frames = []

STRATEGY_FUNCTIONS = [
    make_easy_negatives,
    make_different_top_category_negatives,
    make_same_top_category_hard_negatives,
    make_same_gender_age_hard_negatives,
    make_brand_hard_negatives,
    make_title_similar_hard_negatives,
    make_attribute_mismatch_negatives,
]

for strategy_function in STRATEGY_FUNCTIONS:
    frame = strategy_function(positive_df)
    if frame is None or frame.empty:
        print(f"{strategy_function.__name__}: 0")
        continue
    negative_frames.append(frame)
    print(f"{strategy_function.__name__}: {len(frame):,}")

if not negative_frames:
    raise RuntimeError("Hiç negatif üretilemedi. Kategori/metin alanlarını kontrol edin.")

negatives = pd.concat(negative_frames, ignore_index=True)
del negative_frames
gc.collect()


# %% [markdown]
# ## 12. Duplicate ve collision temizliği
#
# Aynı çift birden fazla stratejiyle üretilmişse daha değerli hard-negative türü korunur.

# %%
TYPE_PRIORITY = {
    "attribute_mismatch": 1,
    "title_similar_hard": 2,
    "brand_hard": 3,
    "same_gender_age_hard": 4,
    "same_top_category_hard": 5,
    "different_top_category": 6,
    "easy_random": 7,
}

negatives["_priority"] = negatives["negative_type"].map(TYPE_PRIORITY).fillna(99)
negatives = (
    negatives.sort_values("_priority")
    .drop_duplicates(["term_id", "item_id"], keep="first")
    .drop(columns="_priority")
    .reset_index(drop=True)
)

# Tüm katalog gerçekliği ve pozitif collision kontrolleri.
catalog_index = pd.Index(items["item_id"].astype(str))
assert negatives["item_id"].isin(catalog_index).all(), "Katalog dışı negatif item_id bulundu."

collision_mask = np.fromiter(
    (
        is_known_positive(term_id, item_id)
        for term_id, item_id in zip(negatives["term_id"], negatives["item_id"])
    ),
    dtype=bool,
    count=len(negatives),
)
if collision_mask.any():
    print(f"Bilinen pozitif collision temizlendi: {int(collision_mask.sum()):,}")
    negatives = negatives.loc[~collision_mask].reset_index(drop=True)

# NEGATIVE_RATIO üst sınırı. Strateji oranları nedeniyle genelde zaten bunun altındadır.
target_negative_count = int(round(len(positive_df) * NEGATIVE_RATIO))
if len(negatives) > target_negative_count:
    negatives = (
        negatives.groupby("negative_type", group_keys=False)
        .apply(lambda group: group.sample(
            n=max(1, int(round(target_negative_count * len(group) / len(negatives)))),
            random_state=RANDOM_STATE,
        ))
        .drop_duplicates(["term_id", "item_id"])
        .head(target_negative_count)
        .reset_index(drop=True)
    )

print(f"Temizlenmiş toplam negatif: {len(negatives):,}")


# %% [markdown]
# ## 13. Negatifleri metinlerle zenginleştirip pozitiflerle birleştirme

# %%
negatives_enriched = (
    negatives
    .merge(terms, on="term_id", how="left", validate="many_to_one")
    .merge(items, on="item_id", how="left", validate="many_to_one")
)
negatives_enriched["attributes"] = normalize_series(negatives_enriched["attributes"])

TYPE_WEIGHTS = {
    "positive": 1.00,
    "easy_random": 1.00,
    "different_top_category": 1.00,
    "attribute_mismatch": 0.90,
    "same_gender_age_hard": 0.75,
    "same_top_category_hard": 0.70,
    "brand_hard": 0.70,
    "title_similar_hard": 0.55,
}
positive_df["sample_weight"] = positive_df["negative_type"].map(TYPE_WEIGHTS).astype("float32")
negatives_enriched["sample_weight"] = (
    negatives_enriched["negative_type"].map(TYPE_WEIGHTS).fillna(0.60).astype("float32")
)

FINAL_COLUMNS = [
    "term_id", "item_id", "query", "title", "category", "brand", "gender",
    "age_group", "attributes", "label", "negative_type", "negative_confidence",
    "sample_weight",
]

train_with_negatives = pd.concat(
    [positive_df[FINAL_COLUMNS], negatives_enriched[FINAL_COLUMNS]],
    ignore_index=True,
)
train_with_negatives["label"] = train_with_negatives["label"].astype("int8")
train_with_negatives = train_with_negatives.sample(
    frac=1.0, random_state=RANDOM_STATE
).reset_index(drop=True)

assert not train_with_negatives.duplicated(["term_id", "item_id"]).any()
assert train_with_negatives.loc[
    train_with_negatives["label"].eq(0), "item_id"
].isin(catalog_index).all()

del negatives_enriched
gc.collect()


# %% [markdown]
# ## 14. Dağılım raporu

# %%
print("\nLABEL DAĞILIMI")
print(train_with_negatives["label"].value_counts(dropna=False))

print("\nNEGATIVE TYPE DAĞILIMI")
print(train_with_negatives["negative_type"].value_counts(dropna=False))

print("\nTOP 20 CATEGORY")
print(train_with_negatives["category"].value_counts(dropna=False).head(20))

print("\nGENDER DAĞILIMI")
print(train_with_negatives["gender"].value_counts(dropna=False).head(20))

print("\nAGE GROUP DAĞILIMI")
print(train_with_negatives["age_group"].value_counts(dropna=False).head(20))


# %% [markdown]
# ## 15. Küçük örnekleri gösterme

# %%
DISPLAY_COLUMNS = [
    "query", "title", "category", "label", "negative_type", "negative_confidence",
    "sample_weight",
]


def show_examples(mask: pd.Series, title: str, count: int = 10) -> None:
    sample = train_with_negatives.loc[mask, DISPLAY_COLUMNS]
    print(f"\n--- {title} ({len(sample):,} satırdan en fazla {count}) ---")
    if sample.empty:
        print("Örnek bulunamadı.")
    else:
        print(sample.sample(min(count, len(sample)), random_state=RANDOM_STATE).to_string(index=False))


show_examples(train_with_negatives["label"].eq(1), "POZİTİFLER")
show_examples(train_with_negatives["negative_type"].eq("easy_random"), "EASY NEGATIVE")
show_examples(
    train_with_negatives["negative_type"].isin([
        "same_top_category_hard", "same_gender_age_hard", "brand_hard", "title_similar_hard"
    ]),
    "HARD NEGATIVE",
)
show_examples(
    train_with_negatives["negative_type"].eq("attribute_mismatch"),
    "ATTRIBUTE MISMATCH",
)


# %% [markdown]
# ## 16. Eğitim dataframe'ini kaydetme
#
# CSV her zaman yazılır. Ortamda `pyarrow` veya `fastparquet` varsa daha hızlı ve küçük
# olan Parquet dosyası da yazılır.

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_with_negatives.to_csv(OUTPUT_CSV, index=False)
print(f"\nCSV kaydedildi: {OUTPUT_CSV} ({len(train_with_negatives):,} satır)")

# Her negatif türünden en fazla 50 satırlık dengeli göz kontrol dosyası.
audit_parts = []
for negative_type, group in train_with_negatives.groupby("negative_type", sort=True):
    audit_parts.append(
        group.sample(min(50, len(group)), random_state=RANDOM_STATE)
    )
audit_sample = pd.concat(audit_parts, ignore_index=True)
audit_sample.to_csv(OUTPUT_AUDIT, index=False)

sampling_stats = (
    train_with_negatives.groupby(
        ["label", "negative_type", "negative_confidence"], dropna=False
    )
    .agg(row_count=("item_id", "size"), mean_weight=("sample_weight", "mean"))
    .reset_index()
)
sampling_stats.to_csv(OUTPUT_STATS, index=False)
print(f"Göz kontrol örneği: {OUTPUT_AUDIT}")
print(f"Strateji istatistikleri: {OUTPUT_STATS}")

try:
    train_with_negatives.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Parquet kaydedildi: {OUTPUT_PARQUET}")
except (ImportError, ModuleNotFoundError, ValueError) as error:
    print(f"Parquet yazılamadı; CSV hazır. Sebep: {error}")

print("\nTamamlandı.")
