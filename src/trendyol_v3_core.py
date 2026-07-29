#!/usr/bin/env python3
"""Schema-safe data loading, Turkish text views, intent parsing and dataset audit.

The module deliberately has no deep-learning dependency.  It is the first stage
of the v3 relevance pipeline and can be executed on a CPU-only Kaggle session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "pair_id", "row_id", "pairid"),
    "term_id": ("term_id", "query_id", "search_term_id", "termid", "queryid"),
    "item_id": ("item_id", "product_id", "sku", "productid", "itemid"),
    "query": ("query", "term", "search_term", "search_query", "query_text"),
    "label": ("label", "target", "relevance", "is_relevant", "prediction"),
    "prediction": ("prediction", "label", "target", "relevance"),
    "title": ("title", "item_title", "product_title", "name", "product_name"),
    "category": ("category", "category_path", "category_name", "taxonomy"),
    "brand": ("brand", "brand_name", "manufacturer"),
    "gender": ("gender", "sex", "cinsiyet"),
    "age_group": ("age_group", "age", "agegroup", "yas_grubu"),
    "attributes": ("attributes", "attribute", "attrs", "product_attributes", "features"),
}

FILE_ALIASES: dict[str, tuple[str, ...]] = {
    "items": ("items.csv", "products.csv", "catalog.csv"),
    "terms": ("terms.csv", "queries.csv", "search_terms.csv"),
    "training_pairs": ("training_pairs.csv", "train_pairs.csv", "train.csv"),
    "submission_pairs": ("submission_pairs.csv", "test_pairs.csv", "test.csv"),
    "sample_submission": ("sample_submission.csv", "submission_sample.csv"),
}

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "items": ("item_id", "title", "category", "brand", "gender", "age_group", "attributes"),
    "terms": ("term_id", "query"),
    "training_pairs": ("id", "term_id", "item_id", "label"),
    "submission_pairs": ("id", "term_id", "item_id"),
    "sample_submission": ("id", "prediction"),
}

OPTIONAL_COLUMNS: dict[str, set[str]] = {
    "items": {"category", "brand", "gender", "age_group", "attributes"},
    "training_pairs": {"id", "label"},
    "sample_submission": {"prediction"},
}

TURKISH_ASCII = str.maketrans({
    "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G", "ı": "i", "İ": "I",
    "ö": "o", "Ö": "O", "ş": "s", "Ş": "S", "ü": "u", "Ü": "U",
})

UNIT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\bterabyte\b", "tb"), (r"\bgigabyte\b", "gb"),
    (r"\bmililitre\b|\bmililiter\b", "ml"), (r"\blitre\b|\bliter\b|\blt\b", "l"),
    (r"\bkilogram\b|\bkilo\b", "kg"), (r"\bgram\b|\bgr\b", "g"),
    (r"\bsantimetre\b", "cm"), (r"\bmilimetre\b", "mm"),
    (r"\binch\b", "inç"),
)

COLOR_WORDS = {
    "siyah", "beyaz", "kırmızı", "kirmizi", "mavi", "lacivert", "bej", "krem",
    "haki", "bordo", "gri", "yeşil", "yesil", "pembe", "mor", "turuncu",
    "kahverengi", "ekru", "vizon", "taş", "tas", "füme", "fume", "antrasit",
    "sarı", "sari", "altın", "altin", "gümüş", "gumus",
}

GENDER_TERMS: dict[str, set[str]] = {
    "kadın": {"kadın", "kadin", "bayan"},
    "erkek": {"erkek", "bay"},
    "kız": {"kız", "kiz"},
    "erkek çocuk": {"erkek çocuk", "erkek cocuk"},
    "unisex": {"unisex"},
}

AGE_TERMS: dict[str, set[str]] = {
    "bebek": {"bebek", "yenidoğan", "yenidogan"},
    "çocuk": {"çocuk", "cocuk", "kids"},
    "yetişkin": {"yetişkin", "yetiskin"},
}

ACCESSORY_WORDS = {
    "kılıf", "kilif", "kapak", "ekran koruyucu", "şarj", "sarj", "adaptör", "adaptor",
    "kablo", "askı", "aski", "yedek parça", "filtre", "çanta", "canta", "stand",
}

MAIN_PRODUCT_WORDS = {
    "telefon", "tablet", "bilgisayar", "laptop", "televizyon", "tv", "kamera",
    "saat", "kulaklık", "kulaklik", "ayakkabı", "ayakkabi", "elbise", "mont",
}

ATTRIBUTE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "color": ("renk", "color detail", "color"),
    "material": ("materyal", "kumaş tipi", "kumas tipi", "malzeme"),
    "size": ("beden", "numara", "boyut/ebat", "ölçü", "olcu"),
    "capacity": ("kapasite", "hacim", "hafıza", "hafiza"),
    "model": ("model", "uyumlu model", "araç marka ve model", "arac marka ve model"),
    "gender": ("cinsiyet",),
    "age_group": ("yaş", "yas", "yaş grubu", "yas grubu"),
}

NUMBER_UNIT_PATTERN = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(tb|gb|mb|mah|ml|cl|l|kg|g|cm|mm|m|w|v|inç|inc)\b",
    flags=re.IGNORECASE,
)
MODEL_PATTERN = re.compile(r"\b(?=[a-zçğıöşü0-9-]*[a-zçğıöşü])(?=[a-zçğıöşü0-9-]*\d)[a-zçğıöşü0-9-]{2,}\b")
AGE_PATTERN = re.compile(r"\b(\d{1,2})\s*(?:yaş|yas)\b")
SHOE_SIZE_PATTERN = re.compile(r"\b(\d{2})\s*(?:numara|no)\b")
LETTER_SIZE_PATTERN = re.compile(r"\b(xxs|xs|s|m|l|xl|xxl|xxxl)\b")
NUMERIC_APPAREL_SIZE_PATTERN = re.compile(r"\b(\d{2,3})\s*beden\b")


@dataclass(frozen=True)
class DataPaths:
    """Resolved competition input paths."""

    items: Path
    terms: Path
    training_pairs: Path
    submission_pairs: Path
    sample_submission: Path


@dataclass
class DataBundle:
    """Canonical in-memory competition tables."""

    items: pd.DataFrame
    terms: pd.DataFrame
    training_pairs: pd.DataFrame
    submission_pairs: pd.DataFrame
    sample_submission: pd.DataFrame
    paths: DataPaths


@dataclass(frozen=True)
class AuditConfig:
    """Controls bounded expensive checks in the schema audit."""

    debug: bool = True
    debug_items: int = 5_000
    debug_train_pairs: int = 5_000
    debug_submission_pairs: int = 5_000
    top_n: int = 50
    seed: int = 42


@dataclass(frozen=True)
class IntentLexicons:
    """Catalog-derived dictionaries used by the query parser."""

    brands: frozenset[str]
    categories: frozenset[str]
    product_types: frozenset[str]


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (when installed) PyTorch deterministically."""

    if seed < 0:
        raise ValueError("seed negatif olamaz")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def _name_key(value: object) -> str:
    """Normalize a schema token for conservative alias matching."""

    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_column(
    columns: Sequence[object], canonical: str, *, required: bool = True,
) -> str | None:
    """Resolve one canonical column using exact normalized aliases.

    Raises an explicit error on missing or ambiguous matches.  Fuzzy substring
    matching is intentionally avoided because mapping ``item`` to ``item_title``
    silently would be more dangerous than stopping.
    """

    aliases = COLUMN_ALIASES.get(canonical, (canonical,))
    alias_keys = {_name_key(alias) for alias in aliases}
    matches = [str(column) for column in columns if _name_key(column) in alias_keys]
    if len(matches) > 1:
        raise ValueError(
            f"'{canonical}' kolonu belirsiz: {matches}. Kolonlardan birini yeniden adlandırın."
        )
    if not matches:
        if required:
            raise ValueError(
                f"'{canonical}' kolonu bulunamadı. Kabul edilen adlar={list(aliases)}; "
                f"mevcut kolonlar={list(map(str, columns))}"
            )
        return None
    return matches[0]


def discover_data_paths(data_dir: Path) -> DataPaths:
    """Find every required CSV using safe filename aliases."""

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Veri klasörü bulunamadı: {data_dir}")
    resolved: dict[str, Path] = {}
    lower_files = {path.name.lower(): path for path in data_dir.glob("*.csv")}
    for table, aliases in FILE_ALIASES.items():
        candidates = [lower_files[name.lower()] for name in aliases if name.lower() in lower_files]
        if len(candidates) > 1:
            raise ValueError(f"{table} için birden fazla aday dosya bulundu: {candidates}")
        if not candidates:
            raise FileNotFoundError(
                f"{table} CSV bulunamadı. Aranan dosyalar={list(aliases)}; klasör={data_dir}"
            )
        resolved[table] = candidates[0]
    return DataPaths(**resolved)


def read_canonical_csv(
    path: Path,
    table: str,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read a CSV with canonical names and stable ID dtypes."""

    if table not in TABLE_COLUMNS:
        raise ValueError(f"Bilinmeyen tablo türü: {table}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {path}")
    header = pd.read_csv(path, nrows=0)
    optional = OPTIONAL_COLUMNS.get(table, set())
    mapping: dict[str, str] = {}
    for canonical in TABLE_COLUMNS[table]:
        actual = resolve_column(header.columns, canonical, required=canonical not in optional)
        if actual is not None:
            mapping[actual] = canonical
    usecols = list(mapping)
    frame = pd.read_csv(path, usecols=usecols, dtype="string", nrows=nrows).rename(columns=mapping)
    if table == "training_pairs" and "label" not in frame:
        frame["label"] = np.int8(1)
    if table == "sample_submission" and "prediction" not in frame:
        frame["prediction"] = np.int8(0)
    for column in ("id", "term_id", "item_id"):
        if column in frame:
            frame[column] = frame[column].astype("string")
    if "label" in frame:
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int8")
    if "prediction" in frame:
        frame["prediction"] = pd.to_numeric(frame["prediction"], errors="raise").astype("int8")
    return frame


def read_debug_item_subset(
    path: Path,
    referenced_item_ids: Iterable[object],
    *,
    catalog_rows: int,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Read a small catalog sample plus every item referenced by debug pairs."""

    if catalog_rows < 1 or chunksize < 1:
        raise ValueError("catalog_rows/chunksize pozitif olmalıdır")
    first = read_canonical_csv(path, "items", nrows=catalog_rows)
    required_ids = {str(value) for value in referenced_item_ids if value is not None and not pd.isna(value)}
    missing_ids = required_ids - set(first["item_id"].astype(str))
    frames = [first]
    if missing_ids:
        header = pd.read_csv(path, nrows=0)
        mapping: dict[str, str] = {}
        optional = OPTIONAL_COLUMNS.get("items", set())
        for canonical in TABLE_COLUMNS["items"]:
            actual = resolve_column(header.columns, canonical, required=canonical not in optional)
            if actual is not None:
                mapping[actual] = canonical
        for chunk in pd.read_csv(path, usecols=list(mapping), dtype="string", chunksize=chunksize):
            chunk = chunk.rename(columns=mapping)
            selected = chunk[chunk["item_id"].astype(str).isin(missing_ids)]
            if not selected.empty:
                frames.append(selected)
                missing_ids -= set(selected["item_id"].astype(str))
            if not missing_ids:
                break
    if missing_ids:
        raise ValueError(f"Debug pair'lerinde katalogda bulunamayan item sayısı={len(missing_ids):,}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("item_id").reset_index(drop=True)


def read_grouped_pair_sample(
    path: Path,
    table: str,
    *,
    target_rows: int,
    seed: int,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Sample complete term groups so debug candidate/positive counts remain meaningful."""

    if table not in {"training_pairs", "submission_pairs"}:
        raise ValueError("Grouped pair sample yalnız training/submission pairs içindir")
    if target_rows < 1 or chunksize < 1:
        raise ValueError("target_rows/chunksize pozitif olmalıdır")
    header = pd.read_csv(path, nrows=0)
    term_actual = resolve_column(header.columns, "term_id")
    counts: Counter[str] = Counter()
    for chunk in pd.read_csv(path, usecols=[term_actual], dtype="string", chunksize=chunksize):
        counts.update(chunk[term_actual].dropna().astype(str).value_counts().to_dict())
    ordered_terms = sorted(
        counts,
        key=lambda term_id: int.from_bytes(
            hashlib.blake2b(f"{seed}:{table}:{term_id}".encode(), digest_size=8).digest(), "little"
        ),
    )
    selected: set[str] = set()
    selected_rows = 0
    for term_id in ordered_terms:
        selected.add(term_id)
        selected_rows += counts[term_id]
        if selected_rows >= target_rows:
            break
    mapping: dict[str, str] = {}
    optional = OPTIONAL_COLUMNS.get(table, set())
    for canonical in TABLE_COLUMNS[table]:
        actual = resolve_column(header.columns, canonical, required=canonical not in optional)
        if actual is not None:
            mapping[actual] = canonical
    frames = []
    for chunk in pd.read_csv(path, usecols=list(mapping), dtype="string", chunksize=chunksize):
        chunk = chunk.rename(columns=mapping)
        chosen = chunk[chunk["term_id"].astype(str).isin(selected)]
        if not chosen.empty:
            frames.append(chosen)
    if not frames:
        raise RuntimeError(f"{table} grouped debug sample boş")
    frame = pd.concat(frames, ignore_index=True)
    if table == "training_pairs" and "label" not in frame:
        frame["label"] = np.int8(1)
    if "label" in frame:
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int8")
    return frame


def read_sample_submission_for_ids(
    path: Path,
    ordered_ids: pd.Series,
    *,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Read sample-submission rows matching a grouped debug pair sample in source order."""

    id_set = set(ordered_ids.astype(str))
    header = pd.read_csv(path, nrows=0)
    mapping: dict[str, str] = {}
    for canonical in TABLE_COLUMNS["sample_submission"]:
        actual = resolve_column(
            header.columns, canonical,
            required=canonical not in OPTIONAL_COLUMNS.get("sample_submission", set()),
        )
        if actual is not None:
            mapping[actual] = canonical
    frames = []
    for chunk in pd.read_csv(path, usecols=list(mapping), dtype="string", chunksize=chunksize):
        chunk = chunk.rename(columns=mapping)
        chosen = chunk[chunk["id"].astype(str).isin(id_set)]
        if not chosen.empty:
            frames.append(chosen)
    if not frames:
        raise RuntimeError("Grouped debug sample için sample_submission satırı bulunamadı")
    frame = pd.concat(frames, ignore_index=True)
    if "prediction" not in frame:
        frame["prediction"] = np.int8(0)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="raise").astype("int8")
    if not frame["id"].astype(str).equals(ordered_ids.astype(str).reset_index(drop=True)):
        raise ValueError("Grouped debug sample ile sample_submission ID sırası uyuşmuyor")
    return frame


def load_data_bundle(data_dir: Path, config: AuditConfig | None = None) -> DataBundle:
    """Load all canonical tables and run referential safety assertions."""

    config = config or AuditConfig(debug=False)
    set_global_seed(config.seed)
    paths = discover_data_paths(data_dir)
    terms = read_canonical_csv(paths.terms, "terms")
    if config.debug:
        train = read_grouped_pair_sample(
            paths.training_pairs, "training_pairs",
            target_rows=config.debug_train_pairs, seed=config.seed,
        )
        test = read_grouped_pair_sample(
            paths.submission_pairs, "submission_pairs",
            target_rows=config.debug_submission_pairs, seed=config.seed,
        )
        sample = read_sample_submission_for_ids(paths.sample_submission, test["id"])
        items = read_debug_item_subset(
            paths.items,
            pd.concat([train["item_id"], test["item_id"]], ignore_index=True),
            catalog_rows=config.debug_items,
        )
    else:
        train = read_canonical_csv(paths.training_pairs, "training_pairs")
        test = read_canonical_csv(paths.submission_pairs, "submission_pairs")
        sample = read_canonical_csv(paths.sample_submission, "sample_submission")
        items = read_canonical_csv(paths.items, "items")
    for column in ("category", "brand", "gender", "age_group", "attributes"):
        if column not in items:
            items[column] = pd.Series("", index=items.index, dtype="string")
    _validate_bundle(items, terms, train, test, sample, debug=config.debug)
    return DataBundle(items, terms, train, test, sample, paths)


def _validate_bundle(
    items: pd.DataFrame,
    terms: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    debug: bool,
) -> None:
    """Validate IDs, labels, pairs and the sample-submission order."""

    for name, frame, key in (("items", items, "item_id"), ("terms", terms, "term_id")):
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{name}.{key} boş veya duplicate değer içeriyor")
    for name, frame in (("training_pairs", train), ("submission_pairs", test)):
        if frame[["term_id", "item_id"]].isna().any().any():
            raise ValueError(f"{name} boş term_id/item_id içeriyor")
        if frame.duplicated(["term_id", "item_id"]).any():
            raise ValueError(f"{name} duplicate (term_id,item_id) çifti içeriyor")
        if "id" in frame and (frame["id"].isna().any() or frame["id"].duplicated().any()):
            raise ValueError(f"{name}.id boş veya duplicate")
    if not set(train["label"].unique()).issubset({0, 1}):
        raise ValueError("training_pairs.label yalnızca 0/1 olmalıdır")
    known_terms = set(terms["term_id"].dropna())
    unknown_terms = (set(train["term_id"]) | set(test["term_id"])) - known_terms
    if unknown_terms:
        raise ValueError(f"terms tablosunda olmayan term_id sayısı={len(unknown_terms):,}")
    known_items = set(items["item_id"].dropna())
    unknown_items = (set(train["item_id"]) | set(test["item_id"])) - known_items
    if unknown_items:
        raise ValueError(f"items tablosunda olmayan item_id sayısı={len(unknown_items):,}")
    if len(sample) != len(test):
        raise ValueError(f"sample_submission={len(sample):,}, submission_pairs={len(test):,} satır")
    if "id" in test and not sample["id"].astype(str).equals(test["id"].astype(str)):
        raise ValueError("sample_submission id sırası submission_pairs ile aynı değil")


def turkish_lower(value: object) -> str:
    """Lowercase text with explicit Turkish I/İ behavior."""

    if value is None or pd.isna(value):
        return ""
    return str(value).replace("I", "ı").replace("İ", "i").lower()


def normalize_text(value: object, *, ascii_fold: bool = False) -> str:
    """Create a conservative, unit-preserving Turkish normalized view."""

    text = unicodedata.normalize("NFKC", turkish_lower(value))
    compact_units = {
        "terabyte": "tb", "gigabyte": "gb", "mililitre": "ml", "mililiter": "ml",
        "litre": "l", "liter": "l", "kilogram": "kg", "kilo": "kg",
        "gram": "g", "santimetre": "cm", "milimetre": "mm",
    }
    for unit_name, unit_symbol in compact_units.items():
        text = re.sub(rf"(?<=\d)\s*{unit_name}\b", f" {unit_symbol}", text)
    for pattern, replacement in UNIT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s+(tb|gb|mb|mah|ml|cl|l|kg|g|cm|mm|m|w|v|inç)\b", r"\1 \2", text)
    text = re.sub(r"[/|]+", " / ", text)
    text = re.sub(r"[^0-9a-zçğıöşü+./%-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if ascii_fold:
        text = text.translate(TURKISH_ASCII)
    return text


def normalize_category(value: object) -> str:
    """Normalize category parts while preserving hierarchy separators."""

    if value is None or pd.isna(value):
        return ""
    parts = [normalize_text(part) for part in str(value).split("/")]
    return "/".join(part for part in parts if part)


def parse_attribute_map(value: object) -> dict[str, str]:
    """Parse comma-separated ``key: value`` product attributes."""

    if value is None or pd.isna(value):
        return {}
    output: dict[str, str] = {}
    for part in str(value).split(","):
        if ":" not in part:
            continue
        key, raw_value = part.split(":", 1)
        key_norm, value_norm = normalize_text(key), normalize_text(raw_value)
        if key_norm and value_norm:
            output[key_norm] = value_norm
    return output


def select_relevant_attributes(value: object, query: object = "") -> str:
    """Keep important or query-overlapping attributes for the long text view."""

    parsed = parse_attribute_map(value)
    query_tokens = set(normalize_text(query).split())
    important = {alias for aliases in ATTRIBUTE_KEY_ALIASES.values() for alias in aliases}
    selected = [
        f"{key}: {item_value}"
        for key, item_value in parsed.items()
        if key in important or query_tokens.intersection(item_value.split())
    ]
    return " | ".join(selected)


def add_item_text_views(items: pd.DataFrame) -> pd.DataFrame:
    """Add original, normalized and ASCII short/long field-tagged item views."""

    required = {"title", "category", "brand", "gender", "age_group", "attributes"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"Ürün metni alanları eksik: {sorted(missing)}")
    out = items.copy()
    raw = {column: out[column].fillna("").astype(str) for column in required}
    out["item_text_short_original"] = (
        "[TITLE] " + raw["title"] + " [CATEGORY] " + raw["category"] + " [BRAND] " + raw["brand"]
    )
    out["item_text_long_original"] = (
        out["item_text_short_original"] + " [GENDER] " + raw["gender"]
        + " [AGE] " + raw["age_group"] + " [ATTRIBUTES] " + raw["attributes"]
    )
    normalized = {column: raw[column].map(normalize_category if column == "category" else normalize_text) for column in required}
    out["item_text_short"] = (
        "[TITLE] " + normalized["title"] + " [CATEGORY] " + normalized["category"]
        + " [BRAND] " + normalized["brand"]
    )
    out["item_text_long"] = (
        out["item_text_short"] + " [GENDER] " + normalized["gender"]
        + " [AGE] " + normalized["age_group"] + " [ATTRIBUTES] " + normalized["attributes"]
    )
    out["item_text_short_ascii"] = out["item_text_short"].map(lambda x: normalize_text(x, ascii_fold=True))
    out["item_text_long_ascii"] = out["item_text_long"].map(lambda x: normalize_text(x, ascii_fold=True))
    return out


def build_intent_lexicons(items: pd.DataFrame, *, min_brand_frequency: int = 2) -> IntentLexicons:
    """Learn conservative brand/category/product-type dictionaries from the catalog."""

    if min_brand_frequency < 1:
        raise ValueError("min_brand_frequency en az 1 olmalıdır")
    required = {"brand", "category"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"Intent sözlüğü için eksik kolonlar: {sorted(missing)}")
    brands = items["brand"].fillna("").map(normalize_text)
    invalid = {"", "unknown", "bilinmiyor", "none", "nan", "null", "-"}
    counts = brands[brands.str.len().between(2, 40) & ~brands.isin(invalid)].value_counts()
    brand_set = frozenset(counts[counts >= min_brand_frequency].index.astype(str))
    categories: set[str] = set()
    leaves: set[str] = set()
    for value in items["category"].fillna(""):
        parts = [normalize_text(part) for part in str(value).split("/") if normalize_text(part)]
        categories.update(parts)
        if parts:
            leaves.add(parts[-1])
    product_types = frozenset(value for value in leaves if 2 <= len(value) <= 60)
    return IntentLexicons(brand_set, frozenset(categories), product_types)


def _longest_lexicon_matches(text: str, lexicon: Iterable[str], limit: int = 5) -> list[str]:
    """Return longest boundary-safe phrase matches."""

    values = lexicon if isinstance(lexicon, (set, frozenset)) else set(lexicon)
    tokens = text.split()
    candidates = {
        " ".join(tokens[start:start + width])
        for width in range(1, min(6, len(tokens)) + 1)
        for start in range(0, len(tokens) - width + 1)
    }
    matches = list(candidates.intersection(values))
    matches.sort(key=lambda value: (-len(value.split()), -len(value), value))
    selected: list[str] = []
    for match in matches:
        if not any(f" {match} " in f" {existing} " for existing in selected):
            selected.append(match)
        if len(selected) >= limit:
            break
    return selected


def parse_query_intent(query: object, lexicons: IntentLexicons) -> dict[str, Any]:
    """Parse dictionary and regex-based query intent with mandatory-field hints."""

    text = normalize_text(query)
    ascii_text = normalize_text(query, ascii_fold=True)
    tokens = set(text.split()) | set(ascii_text.split())
    brands = _longest_lexicon_matches(text, lexicons.brands, limit=2)
    product_types = _longest_lexicon_matches(text, lexicons.product_types, limit=3)
    categories = _longest_lexicon_matches(text, lexicons.categories, limit=3)
    colors = sorted(tokens.intersection(COLOR_WORDS))
    genders = [canonical for canonical, values in GENDER_TERMS.items() if any(value in text for value in values)]
    ages = [canonical for canonical, values in AGE_TERMS.items() if any(value in text for value in values)]
    explicit_age = AGE_PATTERN.findall(text)
    if explicit_age:
        ages.extend(f"{age} yaş" for age in explicit_age)
    units = [f"{number.replace(',', '.')} {unit.lower()}" for number, unit in NUMBER_UNIT_PATTERN.findall(text)]
    models = sorted(set(MODEL_PATTERN.findall(text)))
    shoe_sizes = sorted(set(SHOE_SIZE_PATTERN.findall(text)))
    apparel_sizes = sorted(set(
        LETTER_SIZE_PATTERN.findall(text) + NUMERIC_APPAREL_SIZE_PATTERN.findall(text)
    ))
    accessory = any(word in text for word in ACCESSORY_WORDS)
    main_product = any(re.search(rf"\b{re.escape(word)}\b", text) for word in MAIN_PRODUCT_WORDS)
    mandatory: list[str] = []
    for field, values in (
        ("brand", brands), ("model", models), ("capacity_or_measure", units),
        ("gender", genders), ("age_group", ages), ("size", shoe_sizes + apparel_sizes),
    ):
        if values:
            mandatory.append(field)
    if colors:
        mandatory.append("color")
    if product_types:
        mandatory.append("product_type")
    return {
        "query_normalized": text,
        "query_ascii": ascii_text,
        "brands": brands,
        "product_types": product_types,
        "categories": categories,
        "colors": colors,
        "genders": sorted(set(genders)),
        "age_groups": sorted(set(ages)),
        "models": models,
        "capacities_measures": sorted(set(units)),
        "shoe_sizes": shoe_sizes,
        "apparel_sizes": apparel_sizes,
        "accessory_intent": accessory,
        "main_product_intent": main_product and not accessory,
        "mandatory_fields": sorted(set(mandatory)),
    }


def _normalized_contains_any(text: str, values: Iterable[str]) -> bool:
    """Check whether normalized text contains any full phrase."""

    padded = f" {normalize_text(text)} "
    return any(f" {normalize_text(value)} " in padded for value in values if value)


def contradiction_features(
    query: object,
    item: Mapping[str, object],
    lexicons: IntentLexicons,
) -> dict[str, int | float | str]:
    """Create query–item critical-attribute mismatch signals."""

    intent = parse_query_intent(query, lexicons)
    title = normalize_text(item.get("title", ""))
    category = normalize_category(item.get("category", ""))
    brand = normalize_text(item.get("brand", ""))
    gender = normalize_text(item.get("gender", ""))
    age_group = normalize_text(item.get("age_group", ""))
    attributes = parse_attribute_map(item.get("attributes", ""))
    attribute_text = " ".join(f"{key} {value}" for key, value in attributes.items())
    full_text = " ".join((title, category.replace("/", " "), brand, gender, age_group, attribute_text))

    def mismatch(values: Iterable[str], product_text: str) -> int:
        """Return one when an explicit query value is absent from product text."""

        values_list = list(values)
        return int(bool(values_list) and not _normalized_contains_any(product_text, values_list))

    query_accessory = bool(intent["accessory_intent"])
    product_accessory = any(word in full_text for word in ACCESSORY_WORDS)
    accessory_main_mismatch = int(query_accessory != product_accessory and (query_accessory or intent["main_product_intent"]))
    result: dict[str, int | float | str] = {
        "wrong_product_type": mismatch(intent["product_types"], f"{title} {category}"),
        "wrong_category": mismatch(intent["categories"], category),
        "wrong_brand": mismatch(intent["brands"], f"{brand} {title}"),
        "wrong_gender": mismatch(intent["genders"], f"{gender} {title} {attribute_text}"),
        "wrong_age_group": mismatch(intent["age_groups"], f"{age_group} {title} {attribute_text}"),
        "wrong_color": mismatch(intent["colors"], f"{title} {attribute_text}"),
        "wrong_size_measure": mismatch(
            list(intent["shoe_sizes"]) + list(intent["apparel_sizes"]), f"{title} {attribute_text}"
        ),
        "wrong_model_capacity": mismatch(
            list(intent["models"]) + list(intent["capacities_measures"]), f"{title} {attribute_text}"
        ),
        "accessory_main_mismatch": accessory_main_mismatch,
        "complementary_product_mismatch": int(
            bool(intent["product_types"]) and not _normalized_contains_any(f"{title} {category}", intent["product_types"])
            and _normalized_contains_any(full_text, intent["categories"])
        ),
    }
    critical = [key for key, value in result.items() if key.startswith("wrong_") or key.endswith("mismatch")]
    result["contradiction_count"] = int(sum(int(result[key]) for key in critical))
    result["contradiction_any"] = int(result["contradiction_count"] > 0)
    return result


def add_contradiction_features(pairs: pd.DataFrame, lexicons: IntentLexicons) -> pd.DataFrame:
    """Vector-like DataFrame wrapper for ``contradiction_features``."""

    required = {"query", "title", "category", "brand", "gender", "age_group", "attributes"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Çelişki feature kolonları eksik: {sorted(missing)}")
    records = [
        contradiction_features(row.query, row._asdict(), lexicons)
        for row in pairs[list(required)].itertuples(index=False)
    ]
    return pairs.reset_index(drop=True).join(pd.DataFrame.from_records(records))


def product_family_key(row: Mapping[str, object]) -> str:
    """Build a conservative variant-family key from title/category/brand."""

    title = normalize_text(row.get("title", ""), ascii_fold=True)
    title = NUMBER_UNIT_PATTERN.sub(" ", title)
    title = re.sub(r"\b(?:" + "|".join(map(re.escape, sorted(COLOR_WORDS, key=len, reverse=True))) + r")\b", " ", title)
    title = re.sub(
        r"\b(?:xxs|xs|s|m|l|xl|xxl|xxxl)\b|\b\d{2,3}\s*(?:beden|numara|no)\b",
        " ", title,
    )
    title = re.sub(r"\s+", " ", title).strip()
    category = normalize_category(row.get("category", ""))
    brand = normalize_text(row.get("brand", ""), ascii_fold=True)
    return hashlib.blake2b(f"{brand}|{category}|{title}".encode(), digest_size=12).hexdigest()


def _distribution(series: pd.Series) -> dict[str, float | int]:
    """Return stable descriptive statistics for a numeric series."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(values.mean()), "std": float(values.std(ddof=0)),
        "min": float(values.min()), "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)), "median": float(values.median()),
        "p75": float(values.quantile(0.75)), "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _query_profile(
    queries: pd.Series,
    lexicons: IntentLexicons,
) -> dict[str, float | int | dict[str, int]]:
    """Summarize query lexical and intent attributes."""

    normalized = queries.fillna("").map(normalize_text)
    intents = [parse_query_intent(value, lexicons) for value in normalized]
    return {
        "count": int(len(normalized)),
        "char_length": _distribution(normalized.str.len()),
        "token_length": _distribution(normalized.str.split().str.len()),
        "brand_rate": float(np.mean([bool(value["brands"]) for value in intents])) if intents else 0.0,
        "category_rate": float(np.mean([bool(value["categories"]) for value in intents])) if intents else 0.0,
        "gender_rate": float(np.mean([bool(value["genders"]) for value in intents])) if intents else 0.0,
        "age_rate": float(np.mean([bool(value["age_groups"]) for value in intents])) if intents else 0.0,
        "number_rate": float(normalized.str.contains(r"\d", regex=True).mean()) if len(normalized) else 0.0,
        "turkish_char_rate": float(normalized.str.contains(r"[çğıöşü]", regex=True).mean()) if len(normalized) else 0.0,
        "top_tokens": dict(Counter(token for text in normalized for token in text.split()).most_common(50)),
    }


def _table_audit(frame: pd.DataFrame, primary_id: str | None = None) -> dict[str, Any]:
    """Collect shape, dtype, null and duplicate diagnostics for one table."""

    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {column: int(value) for column, value in frame.isna().sum().items()},
        "semantic_missing": {
            column: int(
                frame[column].fillna("").astype(str).str.strip().str.lower().isin(
                    {"", "unknown", "bilinmiyor", "none", "nan", "null", "-"}
                ).sum()
            )
            for column in frame.columns
        },
        "duplicate_rows": int(frame.duplicated().sum()),
    }
    if primary_id and primary_id in frame:
        report["unique_ids"] = int(frame[primary_id].nunique(dropna=True))
        report["duplicate_ids"] = int(frame[primary_id].duplicated().sum())
    if {"term_id", "item_id"}.issubset(frame.columns):
        report["duplicate_pairs"] = int(frame.duplicated(["term_id", "item_id"]).sum())
    return report


def run_dataset_audit(bundle: DataBundle, output_dir: Path, config: AuditConfig) -> dict[str, Any]:
    """Run schema, distribution-shift and duplicate-family diagnostics and persist them."""

    set_global_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items, terms = bundle.items, bundle.terms
    train, test, sample = bundle.training_pairs, bundle.submission_pairs, bundle.sample_submission
    lexicons = build_intent_lexicons(items, min_brand_frequency=1 if config.debug else 2)
    query_map = terms.set_index("term_id")["query"]
    train_term_ids = pd.Index(train["term_id"].unique())
    test_term_ids = pd.Index(test["term_id"].unique())
    train_queries = query_map.reindex(train_term_ids).dropna()
    test_queries = query_map.reindex(test_term_ids).dropna()

    positive_per_term = train.groupby("term_id", sort=False).size().rename("positive_count")
    terms_per_item = train.groupby("item_id", sort=False)["term_id"].nunique().rename("term_count")
    candidates_per_term = test.groupby("term_id", sort=False).size().rename("candidate_count")
    positive_per_term.to_csv(output_dir / "positive_count_per_term.csv")
    terms_per_item.to_csv(output_dir / "term_count_per_item.csv")
    candidates_per_term.to_csv(output_dir / "candidate_count_per_term.csv")

    item_meta = items.set_index("item_id")
    train_item_ids = pd.Index(train["item_id"].unique()).intersection(item_meta.index)
    test_item_ids = pd.Index(test["item_id"].unique()).intersection(item_meta.index)
    train_products = item_meta.reindex(train_item_ids)
    test_products = item_meta.reindex(test_item_ids)
    overlap: dict[str, Any] = {
        "term_id_count": int(len(set(train_term_ids) & set(test_term_ids))),
        "item_id_count": int(len(set(train_item_ids) & set(test_item_ids))),
        "item_jaccard": float(len(set(train_item_ids) & set(test_item_ids)) / max(1, len(set(train_item_ids) | set(test_item_ids)))),
    }
    for field in ("category", "brand"):
        left = set(train_products[field].fillna("").map(normalize_text)) - {""}
        right = set(test_products[field].fillna("").map(normalize_text)) - {""}
        overlap[f"{field}_intersection"] = int(len(left & right))
        overlap[f"{field}_jaccard"] = float(len(left & right) / max(1, len(left | right)))

    candidate_meta = test[["term_id", "item_id"]].merge(
        items[["item_id", "category", "brand"]], on="item_id", how="left", validate="many_to_one"
    )
    category_distribution = (
        candidate_meta["category"].fillna("").map(normalize_category).value_counts(dropna=False).head(config.top_n)
        .rename_axis("category").reset_index(name="count")
    )
    category_distribution["rate"] = category_distribution["count"] / max(1, len(candidate_meta))
    category_distribution.to_csv(output_dir / "submission_candidate_category_distribution.csv", index=False)

    title_norm = items["title"].fillna("").map(lambda value: normalize_text(value, ascii_fold=True))
    title_groups = pd.DataFrame({"item_id": items["item_id"], "title_normalized": title_norm})
    exact = title_groups[title_groups["title_normalized"].duplicated(keep=False) & title_groups["title_normalized"].ne("")]
    exact = exact.sort_values(["title_normalized", "item_id"])
    exact.head(100_000).to_csv(output_dir / "duplicate_normalized_titles.csv", index=False)
    family_keys = [product_family_key(row._asdict()) for row in items.itertuples(index=False)]
    family = pd.DataFrame({"item_id": items["item_id"], "family_key": family_keys, "title": items["title"]})
    family = family[family["family_key"].duplicated(keep=False)].sort_values(["family_key", "item_id"])
    family.head(100_000).to_csv(output_dir / "near_duplicate_product_families.csv", index=False)

    report: dict[str, Any] = {
        "config": asdict(config),
        "files": {
            name: {"path": str(path), "bytes": int(path.stat().st_size)}
            for name, path in asdict(bundle.paths).items()
        },
        "tables": {
            "items": _table_audit(items, "item_id"),
            "terms": _table_audit(terms, "term_id"),
            "training_pairs": _table_audit(train, "id"),
            "submission_pairs": _table_audit(test, "id"),
            "sample_submission": _table_audit(sample, "id"),
        },
        "train": {
            "unique_terms": int(train["term_id"].nunique()),
            "unique_items": int(train["item_id"].nunique()),
            "positive_count_per_term": _distribution(positive_per_term),
            "term_count_per_item": _distribution(terms_per_item),
            "label_counts": {str(key): int(value) for key, value in train["label"].value_counts().items()},
        },
        "submission": {
            "unique_terms": int(test["term_id"].nunique()),
            "unique_items": int(test["item_id"].nunique()),
            "candidate_count_per_term": _distribution(candidates_per_term),
            "sample_order_equal": bool(sample["id"].astype(str).equals(test["id"].astype(str))),
        },
        "query_profiles": {
            "train": _query_profile(train_queries, lexicons),
            "test": _query_profile(test_queries, lexicons),
        },
        "overlap": overlap,
        "duplicate_products": {
            "exact_normalized_title_rows": int(len(exact)),
            "exact_normalized_title_groups": int(exact["title_normalized"].nunique()),
            "near_family_rows": int(len(family)),
            "near_family_groups": int(family["family_key"].nunique()),
        },
    }
    (output_dir / "schema_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    """Parse the standalone audit CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3/audit"))
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-items", type=int, default=5_000)
    parser.add_argument("--debug-train-pairs", type=int, default=5_000)
    parser.add_argument("--debug-submission-pairs", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Execute data loading and persist a human-readable audit summary."""

    args = parse_args()
    config = AuditConfig(
        debug=args.debug,
        debug_items=args.debug_items,
        debug_train_pairs=args.debug_train_pairs,
        debug_submission_pairs=args.debug_submission_pairs,
        seed=args.seed,
    )
    bundle = load_data_bundle(args.data_dir, config)
    report = run_dataset_audit(bundle, args.output_dir, config)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
