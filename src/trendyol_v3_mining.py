#!/usr/bin/env python3
"""Fold-local curriculum negative mining with explicit false-negative triage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from trendyol_v3_core import (
    ACCESSORY_WORDS,
    MAIN_PRODUCT_WORDS,
    IntentLexicons,
    add_contradiction_features,
    build_intent_lexicons,
    normalize_category,
    normalize_text,
    product_family_key,
    set_global_seed,
)
from trendyol_v3_validation import CandidateProfile, SplitManifest


NEGATIVE_TYPES: tuple[str, ...] = (
    "random",
    "same_parent_different_leaf",
    "same_leaf_wrong_brand",
    "same_category_brand_wrong_model",
    "same_product_wrong_gender",
    "same_product_wrong_age_group",
    "same_product_wrong_color_material",
    "lexical_overlap_wrong_product",
    "accessory_main_product",
    "complementary_product",
    "near_title_critical_attribute",
    "bi_encoder_high_score",
    "cross_encoder_uncertain",
    "ensemble_disagreement",
)


@dataclass(frozen=True)
class NegativeMiningConfig:
    """Reproducible curriculum and false-negative filtering configuration."""

    negative_ratio: int = 2
    max_negatives_per_term: int = 0
    curriculum_easy_epochs: int = 1
    curriculum_medium_epochs: int = 2
    suspicious_embedding_threshold: float = 0.93
    suspicious_cross_encoder_threshold: float = 0.90
    suspicious_teacher_threshold: float = 0.90
    suspicious_vote_threshold: int = 2
    uncertainty_low: float = 0.35
    uncertainty_high: float = 0.65
    suspicious_weight: float = 0.05
    ambiguous_weight: float = 0.20
    hard_weight: float = 0.75
    medium_weight: float = 0.90
    easy_weight: float = 1.00
    seed: int = 42

    def validate(self) -> None:
        """Validate ratios, curriculum and probability thresholds."""

        if self.negative_ratio not in {1, 2, 3}:
            raise ValueError("negative_ratio yalnızca 1, 2 veya 3 olmalıdır")
        if self.max_negatives_per_term < 0:
            raise ValueError("max_negatives_per_term negatif olamaz")
        for name, value in (
            ("suspicious_embedding_threshold", self.suspicious_embedding_threshold),
            ("suspicious_cross_encoder_threshold", self.suspicious_cross_encoder_threshold),
            ("suspicious_teacher_threshold", self.suspicious_teacher_threshold),
            ("uncertainty_low", self.uncertainty_low),
            ("uncertainty_high", self.uncertainty_high),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} [0,1] aralığında olmalıdır")
        if self.uncertainty_low >= self.uncertainty_high:
            raise ValueError("uncertainty_low, uncertainty_high değerinden küçük olmalı")


@dataclass(frozen=True)
class FoldScope:
    """Immutable training/validation boundary carried by every miner artifact."""

    scheme: str
    fold: int
    train_term_ids: frozenset[str]
    valid_term_ids: frozenset[str]
    manifest_hash: str
    forbidden_item_ids: frozenset[str] = frozenset()
    forbidden_family_ids: frozenset[str] = frozenset()

    @property
    def train_term_hash(self) -> str:
        """Hash the exact train-term lineage."""

        joined = "\n".join(sorted(self.train_term_ids))
        return hashlib.sha256(joined.encode()).hexdigest()

    def validate(self) -> None:
        """Assert non-empty, term-disjoint scope sets."""

        if not self.train_term_ids or not self.valid_term_ids:
            raise ValueError("FoldScope train/valid term kümeleri boş olamaz")
        overlap = self.train_term_ids.intersection(self.valid_term_ids)
        if overlap:
            raise ValueError(f"FoldScope term leakage: {len(overlap):,}")


@dataclass
class CatalogIndex:
    """Compact catalog arrays and metadata lookup pools."""

    items: pd.DataFrame
    id_to_position: dict[str, int]
    parent_to_positions: dict[str, np.ndarray]
    leaf_to_positions: dict[str, np.ndarray]
    leaf_brand_to_positions: dict[tuple[str, str], np.ndarray]
    token_to_positions: dict[str, np.ndarray]
    accessory_positions: np.ndarray
    main_product_positions: np.ndarray
    lexicons: IntentLexicons

    @classmethod
    def build(
        cls,
        items: pd.DataFrame,
        query_vocabulary: Iterable[str] = (),
    ) -> "CatalogIndex":
        """Build deterministic metadata pools, indexing only needed query tokens."""

        required = {"item_id", "title", "category", "brand", "gender", "age_group", "attributes"}
        if missing := required - set(items.columns):
            raise ValueError(f"Catalog index kolonları eksik: {sorted(missing)}")
        if items["item_id"].isna().any() or items["item_id"].duplicated().any():
            raise ValueError("Catalog item_id boş veya duplicate")
        frame = items[list(required)].copy().reset_index(drop=True)
        frame["item_id"] = frame["item_id"].astype(str)
        frame["title_norm"] = frame["title"].fillna("").map(normalize_text)
        frame["category_norm"] = frame["category"].fillna("").map(normalize_category)
        parts = frame["category_norm"].str.split("/")
        frame["top_category"] = parts.str[0].fillna("")
        frame["leaf_category"] = parts.str[-1].fillna("")
        frame["parent_category"] = parts.map(lambda values: "/".join(values[:-1]) if len(values) > 1 else "")
        for column in ("brand", "gender", "age_group", "attributes"):
            frame[f"{column}_norm"] = frame[column].fillna("").map(normalize_text)
        frame["family_id"] = [product_family_key(row._asdict()) for row in frame.itertuples(index=False)]
        frame["is_accessory"] = frame["title_norm"].map(lambda text: any(word in text for word in ACCESSORY_WORDS))
        frame["is_main_product"] = frame["title_norm"].map(
            lambda text: any(f" {word} " in f" {text} " for word in MAIN_PRODUCT_WORDS)
        )

        def group_positions(columns: str | list[str]) -> dict[Any, np.ndarray]:
            """Map group keys to integer catalog positions."""

            groups = frame.groupby(columns, sort=False, observed=True).indices
            return {key: np.asarray(value, dtype=np.int32) for key, value in groups.items()}

        wanted_tokens = {normalize_text(token) for token in query_vocabulary if normalize_text(token)}
        token_lists: dict[str, list[int]] = {token: [] for token in wanted_tokens}
        if wanted_tokens:
            for position, title in enumerate(frame["title_norm"]):
                for token in set(title.split()).intersection(wanted_tokens):
                    token_lists[token].append(position)
        token_to_positions = {
            token: np.asarray(positions, dtype=np.int32)
            for token, positions in token_lists.items() if positions
        }
        return cls(
            items=frame,
            id_to_position={item_id: int(index) for index, item_id in enumerate(frame["item_id"])},
            parent_to_positions=group_positions("parent_category"),
            leaf_to_positions=group_positions("leaf_category"),
            leaf_brand_to_positions=group_positions(["leaf_category", "brand_norm"]),
            token_to_positions=token_to_positions,
            accessory_positions=np.flatnonzero(frame["is_accessory"].to_numpy()).astype(np.int32),
            main_product_positions=np.flatnonzero(frame["is_main_product"].to_numpy()).astype(np.int32),
            lexicons=build_intent_lexicons(frame),
        )


def build_fold_scope(
    manifest: SplitManifest,
    positives: pd.DataFrame,
    fold: int,
    *,
    item_family_map: Mapping[str, str] | None = None,
) -> FoldScope:
    """Create a scope and optional item/family purge sets from validation positives."""

    manifest.validate()
    train_terms, valid_terms = manifest.scope(fold)
    if not {"term_id", "item_id"}.issubset(positives.columns):
        raise ValueError("positives term_id,item_id içermeli")
    valid_items = frozenset(
        positives.loc[positives["term_id"].astype(str).isin(valid_terms), "item_id"].astype(str)
    )
    forbidden_items = valid_items if manifest.config.item_policy in {"purge_item", "purge_family"} else frozenset()
    forbidden_families: frozenset[str] = frozenset()
    if manifest.config.item_policy == "purge_family":
        if item_family_map is None:
            raise ValueError("purge_family politikası item_family_map gerektirir")
        forbidden_families = frozenset(
            str(item_family_map[item_id]) for item_id in valid_items if item_id in item_family_map
        )
    scope = FoldScope(
        scheme=manifest.config.kind,
        fold=fold,
        train_term_ids=train_terms,
        valid_term_ids=valid_terms,
        manifest_hash=manifest.source_hash,
        forbidden_item_ids=forbidden_items,
        forbidden_family_ids=forbidden_families,
    )
    scope.validate()
    return scope


def _rng_for(scope: FoldScope, config: NegativeMiningConfig, epoch: int, term_id: str, strategy: str) -> np.random.Generator:
    """Create a scope-stable per-term per-strategy RNG."""

    token = f"{config.seed}|{scope.scheme}|{scope.fold}|{epoch}|{term_id}|{strategy}"
    seed = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
    return np.random.default_rng(seed)


def _sample_positions(
    pool: np.ndarray,
    count: int,
    rng: np.random.Generator,
    excluded_positions: set[int],
) -> list[int]:
    """Sample unique available catalog positions without replacement."""

    if count <= 0 or len(pool) == 0:
        return []
    available = np.asarray([int(value) for value in pool if int(value) not in excluded_positions], dtype=np.int64)
    if len(available) == 0:
        return []
    size = min(count, len(available))
    chosen = rng.choice(available, size=size, replace=False)
    return [int(value) for value in np.atleast_1d(chosen)]


def _mode(series: pd.Series) -> str:
    """Return a deterministic normalized mode or an empty value."""

    values = series.dropna().astype(str)
    if values.empty:
        return ""
    counts = values.value_counts()
    maximum = counts.max()
    return sorted(counts[counts.eq(maximum)].index)[0]


def _strategy_schedule(epoch: int, config: NegativeMiningConfig) -> list[str]:
    """Return allowed negative types for the current curriculum phase."""

    easy = ["random", "same_parent_different_leaf", "complementary_product"]
    medium = easy + [
        "same_leaf_wrong_brand", "same_product_wrong_gender", "same_product_wrong_age_group",
        "same_product_wrong_color_material", "lexical_overlap_wrong_product", "accessory_main_product",
    ]
    hard = list(NEGATIVE_TYPES)
    if epoch < config.curriculum_easy_epochs:
        return easy
    if epoch < config.curriculum_medium_epochs:
        return medium
    return hard


def _external_candidates_by_type(
    external_candidates: pd.DataFrame | None,
    scope: FoldScope,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Validate and index model/retrieval candidates by term and negative type."""

    if external_candidates is None:
        return {}
    required = {"term_id", "item_id", "negative_type"}
    if missing := required - set(external_candidates.columns):
        raise ValueError(f"External candidate kolonları eksik: {sorted(missing)}")
    invalid_types = set(external_candidates["negative_type"].dropna()) - set(NEGATIVE_TYPES)
    if invalid_types:
        raise ValueError(f"Bilinmeyen negative_type: {sorted(invalid_types)}")
    terms = set(external_candidates["term_id"].astype(str))
    if not terms.issubset(scope.train_term_ids):
        raise ValueError(f"External candidates validation term içeriyor: {len(terms - scope.train_term_ids):,}")
    if "miner_train_term_hash" in external_candidates:
        hashes = set(external_candidates["miner_train_term_hash"].dropna().astype(str))
        if hashes and hashes != {scope.train_term_hash}:
            raise ValueError("External miner lineage mevcut fold scope ile uyuşmuyor")
    return {
        (str(term_id), str(negative_type)): group.copy()
        for (term_id, negative_type), group in external_candidates.groupby(["term_id", "negative_type"], sort=False)
    }


def _candidate_record(
    term_id: str,
    query: str,
    item_id: str,
    negative_type: str,
    hardness_score: float,
    source_positive_item: str,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one canonical negative candidate record."""

    record: dict[str, Any] = {
        "term_id": term_id,
        "query": query,
        "item_id": item_id,
        "label": np.int8(0),
        "negative_type": negative_type,
        "hardness_score": np.float32(np.clip(hardness_score, 0.0, 1.0)),
        "source_positive_item": source_positive_item,
    }
    if extras:
        record.update(extras)
    return record


def generate_fold_negatives(
    scope: FoldScope,
    positives: pd.DataFrame,
    terms: pd.DataFrame,
    catalog: CatalogIndex,
    config: NegativeMiningConfig,
    *,
    epoch: int = 0,
    external_candidates: pd.DataFrame | None = None,
    miner_model_hash: str = "catalog_only",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate curriculum negatives only for the outer fold's training queries.

    The first returned frame contains usable negatives.  The second contains
    suspicious/excluded candidates for audit and optional low-weight ablation.
    """

    config.validate()
    scope.validate()
    if epoch < 0:
        raise ValueError("epoch negatif olamaz")
    set_global_seed(config.seed)
    required_pos = {"term_id", "item_id"}
    if missing := required_pos - set(positives.columns):
        raise ValueError(f"Pozitif kolonları eksik: {sorted(missing)}")
    train_pos = positives[positives["term_id"].astype(str).isin(scope.train_term_ids)][["term_id", "item_id"]].copy()
    if train_pos.empty:
        raise ValueError("Fold training pozitifleri boş")
    if set(train_pos["term_id"].astype(str)) - scope.train_term_ids:
        raise RuntimeError("Train positives scope dışı term içeriyor")
    query_map = terms.drop_duplicates("term_id").set_index("term_id")["query"]
    missing_terms = set(train_pos["term_id"].astype(str)) - set(query_map.index.astype(str))
    if missing_terms:
        raise ValueError(f"terms tablosunda olmayan train term sayısı={len(missing_terms):,}")
    known = train_pos.groupby("term_id", sort=False)["item_id"].agg(lambda values: set(values.astype(str))).to_dict()
    catalog_frame = catalog.items
    item_lookup = catalog_frame.set_index("item_id", drop=False)
    missing_items = set(train_pos["item_id"].astype(str)) - set(catalog.id_to_position)
    if missing_items:
        raise ValueError(f"Catalogda bulunmayan train positive item sayısı={len(missing_items):,}")
    external = _external_candidates_by_type(external_candidates, scope)
    schedule = _strategy_schedule(epoch, config)
    records: list[dict[str, Any]] = []

    for term_id, positive_group in train_pos.groupby("term_id", sort=False):
        term_id = str(term_id)
        query = str(query_map.loc[term_id])
        positive_ids = known[term_id]
        positive_meta = item_lookup.loc[sorted(positive_ids)]
        if isinstance(positive_meta, pd.Series):
            positive_meta = positive_meta.to_frame().T
        target = config.negative_ratio * len(positive_ids)
        if config.max_negatives_per_term:
            target = min(target, config.max_negatives_per_term)
        per_strategy = max(1, int(math.ceil(target / max(1, len(schedule)))))
        excluded_positions = {catalog.id_to_position[item_id] for item_id in positive_ids}
        source_item = sorted(positive_ids)[0]
        profile = {
            "parent": _mode(positive_meta["parent_category"]),
            "leaf": _mode(positive_meta["leaf_category"]),
            "brand": _mode(positive_meta["brand_norm"]),
            "gender": _mode(positive_meta["gender_norm"]),
            "age_group": _mode(positive_meta["age_group_norm"]),
        }
        pools: dict[str, np.ndarray] = {
            "random": np.arange(len(catalog_frame), dtype=np.int32),
            "same_parent_different_leaf": catalog.parent_to_positions.get(profile["parent"], np.empty(0, dtype=np.int32)),
            "same_leaf_wrong_brand": catalog.leaf_to_positions.get(profile["leaf"], np.empty(0, dtype=np.int32)),
            "same_category_brand_wrong_model": catalog.leaf_brand_to_positions.get(
                (profile["leaf"], profile["brand"]), np.empty(0, dtype=np.int32)
            ),
            "same_product_wrong_gender": catalog.leaf_to_positions.get(profile["leaf"], np.empty(0, dtype=np.int32)),
            "same_product_wrong_age_group": catalog.leaf_to_positions.get(profile["leaf"], np.empty(0, dtype=np.int32)),
            "same_product_wrong_color_material": catalog.leaf_to_positions.get(profile["leaf"], np.empty(0, dtype=np.int32)),
            "accessory_main_product": catalog.accessory_positions if not any(word in normalize_text(query) for word in ACCESSORY_WORDS) else catalog.main_product_positions,
            "complementary_product": catalog.parent_to_positions.get(profile["parent"], np.empty(0, dtype=np.int32)),
            "near_title_critical_attribute": catalog.leaf_to_positions.get(profile["leaf"], np.empty(0, dtype=np.int32)),
        }
        query_tokens = [token for token in normalize_text(query).split() if len(token) >= 3]
        lexical_arrays = [catalog.token_to_positions[token] for token in query_tokens if token in catalog.token_to_positions]
        pools["lexical_overlap_wrong_product"] = (
            np.unique(np.concatenate(lexical_arrays)).astype(np.int32)
            if lexical_arrays else np.empty(0, dtype=np.int32)
        )
        selected_for_term: set[str] = set()
        for strategy in schedule:
            added_for_strategy = 0
            ext = external.get((term_id, strategy))
            if ext is not None:
                score_col = next(
                    (column for column in ("hardness_score", "bi_encoder_score", "cross_encoder_probability", "disagreement_score") if column in ext),
                    None,
                )
                ext_ordered = ext.sort_values(score_col, ascending=False) if score_col else ext
                for _, row in ext_ordered.head(per_strategy * 3).iterrows():
                    item_id = str(row["item_id"])
                    if item_id in positive_ids or item_id in selected_for_term or item_id not in catalog.id_to_position:
                        continue
                    extras = {
                        column: row[column]
                        for column in (
                            "bi_encoder_score", "cross_encoder_probability", "teacher_probability",
                            "ensemble_vote_count", "disagreement_score", "retrieval_rank", "retrieval_score",
                        ) if column in row.index and pd.notna(row[column])
                    }
                    hardness = float(row[score_col]) if score_col and pd.notna(row[score_col]) else 0.8
                    records.append(_candidate_record(term_id, query, item_id, strategy, hardness, source_item, extras))
                    selected_for_term.add(item_id)
                    added_for_strategy += 1
                    if len(selected_for_term) >= target or added_for_strategy >= per_strategy:
                        break
            if len(selected_for_term) >= target:
                break
            if added_for_strategy >= per_strategy:
                continue
            pool = pools.get(strategy, np.empty(0, dtype=np.int32))
            rng = _rng_for(scope, config, epoch, term_id, strategy)
            candidates = _sample_positions(pool, per_strategy * 3, rng, excluded_positions)
            for position in candidates:
                item = catalog_frame.iloc[position]
                item_id = str(item["item_id"])
                if item_id in selected_for_term:
                    continue
                if strategy == "same_parent_different_leaf" and item["leaf_category"] == profile["leaf"]:
                    continue
                if strategy == "same_leaf_wrong_brand" and item["brand_norm"] == profile["brand"]:
                    continue
                if strategy == "same_product_wrong_gender" and item["gender_norm"] == profile["gender"]:
                    continue
                if strategy == "same_product_wrong_age_group" and item["age_group_norm"] == profile["age_group"]:
                    continue
                hardness_map = {
                    "random": 0.10, "same_parent_different_leaf": 0.35,
                    "complementary_product": 0.40, "same_leaf_wrong_brand": 0.60,
                    "same_product_wrong_gender": 0.65, "same_product_wrong_age_group": 0.65,
                    "same_product_wrong_color_material": 0.70, "lexical_overlap_wrong_product": 0.72,
                    "accessory_main_product": 0.75, "same_category_brand_wrong_model": 0.82,
                    "near_title_critical_attribute": 0.85,
                }
                records.append(_candidate_record(
                    term_id, query, item_id, strategy, hardness_map.get(strategy, 0.5), source_item
                ))
                selected_for_term.add(item_id)
                added_for_strategy += 1
                if len(selected_for_term) >= target or added_for_strategy >= per_strategy:
                    break
            if len(selected_for_term) >= target:
                break
        if len(selected_for_term) < target:
            rng = _rng_for(scope, config, epoch, term_id, "random_fallback")
            fallback = _sample_positions(
                np.arange(len(catalog_frame), dtype=np.int32), target - len(selected_for_term), rng,
                excluded_positions.union({catalog.id_to_position[value] for value in selected_for_term}),
            )
            for position in fallback:
                item_id = str(catalog_frame.iloc[position]["item_id"])
                records.append(_candidate_record(term_id, query, item_id, "random", 0.05, source_item))
                selected_for_term.add(item_id)

    candidates = pd.DataFrame.from_records(records).drop_duplicates(["term_id", "item_id"], keep="first")
    if candidates.empty:
        raise RuntimeError("Hiç negatif aday üretilemedi")
    enriched = candidates.merge(
        catalog_frame[[
            "item_id", "title", "category", "brand", "gender", "age_group", "attributes", "family_id"
        ]], on="item_id", how="left", validate="many_to_one",
    )
    enriched = add_contradiction_features(enriched, catalog.lexicons)
    usable, suspicious = triage_false_negatives(enriched, train_pos, catalog, config, scope)
    config_hash = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    for frame in (usable, suspicious):
        frame["scheme"] = scope.scheme
        frame["fold"] = np.int16(scope.fold)
        frame["epoch"] = np.int16(epoch)
        frame["manifest_hash"] = scope.manifest_hash
        frame["generator_config_hash"] = config_hash
        frame["miner_model_hash"] = miner_model_hash
        frame["miner_train_term_hash"] = scope.train_term_hash
        frame["pair_uid"] = [
            hashlib.blake2b(f"{scope.scheme}|{scope.fold}|{epoch}|{term}|{item}".encode(), digest_size=12).hexdigest()
            for term, item in zip(frame["term_id"], frame["item_id"])
        ]
    validate_mined_negatives(usable, train_pos, scope, catalog)
    return usable.reset_index(drop=True), suspicious.reset_index(drop=True)


def triage_false_negatives(
    candidates: pd.DataFrame,
    positives: pd.DataFrame,
    catalog: CatalogIndex,
    config: NegativeMiningConfig,
    scope: FoldScope,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter known positives, product families and multi-model suspect positives."""

    required = {"term_id", "item_id", "family_id", "hardness_score", "negative_type", "contradiction_any"}
    if missing := required - set(candidates.columns):
        raise ValueError(f"False-negative triage kolonları eksik: {sorted(missing)}")
    known_index = pd.MultiIndex.from_frame(positives[["term_id", "item_id"]].astype(str))
    pair_index = pd.MultiIndex.from_frame(candidates[["term_id", "item_id"]].astype(str))
    known_collision = pair_index.isin(known_index)
    positive_families = positives.copy()
    positive_families["family_id"] = positive_families["item_id"].astype(str).map(
        lambda item_id: catalog.items.iloc[catalog.id_to_position[item_id]]["family_id"]
    )
    family_sets = positive_families.groupby("term_id")["family_id"].agg(set).to_dict()
    same_family = np.asarray([
        family_id in family_sets.get(str(term_id), set())
        for term_id, family_id in zip(candidates["term_id"], candidates["family_id"])
    ])
    forbidden_item = candidates["item_id"].astype(str).isin(scope.forbidden_item_ids).to_numpy()
    forbidden_family = candidates["family_id"].astype(str).isin(scope.forbidden_family_ids).to_numpy()
    def optional_numeric(column: str, default: float = 0.0) -> np.ndarray:
        """Read an optional numeric score column as a dense aligned array."""

        values = candidates[column] if column in candidates else pd.Series(default, index=candidates.index)
        return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy()

    embedding = optional_numeric("bi_encoder_score")
    cross = optional_numeric("cross_encoder_probability")
    teacher = optional_numeric("teacher_probability")
    votes = optional_numeric("ensemble_vote_count")
    contradiction = candidates["contradiction_any"].fillna(0).astype(bool).to_numpy()
    model_signal_count = (
        (cross >= config.suspicious_cross_encoder_threshold).astype(int)
        + (teacher >= config.suspicious_teacher_threshold).astype(int)
        + (embedding >= config.suspicious_embedding_threshold).astype(int)
    )
    model_suspect = (
        (model_signal_count >= config.suspicious_vote_threshold)
        | (votes >= config.suspicious_vote_threshold)
    )
    high_embedding_without_conflict = (embedding >= config.suspicious_embedding_threshold) & ~contradiction
    excluded = known_collision | same_family | forbidden_item | forbidden_family
    suspicious_mask = ~excluded & (model_suspect | high_embedding_without_conflict)
    reasons = np.full(len(candidates), "", dtype=object)
    reasons[known_collision] = "known_positive_collision"
    reasons[~known_collision & same_family] = "same_product_family"
    reasons[~known_collision & ~same_family & forbidden_item] = "validation_item_purge"
    reasons[~known_collision & ~same_family & ~forbidden_item & forbidden_family] = "validation_family_purge"
    reasons[suspicious_mask & model_suspect] = "multi_model_suspect_positive"
    reasons[suspicious_mask & ~model_suspect & high_embedding_without_conflict] = "very_high_embedding_no_contradiction"
    status = np.where(excluded, "excluded", np.where(suspicious_mask, "suspicious", "train"))
    out = candidates.copy()
    out["same_family"] = same_family.astype(np.int8)
    out["false_negative_reason"] = reasons
    out["triage_status"] = status
    base_weight = np.select(
        [out["hardness_score"].ge(0.75), out["hardness_score"].ge(0.45)],
        [config.hard_weight, config.medium_weight], default=config.easy_weight,
    ).astype(np.float32)
    ambiguous = cross.between(config.uncertainty_low, config.uncertainty_high) if isinstance(cross, pd.Series) else (
        (cross >= config.uncertainty_low) & (cross <= config.uncertainty_high)
    )
    base_weight = np.where(ambiguous, np.minimum(base_weight, config.ambiguous_weight), base_weight)
    out["sample_weight"] = np.where(suspicious_mask, config.suspicious_weight, base_weight).astype(np.float32)
    usable = out[out["triage_status"].eq("train")].copy()
    audit = out[~out["triage_status"].eq("train")].copy()
    return usable, audit


def validate_mined_negatives(
    negatives: pd.DataFrame,
    positives: pd.DataFrame,
    scope: FoldScope,
    catalog: CatalogIndex,
) -> None:
    """Assert fold boundary, pair uniqueness, lineage and purge invariants."""

    scope.validate()
    required = {"term_id", "item_id", "label", "negative_type", "sample_weight", "pair_uid"}
    if missing := required - set(negatives.columns):
        raise ValueError(f"Mined negative kolonları eksik: {sorted(missing)}")
    terms = set(negatives["term_id"].astype(str))
    if not terms.issubset(scope.train_term_ids):
        raise ValueError(f"Mined negatives validation term içeriyor: {len(terms - scope.train_term_ids):,}")
    if negatives.duplicated(["term_id", "item_id"]).any() or negatives["pair_uid"].duplicated().any():
        raise ValueError("Mined negatives duplicate pair/pair_uid içeriyor")
    if not negatives["label"].eq(0).all():
        raise ValueError("Mined negative label değerleri 0 olmalıdır")
    positive_index = pd.MultiIndex.from_frame(positives[["term_id", "item_id"]].astype(str))
    negative_index = pd.MultiIndex.from_frame(negatives[["term_id", "item_id"]].astype(str))
    if negative_index.isin(positive_index).any():
        raise ValueError("Bilinen positive collision bulundu")
    if negatives["item_id"].astype(str).isin(scope.forbidden_item_ids).any():
        raise ValueError("Validation item purge ihlali")
    family = negatives["item_id"].astype(str).map(
        lambda item_id: catalog.items.iloc[catalog.id_to_position[item_id]]["family_id"]
    )
    if family.astype(str).isin(scope.forbidden_family_ids).any():
        raise ValueError("Validation family purge ihlali")
    weight = pd.to_numeric(negatives["sample_weight"], errors="coerce")
    if not np.isfinite(weight).all() or weight.le(0).any():
        raise ValueError("Mined negative sample_weight pozitif ve sonlu olmalı")


def build_fixed_validation_candidates(
    scope: FoldScope,
    retrieval_candidates: pd.DataFrame,
    positives: pd.DataFrame,
    profile: CandidateProfile,
    *,
    mode: str = "oracle",
    seed: int = 42,
) -> pd.DataFrame:
    """Freeze a test-shaped natural or oracle reranker validation universe."""

    if mode not in {"natural", "oracle"}:
        raise ValueError("mode natural veya oracle olmalıdır")
    required = {"term_id", "item_id"}
    if missing := required - set(retrieval_candidates.columns):
        raise ValueError(f"Retrieval candidate kolonları eksik: {sorted(missing)}")
    if missing := required - set(positives.columns):
        raise ValueError(f"Positive kolonları eksik: {sorted(missing)}")
    candidates = retrieval_candidates[
        retrieval_candidates["term_id"].astype(str).isin(scope.valid_term_ids)
    ].copy()
    if candidates.empty:
        raise ValueError("Validation scope için retrieval candidate yok")
    sort_columns = [column for column in ("term_id", "retrieval_rank", "retrieval_score", "item_id") if column in candidates]
    ascending_by_column = {
        "term_id": True, "retrieval_rank": True,
        "retrieval_score": False, "item_id": True,
    }
    ascending = [ascending_by_column[column] for column in sort_columns]
    candidates = candidates.sort_values(sort_columns, ascending=ascending).drop_duplicates(["term_id", "item_id"])
    known = positives[
        positives["term_id"].astype(str).isin(scope.valid_term_ids)
    ].groupby("term_id")["item_id"].agg(lambda values: set(values.astype(str))).to_dict()
    outputs: list[pd.DataFrame] = []
    for term_id in sorted(scope.valid_term_ids):
        group = candidates[candidates["term_id"].astype(str).eq(term_id)].copy()
        target = profile.deterministic_count(term_id, seed)
        selected = group.head(target).copy()
        positives_for_term = known.get(term_id, set())
        retrieved = set(selected["item_id"].astype(str))
        if mode == "oracle":
            missing_positive = sorted(positives_for_term - retrieved)
            if len(missing_positive) > target:
                raise ValueError(f"{term_id}: pozitif sayısı target candidate count'tan büyük")
            if missing_positive:
                keep = max(0, target - len(missing_positive))
                selected = selected.head(keep)
                extras = pd.DataFrame({"term_id": term_id, "item_id": missing_positive})
                extras["retrieval_rank"] = np.int32(0)
                extras["retrieval_score"] = np.float32(0.0)
                extras["retrieval_source"] = "forced_positive"
                selected = pd.concat([selected, extras], ignore_index=True, sort=False)
        selected["label"] = selected["item_id"].astype(str).isin(positives_for_term).astype(np.int8)
        selected["label_source"] = np.where(
            selected["label"].eq(1), "known_positive", "unobserved_assumed_negative"
        )
        selected["candidate_count"] = np.int32(len(selected))
        outputs.append(selected)
    result = pd.concat(outputs, ignore_index=True)
    result["scheme"] = scope.scheme
    result["fold"] = np.int16(scope.fold)
    result["validation_mode"] = mode
    result["manifest_hash"] = scope.manifest_hash
    result["pair_uid"] = [
        hashlib.blake2b(f"{scope.scheme}|{scope.fold}|valid|{term}|{item}".encode(), digest_size=12).hexdigest()
        for term, item in zip(result["term_id"], result["item_id"])
    ]
    ordered = result[["term_id", "item_id"]].astype(str)
    universe_hash = hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    ).hexdigest()
    result["candidate_universe_hash"] = universe_hash
    result.attrs["candidate_universe_hash"] = universe_hash
    if result.duplicated(["term_id", "item_id"]).any() or result["pair_uid"].duplicated().any():
        raise RuntimeError("Validation candidate universe duplicate içeriyor")
    return result


def false_negative_ablation_report(
    label: np.ndarray,
    baseline_probability: np.ndarray,
    filtered_probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Report before/after Macro-F1 for false-negative filtering ablations."""

    from trendyol_v3_validation import classification_report_dict

    rows: list[dict[str, Any]] = []
    for name, probability in (("unfiltered", baseline_probability), ("filtered", filtered_probability)):
        probability = np.asarray(probability, dtype=np.float64)
        metrics = classification_report_dict(
            label, (probability >= threshold).astype(np.int8), probability=probability, threshold=threshold
        )
        metrics["variant"] = name
        rows.append(metrics)
    report = pd.DataFrame(rows)
    report["macro_f1_delta"] = report["macro_f1"] - float(report.iloc[0]["macro_f1"])
    return report


def parse_args() -> argparse.Namespace:
    """Parse fold-local negative-mining CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3/negatives"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--negative-ratio", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--max-negatives-per-term", type=int, default=0)
    parser.add_argument("--external-candidates", type=Path)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Generate one fold/epoch negative artifact and its suspicious audit pool."""

    from trendyol_v3_core import AuditConfig, load_data_bundle
    from trendyol_v3_validation import SplitConfig

    args = parse_args()
    audit_config = AuditConfig(
        debug=args.debug,
        debug_items=20_000,
        debug_train_pairs=5_000,
        debug_submission_pairs=5_000,
        seed=args.seed,
    )
    bundle = load_data_bundle(args.data_dir, audit_config)
    manifest_frame = pd.read_parquet(args.manifest)
    meta = json.loads(args.manifest_meta.read_text(encoding="utf-8"))
    manifest = SplitManifest(manifest_frame, SplitConfig(**meta["config"]), meta["source_hash"])
    family_map = {
        str(row.item_id): product_family_key(row._asdict()) for row in bundle.items.itertuples(index=False)
    }
    scope = build_fold_scope(manifest, bundle.training_pairs, args.fold, item_family_map=family_map)
    train_terms = bundle.terms[bundle.terms["term_id"].astype(str).isin(scope.train_term_ids)]
    vocabulary = {token for query in train_terms["query"].fillna("") for token in normalize_text(query).split() if len(token) >= 3}
    catalog = CatalogIndex.build(bundle.items, vocabulary)
    external = None
    if args.external_candidates:
        external = pd.read_parquet(args.external_candidates) if args.external_candidates.suffix == ".parquet" else pd.read_csv(args.external_candidates)
    config = NegativeMiningConfig(
        negative_ratio=args.negative_ratio,
        max_negatives_per_term=args.max_negatives_per_term,
        seed=args.seed,
    )
    usable, suspicious = generate_fold_negatives(
        scope, bundle.training_pairs, bundle.terms, catalog, config,
        epoch=args.epoch, external_candidates=external,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{scope.scheme}_fold{scope.fold}_epoch{args.epoch}_ratio{args.negative_ratio}"
    usable.to_parquet(args.output_dir / f"{stem}_train.parquet", index=False)
    suspicious.to_parquet(args.output_dir / f"{stem}_suspicious.parquet", index=False)
    stats = {
        "usable_rows": len(usable), "suspicious_rows": len(suspicious),
        "usable_types": usable["negative_type"].value_counts().to_dict(),
        "suspicious_reasons": suspicious["false_negative_reason"].value_counts().to_dict(),
        "scope_train_term_hash": scope.train_term_hash,
    }
    (args.output_dir / f"{stem}_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
