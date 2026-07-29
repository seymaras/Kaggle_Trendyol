#!/usr/bin/env python3
"""Stage-oriented Kaggle v3 orchestration from audit to model-ready fold artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval_utils import RetrievalConfig, SparseCatalogRetriever, retrieval_metrics
from trendyol_v3_core import (
    AuditConfig,
    DataBundle,
    load_data_bundle,
    normalize_text,
    run_dataset_audit,
    set_global_seed,
)
from trendyol_v3_mining import (
    CatalogIndex,
    NegativeMiningConfig,
    build_fixed_validation_candidates,
    build_fold_scope,
    generate_fold_negatives,
)
from trendyol_v3_reranker import build_selected_item_view, enrich_pair_frame
from trendyol_v3_validation import (
    CandidateProfile,
    SplitConfig,
    SplitManifest,
    audit_item_overlap,
    build_group_manifest,
    build_semantic_manifest,
    build_test_like_manifest,
    encode_query_embeddings,
    run_adversarial_validation,
)


@dataclass(frozen=True)
class PipelineConfig:
    """One reproducible CPU-preparation run configuration."""

    debug: bool = True
    debug_items: int = 5_000
    debug_train_pairs: int = 5_000
    debug_submission_pairs: int = 5_000
    n_splits: int = 3
    semantic_clusters: int = 48
    embedding_backend: Literal["lexical", "qwen"] = "lexical"
    negative_ratio: int = 2
    curriculum_epochs: tuple[int, ...] = (0, 2)
    retrieval_top_k: int = 300
    retrieval_batch_size: int = 64
    item_policy: Literal["audit", "purge_item", "purge_family"] = "audit"
    seed: int = 42

    def validate(self) -> None:
        """Validate debug bounds, folds and stage parameters."""

        if self.n_splits < 2 or self.semantic_clusters < self.n_splits:
            raise ValueError("n_splits/semantic_clusters geçersiz")
        if self.negative_ratio not in {1, 2, 3}:
            raise ValueError("negative_ratio 1,2,3 olmalıdır")
        if not self.curriculum_epochs or min(self.curriculum_epochs) < 0:
            raise ValueError("curriculum_epochs boş veya negatif olamaz")
        if self.retrieval_top_k < 10 or self.retrieval_batch_size < 1:
            raise ValueError("retrieval top-k/batch geçersiz")
        for value in (self.debug_items, self.debug_train_pairs, self.debug_submission_pairs):
            if value < 1:
                raise ValueError("Debug örnek sınırları pozitif olmalıdır")


def _pipeline_hash(config: PipelineConfig) -> str:
    """Hash the complete preparation configuration."""

    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def lexical_semantic_embeddings(term_frame: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Create a deterministic CPU-only semantic proxy for DEBUG smoke tests."""

    if not {"term_id", "query"}.issubset(term_frame.columns):
        raise ValueError("Lexical embeddings term_id,query gerektirir")
    if len(term_frame) < 3:
        raise ValueError("Lexical semantic split en az üç term gerektirir")
    text = term_frame["query"].fillna("").map(normalize_text)
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1,
        max_features=10_000, sublinear_tf=True, dtype=np.float32,
    ).fit_transform(text)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1,
        max_features=15_000, sublinear_tf=True, dtype=np.float32,
    ).fit_transform(text)
    matrix = sparse.hstack([word, char], format="csr")
    components = min(64, len(term_frame) - 1, matrix.shape[1] - 1)
    if components < 2:
        raise ValueError("Lexical embedding SVD için yeterli feature yok")
    embeddings = TruncatedSVD(n_components=components, random_state=seed).fit_transform(matrix)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norm, 1e-12)
    return embeddings.astype(np.float32)


def _term_frame_from_manifest(manifest: SplitManifest) -> pd.DataFrame:
    """Return ordered term/query rows used by semantic embedding generation."""

    return manifest.frame[["term_id", "query", "positive_count", "canonical_query_group"]].copy()


def build_validation_manifests(
    bundle: DataBundle,
    output_dir: Path,
    config: PipelineConfig,
) -> dict[str, SplitManifest]:
    """Build and save Validation A/B/C plus adversarial diagnostics."""

    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_config = SplitConfig(
        kind="group", n_splits=config.n_splits, seed=config.seed,
        semantic_clusters=config.semantic_clusters, item_policy=config.item_policy,
    )
    group = build_group_manifest(bundle.training_pairs, bundle.terms, group_config)
    group.save(output_dir)
    term_frame = _term_frame_from_manifest(group)
    if config.embedding_backend == "lexical":
        embeddings = lexical_semantic_embeddings(term_frame, config.seed)
    else:
        embeddings = encode_query_embeddings(
            term_frame, output_dir / "embedding_cache", seed=config.seed,
            device="cuda",
        )
    np.save(output_dir / "query_split_embeddings.f16.npy", embeddings.astype(np.float16))
    semantic_config = SplitConfig(
        kind="semantic", n_splits=config.n_splits, seed=config.seed,
        semantic_clusters=config.semantic_clusters, item_policy=config.item_policy,
    )
    semantic = build_semantic_manifest(
        bundle.training_pairs, bundle.terms, embeddings, semantic_config
    )
    semantic.save(output_dir)
    train_ids = set(bundle.training_pairs["term_id"].astype(str))
    test_ids = set(bundle.submission_pairs["term_id"].astype(str))
    train_terms = bundle.terms[bundle.terms["term_id"].astype(str).isin(train_ids)].reset_index(drop=True)
    test_terms = bundle.terms[bundle.terms["term_id"].astype(str).isin(test_ids)].reset_index(drop=True)
    adversarial = run_adversarial_validation(
        train_terms, test_terms, n_splits=config.n_splits, seed=config.seed,
        max_word_features=10_000 if config.debug else 40_000,
        max_char_features=10_000 if config.debug else 40_000,
    )
    adversarial.save(output_dir)
    test_like_config = SplitConfig(
        kind="test_like", n_splits=config.n_splits, seed=config.seed,
        semantic_clusters=config.semantic_clusters, test_like_fraction=0.15,
        item_policy=config.item_policy,
    )
    test_like = build_test_like_manifest(
        bundle.training_pairs, bundle.terms, adversarial.term_scores,
        test_like_config, semantic_clusters=semantic.frame["semantic_cluster"],
    )
    test_like.save(output_dir)
    family_map = bundle.items.assign(
        _family=bundle.items.apply(
            lambda row: hashlib.blake2b(
                f"{normalize_text(row.get('brand', ''))}|{normalize_text(row.get('category', ''))}|"
                f"{normalize_text(row.get('title', ''))}".encode(), digest_size=12
            ).hexdigest(), axis=1,
        )
    ).set_index("item_id")["_family"]
    for name, manifest in (("group", group), ("semantic", semantic), ("test_like", test_like)):
        audit_item_overlap(manifest, bundle.training_pairs, family_map=family_map).to_csv(
            output_dir / f"{name}_item_family_overlap.csv", index=False
        )
    return {"group": group, "semantic": semantic, "test_like": test_like}


def fit_catalog_retriever(
    items: pd.DataFrame,
    config: PipelineConfig,
) -> SparseCatalogRetriever:
    """Fit one reusable label-free word+character catalog retriever."""

    if items.empty:
        raise ValueError("Retrieval items boş olamaz")
    views = build_selected_item_view(items, "short")
    return SparseCatalogRetriever(RetrievalConfig(
        word_max_features=30_000 if config.debug else 150_000,
        char_max_features=30_000 if config.debug else 100_000,
        top_k=min(config.retrieval_top_k, len(items)),
        batch_size=config.retrieval_batch_size,
    )).fit(views["item_id"], views["product_text"])


def retrieve_catalog_candidates(
    terms: pd.DataFrame,
    retriever: SparseCatalogRetriever,
) -> pd.DataFrame:
    """Retrieve candidates for terms using an already fitted catalog index."""

    if terms.empty:
        raise ValueError("Retrieval terms boş olamaz")
    frames = list(retriever.iter_search(terms[["term_id", "query"]]))
    if not frames:
        raise RuntimeError("Catalog retrieval hiç aday üretmedi")
    candidates = pd.concat(frames, ignore_index=True)
    if candidates.duplicated(["term_id", "item_id"]).any():
        raise RuntimeError("Catalog retrieval duplicate pair üretti")
    return candidates


def augment_candidate_pool_to_profile(
    candidates: pd.DataFrame,
    term_ids: Sequence[str],
    item_ids: Sequence[str],
    profile: CandidateProfile,
    *,
    seed: int,
) -> pd.DataFrame:
    """Fill rare long-tail candidate counts with deterministic label-free random items."""

    required = {"term_id", "item_id"}
    if missing := required - set(candidates.columns):
        raise ValueError(f"Candidate augmentation kolonları eksik: {sorted(missing)}")
    catalog = np.asarray(list(map(str, item_ids)), dtype=object)
    if len(catalog) == 0:
        raise ValueError("Candidate augmentation item catalog boş")
    outputs = [candidates]
    grouped = {str(term): set(group["item_id"].astype(str)) for term, group in candidates.groupby("term_id")}
    for term_id in map(str, term_ids):
        target = profile.deterministic_count(term_id, seed)
        existing = grouped.get(term_id, set())
        needed = target - len(existing)
        if needed <= 0:
            continue
        digest = hashlib.blake2b(f"{seed}:candidate_fill:{term_id}".encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        if len(catalog) - len(existing) < needed:
            raise ValueError(f"{term_id}: catalog target candidate sayısını dolduramıyor")
        chosen_set: set[str] = set()
        while len(chosen_set) < needed:
            draw_size = max(32, 2 * (needed - len(chosen_set)))
            positions = rng.integers(0, len(catalog), size=draw_size)
            for position in positions:
                item_id = str(catalog[int(position)])
                if item_id not in existing:
                    chosen_set.add(item_id)
                if len(chosen_set) >= needed:
                    break
        chosen = np.asarray(sorted(chosen_set), dtype=object)
        outputs.append(pd.DataFrame({
            "term_id": term_id, "item_id": chosen,
            "retrieval_rank": np.arange(len(existing) + 1, len(existing) + needed + 1, dtype=np.int32),
            "retrieval_score": np.float32(0.0), "retrieval_source": "profile_random_fill",
        }))
    result = pd.concat(outputs, ignore_index=True, sort=False).drop_duplicates(["term_id", "item_id"])
    for term_id in map(str, term_ids):
        target = profile.deterministic_count(term_id, seed)
        if int(result["term_id"].astype(str).eq(term_id).sum()) < target:
            raise RuntimeError(f"{term_id}: augmented candidate pool target altında")
    return result.reset_index(drop=True)


def prepare_fold_training_frame(
    scope: Any,
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    terms: pd.DataFrame,
    items: pd.DataFrame,
    *,
    product_view: str = "long",
) -> pd.DataFrame:
    """Combine positives/negatives and create deterministic model text for one fold."""

    positive = positives[positives["term_id"].astype(str).isin(scope.train_term_ids)][
        ["term_id", "item_id"]
    ].copy()
    positive["label"] = np.int8(1)
    positive["sample_weight"] = np.float32(1.0)
    positive["negative_type"] = "known_positive"
    positive["hardness_score"] = np.float32(0.0)
    for column in (
        "contradiction_any", "wrong_category", "wrong_brand", "wrong_gender",
        "wrong_age_group", "wrong_color", "wrong_product_type",
    ):
        positive[column] = np.int8(0)
    negative_columns = [
        column for column in (
            "term_id", "item_id", "label", "sample_weight", "negative_type", "hardness_score",
            "contradiction_any", "wrong_category", "wrong_brand", "wrong_gender",
            "wrong_age_group", "wrong_color", "wrong_product_type", "teacher_probability",
        ) if column in negatives
    ]
    combined = pd.concat([positive, negatives[negative_columns]], ignore_index=True, sort=False)
    if combined.duplicated(["term_id", "item_id"]).any():
        raise RuntimeError("Fold training frame duplicate term-item içeriyor")
    enriched = enrich_pair_frame(combined, terms, items, product_view=product_view)
    enriched["pair_uid"] = [
        hashlib.blake2b(f"train|{scope.scheme}|{scope.fold}|{term}|{item}".encode(), digest_size=12).hexdigest()
        for term, item in zip(enriched["term_id"], enriched["item_id"])
    ]
    enriched["scheme"] = scope.scheme
    enriched["fold"] = np.int16(scope.fold)
    if not set(enriched["term_id"].astype(str)).issubset(scope.train_term_ids):
        raise RuntimeError("Prepared train frame validation term içeriyor")
    return enriched


def prepare_model_artifacts(
    bundle: DataBundle,
    manifests: dict[str, SplitManifest],
    output_dir: Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Create fold-local curriculum negatives and fixed test-shaped validation pools."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = CandidateProfile.fit(bundle.submission_pairs)
    train_query_ids = set(bundle.training_pairs["term_id"].astype(str))
    train_terms = bundle.terms[bundle.terms["term_id"].astype(str).isin(train_query_ids)]
    query_vocabulary = {
        token for query in train_terms["query"].fillna("")
        for token in normalize_text(query).split() if len(token) >= 3
    }
    catalog = CatalogIndex.build(bundle.items, query_vocabulary)
    family_map = catalog.items.set_index("item_id")["family_id"].to_dict()
    mining_config = NegativeMiningConfig(
        negative_ratio=config.negative_ratio,
        max_negatives_per_term=0,
        seed=config.seed,
    )
    summary: dict[str, Any] = {"candidate_profile": asdict(profile), "folds": {}}
    retriever = fit_catalog_retriever(bundle.items, config)
    # Group folds are the primary train/OOF path; B/C remain stress-evaluation manifests.
    manifest = manifests["group"]
    for fold in range(config.n_splits):
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        scope = build_fold_scope(
            manifest, bundle.training_pairs, fold, item_family_map=family_map
        )
        fold_summary: dict[str, Any] = {}
        for epoch in config.curriculum_epochs:
            negatives, suspicious = generate_fold_negatives(
                scope, bundle.training_pairs, bundle.terms, catalog,
                mining_config, epoch=epoch,
            )
            negative_path = fold_dir / f"epoch_{epoch}_negatives.parquet"
            suspicious_path = fold_dir / f"epoch_{epoch}_suspicious.parquet"
            negatives.to_parquet(negative_path, index=False)
            suspicious.to_parquet(suspicious_path, index=False)
            training = prepare_fold_training_frame(
                scope, bundle.training_pairs, negatives, bundle.terms, bundle.items
            )
            training_path = fold_dir / f"epoch_{epoch}_train.parquet"
            training.to_parquet(training_path, index=False)
            fold_summary[f"epoch_{epoch}"] = {
                "train_rows": len(training), "positive_rows": int(training["label"].sum()),
                "negative_rows": int(training["label"].eq(0).sum()),
                "suspicious_rows": len(suspicious),
                "negative_types": negatives["negative_type"].value_counts().to_dict(),
            }
        valid_terms = bundle.terms[
            bundle.terms["term_id"].astype(str).isin(scope.valid_term_ids)
        ][["term_id", "query"]]
        retrieval = retrieve_catalog_candidates(valid_terms, retriever)
        retrieval = augment_candidate_pool_to_profile(
            retrieval, sorted(scope.valid_term_ids), bundle.items["item_id"].astype(str).tolist(),
            profile, seed=config.seed,
        )
        retrieval.to_parquet(fold_dir / "validation_retrieval_pool.parquet", index=False)
        retrieval_metrics(retrieval, bundle.training_pairs).to_csv(
            fold_dir / "validation_retrieval_metrics.csv", index=False
        )
        for mode in ("natural", "oracle"):
            validation = build_fixed_validation_candidates(
                scope, retrieval, bundle.training_pairs, profile,
                mode=mode, seed=config.seed,
            )
            validation = enrich_pair_frame(
                validation, bundle.terms, bundle.items, product_view="long"
            )
            validation.to_parquet(fold_dir / f"validation_{mode}.parquet", index=False)
            fold_summary[f"validation_{mode}_rows"] = len(validation)
            fold_summary[f"validation_{mode}_positive_rate"] = float(validation["label"].mean())
        summary["folds"][str(fold)] = fold_summary
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def run_cpu_pipeline(
    data_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    *,
    stages: Sequence[str] = ("audit", "validation", "prepare"),
) -> dict[str, Any]:
    """Run selected resumable CPU preparation stages in dependency order."""

    config.validate()
    set_global_seed(config.seed)
    unknown = set(stages) - {"audit", "validation", "prepare"}
    if unknown:
        raise ValueError(f"Bilinmeyen pipeline stage: {sorted(unknown)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_config = AuditConfig(
        debug=config.debug, debug_items=config.debug_items,
        debug_train_pairs=config.debug_train_pairs,
        debug_submission_pairs=config.debug_submission_pairs,
        seed=config.seed,
    )
    bundle = load_data_bundle(data_dir, audit_config)
    results: dict[str, Any] = {"config": asdict(config), "pipeline_hash": _pipeline_hash(config)}
    if "audit" in stages:
        results["audit"] = run_dataset_audit(bundle, output_dir / "audit", audit_config)
    manifests: dict[str, SplitManifest]
    if "validation" in stages or "prepare" in stages:
        manifests = build_validation_manifests(bundle, output_dir / "validation", config)
        results["validation"] = {
            name: {"rows": len(manifest.frame), "source_hash": manifest.source_hash}
            for name, manifest in manifests.items()
        }
    else:
        manifests = {}
    if "prepare" in stages:
        results["prepare"] = prepare_model_artifacts(
            bundle, manifests, output_dir / "folds", config
        )
    (output_dir / "pipeline_run.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return results


def parse_args() -> argparse.Namespace:
    """Parse the stage-oriented preparation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3/debug_run"))
    parser.add_argument("--stages", nargs="+", choices=["audit", "validation", "prepare"], default=["audit", "validation", "prepare"])
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-items", type=int, default=5_000)
    parser.add_argument("--debug-train-pairs", type=int, default=5_000)
    parser.add_argument("--debug-submission-pairs", type=int, default=5_000)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--semantic-clusters", type=int, default=48)
    parser.add_argument("--embedding-backend", choices=["lexical", "qwen"], default="lexical")
    parser.add_argument("--negative-ratio", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--retrieval-top-k", type=int, default=300)
    parser.add_argument("--item-policy", choices=["audit", "purge_item", "purge_family"], default="audit")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the configured CPU pipeline and print its compact summary."""

    args = parse_args()
    config = PipelineConfig(
        debug=args.debug, debug_items=args.debug_items,
        debug_train_pairs=args.debug_train_pairs,
        debug_submission_pairs=args.debug_submission_pairs,
        n_splits=args.n_splits, semantic_clusters=args.semantic_clusters,
        embedding_backend=args.embedding_backend, negative_ratio=args.negative_ratio,
        retrieval_top_k=args.retrieval_top_k, item_policy=args.item_policy,
        seed=args.seed,
    )
    result = run_cpu_pipeline(args.data_dir, args.output_dir, config, stages=args.stages)
    print(json.dumps({
        "pipeline_hash": result["pipeline_hash"],
        "output_dir": str(args.output_dir),
        "stages": list(args.stages),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
