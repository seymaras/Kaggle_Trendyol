#!/usr/bin/env python3
"""Distil round-1 LLM votes into a full-test student and build a new judge pool.

This program never calls Kaggle.  It uses Qwen votes on the original real-test
disagreement pool, optionally strengthens them with Mistral votes, trains a
query-group validated CatBoost student on already-computed full-test features,
and uses that student only to *retrieve* high-value disagreements with the
LB-0.901 anchor.  A later independent LLM stage decides whether to flip them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


LADY = Path("/Users/seyma/2Kaggle_Trendyol/2lady-recall")
VAL = LADY / "data/val_probs"
PROCESSED = LADY / "data/processed"


BASE_SCORES = {
    "v2": (VAL / "v2true_submission_probs.parquet", "cross_encoder_prob"),
    "v4": (VAL / "submission_probs_v4randneg.parquet", "prob"),
    "intent_ens": (VAL / "submission_probs_intent2_ens.parquet", "prob"),
    "intent2": (VAL / "submission_probs_catboost_intent2.parquet", "prob"),
    "cat_emb": (VAL / "submission_probs_catboost_emb.parquet", "prob"),
    "bge_v2": (VAL / "submission_probs_bge_v2.parquet", "prob"),
    "bt128": (VAL / "submission_probs_bt128.parquet", "prob"),
    "qwen_reranker": (VAL / "submission_probs_qwen.parquet", "prob"),
    "pseudo": (VAL / "submission_probs_pseudo.parquet", "prob"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/submission_pairs.csv"))
    parser.add_argument("--terms", type=Path, default=Path("data/terms.csv"))
    parser.add_argument("--items", type=Path, default=Path("data/items.csv"))
    parser.add_argument(
        "--anchor", type=Path, default=Path("/Users/seyma/Downloads/llm_consensus_medium.csv")
    )
    parser.add_argument(
        "--teacher-pool",
        type=Path,
        default=Path("artifacts/llm_judge_v1/input/llm_judge_pool.parquet"),
    )
    parser.add_argument(
        "--qwen-votes",
        type=Path,
        default=Path("artifacts/llm_judge_v1/drive_votes/qwen"),
    )
    parser.add_argument(
        "--mistral-votes",
        type=Path,
        default=Path("artifacts/llm_judge_v1/drive_votes/mistral"),
    )
    parser.add_argument(
        "--round2-pool",
        type=Path,
        default=Path("artifacts/llm_judge_v2/input/llm_judge_pool.parquet"),
    )
    parser.add_argument(
        "--lexical-features",
        type=Path,
        default=PROCESSED / "features_submission.parquet",
    )
    parser.add_argument(
        "--embedding-features",
        type=Path,
        default=PROCESSED / "embedding_features_submission.parquet",
    )
    parser.add_argument(
        "--intent-features",
        type=Path,
        default=PROCESSED / "intent2_features_submission.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/llm_student_v1")
    )
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--target-positive-rate", type=float, default=0.25)
    parser.add_argument("--high-threshold", type=float, default=0.78)
    parser.add_argument("--low-threshold", type=float, default=0.22)
    parser.add_argument("--max-pool-rows", type=int, default=180_000)
    parser.add_argument("--attribute-chars", type=int, default=420)
    parser.add_argument("--predict-chunk-size", type=int, default=250_000)
    return parser.parse_args()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def align_values(
    target_ids: pd.Series, source: pd.DataFrame, value_col: str, source_name: str
) -> np.ndarray:
    if "id" not in source or value_col not in source:
        raise ValueError(f"{source_name}: id/{value_col} columns are required")
    source_ids = source["id"].astype("string")
    if source_ids.duplicated().any():
        raise ValueError(f"{source_name}: duplicate ids")
    if len(source) != len(target_ids):
        raise ValueError(f"{source_name}: row count mismatch")
    if np.array_equal(source_ids.to_numpy(), target_ids.to_numpy()):
        return source[value_col].to_numpy()
    positions = pd.Index(source_ids).get_indexer(target_ids)
    if (positions < 0).any():
        raise ValueError(f"{source_name}: ids do not match")
    return source[value_col].to_numpy()[positions]


def load_vote_parts(directory: Path, name: str) -> pd.DataFrame:
    parts = sorted(directory.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"{name}: no vote parts in {directory}")
    votes = pd.concat((pd.read_parquet(path) for path in parts), ignore_index=True)
    if votes["id"].duplicated().any():
        raise ValueError(f"{name}: duplicate vote ids")
    if not votes["label"].isin([0, 1]).all():
        raise ValueError(f"{name}: non-binary labels")
    return votes


def stable_sigmoid_logit_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    eps = np.float32(1e-5)
    first = np.clip(first, eps, 1 - eps)
    second = np.clip(second, eps, 1 - eps)
    logits = (np.log(first / (1 - first)) + np.log(second / (1 - second))) / 2
    return (1 / (1 + np.exp(-logits))).astype(np.float32)


def term_percentile(values: np.ndarray, term_codes: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values, copy=False)
        .groupby(term_codes, sort=False)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32, copy=False)
    )


def global_percentile(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values, copy=False)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32, copy=False)
    )


def count_preserving_top_k(
    scores: np.ndarray, term_codes: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, -scores, term_codes))
    sorted_codes = term_codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    ranks_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    positive_counts = (
        pd.Series(anchor, copy=False)
        .groupby(term_codes, sort=False)
        .sum()
        .to_numpy(dtype=np.int64, copy=False)
    )
    prediction = np.zeros(n, dtype=np.uint8)
    prediction[order] = (ranks_sorted < positive_counts[sorted_codes]).astype(np.uint8)
    boundary = np.empty(len(positive_counts), dtype=np.float32)
    sorted_scores = scores[order]
    for start, end, code in zip(starts, ends, sorted_codes[starts]):
        k = int(positive_counts[code])
        if k <= 0:
            boundary[code] = np.float32(sorted_scores[start] + 1e-6)
        elif k >= end - start:
            boundary[code] = np.float32(sorted_scores[end - 1] - 1e-6)
        else:
            boundary[code] = np.float32(
                (float(sorted_scores[start + k - 1]) + float(sorted_scores[start + k])) / 2
            )
    return prediction, boundary[term_codes]


def compact_attributes(values: pd.Series, max_chars: int) -> pd.Series:
    values = values.fillna("").astype("string").str.replace(r"\s+", " ", regex=True)
    head = int(math.ceil(max_chars * 0.68))
    tail = max_chars - head - 3
    long = values.str.len().fillna(0) > max_chars
    out = values.copy()
    out.loc[long] = values.loc[long].str.slice(0, head) + " … " + values.loc[long].str.slice(-tail)
    return out


def main() -> None:
    args = parse_args()
    required = [
        args.pairs,
        args.terms,
        args.items,
        args.anchor,
        args.teacher_pool,
        args.round2_pool,
        args.lexical_features,
        args.embedding_features,
        args.intent_features,
        *[path for path, _ in BASE_SCORES.values()],
    ]
    require_files(required)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Full-test IDs, terms and the LB-0.901 anchor are loading...")
    pairs = pd.read_csv(
        args.pairs,
        usecols=["id", "term_id", "item_id"],
        dtype={"id": "string", "term_id": "string", "item_id": "string"},
    )
    if pairs["id"].duplicated().any():
        raise ValueError("submission_pairs has duplicate ids")
    ids = pairs["id"]
    anchor_frame = pd.read_csv(args.anchor, dtype={"id": "string", "prediction": "uint8"})
    anchor = align_values(ids, anchor_frame, "prediction", "anchor").astype(np.uint8)
    if set(np.unique(anchor).tolist()) != {0, 1}:
        raise ValueError("anchor is not binary")
    term_codes, unique_terms = pd.factorize(pairs["term_id"], sort=False)
    term_codes = term_codes.astype(np.int32, copy=False)

    print("[2/8] Diverse full-test features are aligning...")
    features: dict[str, np.ndarray] = {}
    score_names: list[str] = []
    for name, (path, column) in BASE_SCORES.items():
        frame = pd.read_parquet(path, columns=["id", column])
        values = align_values(ids, frame, column, name).astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: NaN/inf")
        features[name] = values
        score_names.append(name)
        del frame
        print(f"  {name}: OK")

    lexical = pd.read_parquet(args.lexical_features)
    lexical_ids = lexical.pop("id").astype("string")
    if not np.array_equal(lexical_ids.to_numpy(), ids.to_numpy()):
        raise ValueError("lexical features are not row-aligned")
    for column in lexical.columns:
        features[f"lex_{column}"] = lexical[column].to_numpy(dtype=np.float32, copy=False)
    del lexical, lexical_ids

    embedding = pd.read_parquet(args.embedding_features)
    features["embedding_cosine"] = align_values(
        ids, embedding, "embedding_cosine", "embedding"
    ).astype(np.float32, copy=False)
    del embedding

    intent = pd.read_parquet(args.intent_features)
    intent_ids = intent.pop("id").astype("string")
    if not np.array_equal(intent_ids.to_numpy(), ids.to_numpy()):
        raise ValueError("intent features are not row-aligned")
    for column in intent.columns:
        values = intent[column].to_numpy(dtype=np.float32, copy=False)
        missing = ~np.isfinite(values)
        if missing.any():
            finite = values[~missing]
            fill_value = np.float32(np.median(finite)) if len(finite) else np.float32(0)
            values = values.copy()
            values[missing] = fill_value
            features[f"missing_{column}"] = missing.astype(np.float32)
        features[f"raw_{column}"] = values
    del intent, intent_ids

    core_rank = ["v2", "v4", "intent_ens", "bge_v2", "bt128", "qwen_reranker"]
    core_rank.append("embedding_cosine")
    for name in core_rank:
        print(f"  transductive ranks: {name}")
        features[f"global_rank_{name}"] = global_percentile(features[name])
        features[f"term_rank_{name}"] = term_percentile(features[name], term_codes)

    score_matrix = np.column_stack([features[name] for name in score_names])
    features["model_mean"] = score_matrix.mean(axis=1, dtype=np.float32)
    features["model_std"] = score_matrix.std(axis=1, dtype=np.float32)
    features["model_range"] = score_matrix.max(axis=1) - score_matrix.min(axis=1)
    del score_matrix
    term_sizes = np.bincount(term_codes).astype(np.float32)
    features["term_size_log"] = np.log1p(term_sizes[term_codes]).astype(np.float32)
    feature_names = list(features)
    matrix = np.column_stack([features[name] for name in feature_names]).astype(np.float32)
    if not np.isfinite(matrix).all():
        bad = [name for name, values in features.items() if not np.isfinite(values).all()]
        raise ValueError(f"feature NaN/inf: {bad}")
    del features
    print(f"  matrix: {matrix.shape[0]:,} x {matrix.shape[1]:,}")

    print("[3/8] Qwen/Mistral teacher probabilities are merging...")
    teacher_pool = pd.read_parquet(args.teacher_pool)
    positions = teacher_pool["row_position"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids.iloc[positions].to_numpy(), teacher_pool["id"].astype("string").to_numpy()):
        raise ValueError("teacher pool row positions do not align")
    qwen = load_vote_parts(args.qwen_votes, "qwen").set_index("id")
    qwen = qwen.loc[teacher_pool["id"].astype(str)]
    qwen_prob = qwen["p_relevant"].to_numpy(dtype=np.float32)
    qwen_label = qwen["label"].to_numpy(dtype=np.uint8)
    mistral = load_vote_parts(args.mistral_votes, "mistral").set_index("id")
    mistral_index = pd.Index(teacher_pool["id"].astype(str)).get_indexer(mistral.index)
    if (mistral_index < 0).any():
        raise ValueError("Mistral contains ids outside teacher pool")
    has_mistral = np.zeros(len(teacher_pool), dtype=bool)
    has_mistral[mistral_index] = True
    mistral_prob = np.full(len(teacher_pool), np.nan, dtype=np.float32)
    mistral_label = np.full(len(teacher_pool), -1, dtype=np.int8)
    mistral_prob[mistral_index] = mistral["p_relevant"].to_numpy(dtype=np.float32)
    mistral_label[mistral_index] = mistral["label"].to_numpy(dtype=np.int8)
    teacher_prob = qwen_prob.copy()
    teacher_prob[has_mistral] = stable_sigmoid_logit_mean(
        qwen_prob[has_mistral], mistral_prob[has_mistral]
    )
    teacher_label = (teacher_prob >= 0.5).astype(np.uint8)
    confidence = np.abs(teacher_prob - 0.5) * 2
    weights = (0.35 + 0.65 * confidence).astype(np.float32)
    agreement = has_mistral & (qwen_label == mistral_label)
    disagreement = has_mistral & (qwen_label != mistral_label)
    weights[agreement] *= 1.35
    weights[disagreement] *= 0.55

    print("[4/8] CatBoost student is training with a query-disjoint validation split...")
    pool_terms = teacher_pool["term_id"].astype("string")
    term_hash = pd.util.hash_pandas_object(pool_terms, index=False).to_numpy(dtype=np.uint64)
    valid_mask = term_hash % 5 == 0
    train_mask = ~valid_mask
    if pool_terms[train_mask].isin(set(pool_terms[valid_mask])).any():
        raise RuntimeError("query leakage in validation split")
    x_pool = matrix[positions]
    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.seed,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        od_type="Iter",
        od_wait=100,
        verbose=50,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(
        x_pool[train_mask],
        teacher_label[train_mask],
        sample_weight=weights[train_mask],
        eval_set=(x_pool[valid_mask], teacher_label[valid_mask]),
        use_best_model=True,
    )
    valid_prob = model.predict_proba(x_pool[valid_mask])[:, 1]
    valid_label = teacher_label[valid_mask]
    validation = {
        "rows": int(valid_mask.sum()),
        "terms": int(pool_terms[valid_mask].nunique()),
        "auc": float(roc_auc_score(valid_label, valid_prob)),
        "average_precision": float(average_precision_score(valid_label, valid_prob)),
        "f1_at_0_5": float(f1_score(valid_label, valid_prob >= 0.5)),
        "logloss": float(log_loss(valid_label, valid_prob)),
        "teacher_probability_correlation": float(
            np.corrcoef(teacher_prob[valid_mask], valid_prob)[0, 1]
        ),
        "best_iteration": int(model.get_best_iteration()),
    }
    model_path = args.output_dir / "llm_student_catboost.cbm"
    model.save_model(model_path)
    del x_pool
    print(json.dumps(validation, indent=2))

    print("[5/8] Student is scoring all 3.36M real test candidates...")
    student_prob = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), args.predict_chunk_size):
        end = min(start + args.predict_chunk_size, len(pairs))
        student_prob[start:end] = model.predict_proba(matrix[start:end])[:, 1]
        print(f"  {end:,}/{len(pairs):,}")
    student_scores_path = args.output_dir / "student_scores.parquet"
    pd.DataFrame({"id": ids, "student_prob": student_prob}).to_parquet(
        student_scores_path, index=False, compression="zstd"
    )

    print("[6/8] Independent disagreement routes are forming the new pool...")
    k = int(round(len(pairs) * args.target_positive_rate))
    global_prediction = np.zeros(len(pairs), dtype=np.uint8)
    global_chosen = np.argpartition(student_prob, len(pairs) - k)[len(pairs) - k :]
    global_prediction[global_chosen] = 1
    count_prediction, term_boundary = count_preserving_top_k(student_prob, term_codes, anchor)
    confident_prediction = np.full(len(pairs), 255, dtype=np.uint8)
    confident_prediction[student_prob >= args.high_threshold] = 1
    confident_prediction[student_prob <= args.low_threshold] = 0
    source_global = global_prediction != anchor
    source_count = count_prediction != anchor
    source_confident = (confident_prediction != 255) & (confident_prediction != anchor)
    source_round2 = np.zeros(len(pairs), dtype=bool)
    round2 = pd.read_parquet(args.round2_pool)
    round2_positions = round2["row_position"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids.iloc[round2_positions].to_numpy(), round2["id"].astype("string").to_numpy()):
        raise ValueError("round2 pool row positions do not align")
    source_round2[round2_positions] = True
    already_judged = np.zeros(len(pairs), dtype=bool)
    already_judged[positions] = True
    union = (source_global | source_count | source_confident | source_round2) & ~already_judged
    source_votes = (
        source_global.astype(np.uint8)
        + source_count.astype(np.uint8)
        + source_confident.astype(np.uint8)
        + source_round2.astype(np.uint8)
    )
    direction_confidence = np.abs(student_prob - 0.5) * 2
    boundary_margin = np.abs(student_prob - term_boundary)
    priority = (
        10.0 * source_votes
        + 15.0 * source_round2.astype(np.float32)
        + 2.0 * direction_confidence
        + matrix[:, feature_names.index("model_std")]
        + 0.5 * (matrix[:, feature_names.index("lex_word_cosine")] == 0)
        + boundary_margin
    ).astype(np.float32)
    selected = np.flatnonzero(union)
    if args.max_pool_rows > 0 and len(selected) > args.max_pool_rows:
        keep = np.argpartition(priority[selected], len(selected) - args.max_pool_rows)[
            len(selected) - args.max_pool_rows :
        ]
        selected = selected[keep]
    selected = selected[np.argsort(-priority[selected], kind="stable")]
    selected_ids = ids.iloc[selected].reset_index(drop=True)

    print("[7/8] Query/item text is attaching only to selected rows...")
    terms = pd.read_csv(args.terms, dtype="string")
    query_column = "term" if "term" in terms.columns else "query"
    term_lookup = terms.set_index("term_id")[query_column]
    items = pd.read_csv(args.items, dtype="string")
    item_lookup = items.set_index("item_id")
    selected_item_ids = pairs["item_id"].iloc[selected]
    selected_term_ids = pairs["term_id"].iloc[selected]
    selected_items = item_lookup.loc[selected_item_ids.to_numpy()].reset_index(drop=True)
    pool_out = pd.DataFrame(
        {
            "id": selected_ids,
            "term_id": selected_term_ids.reset_index(drop=True),
            "item_id": selected_item_ids.reset_index(drop=True),
            "row_position": selected.astype(np.int32),
            "query": term_lookup.loc[selected_term_ids.to_numpy()].reset_index(drop=True),
            "title": selected_items["title"],
            "category": selected_items["category"],
            "brand": selected_items["brand"],
            "gender": selected_items["gender"],
            "age_group": selected_items["age_group"],
            "attributes_compact": compact_attributes(
                selected_items["attributes"], args.attribute_chars
            ),
            "anchor_prediction": anchor[selected],
            "alternative_prediction": (1 - anchor[selected]).astype(np.uint8),
            "student_prob": student_prob[selected],
            "student_confidence": direction_confidence[selected],
            "term_boundary": term_boundary[selected],
            "boundary_margin": boundary_margin[selected],
            "source_global": source_global[selected],
            "source_count": source_count[selected],
            "source_confident": source_confident[selected],
            "source_round2": source_round2[selected],
            "source_votes": source_votes[selected],
            "model_std": matrix[selected, feature_names.index("model_std")],
            "word_cosine": matrix[selected, feature_names.index("lex_word_cosine")],
            "priority": priority[selected],
        }
    )
    pool_path = input_dir / "llm_judge_pool.parquet"
    pool_out.to_parquet(pool_path, index=False, compression="zstd")
    anchor_path = input_dir / "anchor_v6.parquet"
    pd.DataFrame({"id": ids, "prediction": anchor}).to_parquet(
        anchor_path, index=False, compression="zstd"
    )

    print("[8/8] Audit report and hashes are writing...")
    importance = sorted(
        zip(feature_names, model.get_feature_importance()), key=lambda pair: -pair[1]
    )
    report = {
        "schema_version": 1,
        "strategy": "LLM teacher -> query-disjoint CatBoost student -> independent LLM retrieval",
        "rows_total": int(len(pairs)),
        "terms_total": int(len(unique_terms)),
        "anchor_positive_count": int(anchor.sum()),
        "anchor_positive_rate": float(anchor.mean()),
        "teacher_pool_rows": int(len(teacher_pool)),
        "qwen_rows": int(len(qwen)),
        "mistral_rows": int(len(mistral)),
        "teacher_positive_rate": float(teacher_label.mean()),
        "teacher_model_agreement": float((qwen_label[has_mistral] == mistral_label[has_mistral]).mean()),
        "validation": validation,
        "feature_count": len(feature_names),
        "feature_importance": {name: float(value) for name, value in importance},
        "candidate_union_before_cap": int(union.sum()),
        "candidate_rows": int(len(pool_out)),
        "candidate_anchor_0": int((pool_out["anchor_prediction"] == 0).sum()),
        "candidate_anchor_1": int((pool_out["anchor_prediction"] == 1).sum()),
        "candidate_source_counts": {
            name: int(pool_out[name].sum())
            for name in ["source_global", "source_count", "source_confident", "source_round2"]
        },
        "parameters": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "target_positive_rate": args.target_positive_rate,
            "high_threshold": args.high_threshold,
            "low_threshold": args.low_threshold,
            "max_pool_rows": args.max_pool_rows,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in [pool_path, anchor_path, student_scores_path, model_path]
        },
        "kaggle_submission_called": False,
    }
    report_path = args.output_dir / "student_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    manifest_path = input_dir / "manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("=" * 80)
    print("STUDENT POOL READY — Kaggle submission call: NONE")
    print(json.dumps({
        "validation": validation,
        "candidate_rows": len(pool_out),
        "source_counts": report["candidate_source_counts"],
        "output": str(input_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
