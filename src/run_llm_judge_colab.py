#!/usr/bin/env python3
"""Run two or three open-weight relevance judges and build anchored CSVs.

This program never calls the Kaggle submission API.  The models run in
separate subprocesses so GPU memory is fully released between them.  Every
chunk is checkpointed to Google Drive and verified before it is reused.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    ("qwen3_30b_a3b", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"),
    ("mistral_small_24b", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"),
]

SYSTEM_PROMPT = (
    "E-ticaret arama alaka hakemisin. 1: ürün sorgunun istediği ürün türü ve "
    "zorunlu özelliklerle uyumlu; eşanlamlı veya doğrudan muadil olabilir. "
    "0: yanlış ürün türü, yalnız aksesuar/tamamlayıcı/yedek parça ya da marka, "
    "model, cinsiyet, yaş gibi açık bir koşulla çelişiyor. Yalnız 0 veya 1 yaz."
)

GPT_OSS_SYSTEM_PROMPT = "Reasoning: low\n" + SYSTEM_PROMPT
HARMONY_FINAL_PATTERN = re.compile(
    r"<\|channel\|>\s*final(?:<\|constrain\|>[^<]*)?"
    r"<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|$)",
    flags=re.DOTALL | re.IGNORECASE,
)

POOL_COLUMNS = [
    "id",
    "term_id",
    "row_position",
    "query",
    "title",
    "category",
    "brand",
    "gender",
    "age_group",
    "attributes_compact",
    "anchor_prediction",
    "alternative_prediction",
    "source_votes",
    "priority",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["all", "judge", "finalize", "self-test"], default="all"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Repeat as slug=HuggingFace/model. Defaults to Qwen + Mistral.",
    )
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--gpt-oss-max-tokens",
        type=int,
        default=192,
        help="Harmony reasoning + final answer budget used only by GPT-OSS.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--second-model-prefilter",
        type=float,
        default=0.55,
        help=(
            "The second model only sees rows where model 1 chose the alternative "
            "with at least this binary margin-confidence."
        ),
    )
    parser.add_argument(
        "--third-model-prefilter",
        type=float,
        default=0.65,
        help=(
            "With three models, model 3 only sees rows where models 1 and 2 "
            "both chose the alternative at this minimum confidence."
        ),
    )
    parser.add_argument(
        "--strict-threshold", type=float, default=0.80, help="Per-model minimum confidence."
    )
    parser.add_argument(
        "--medium-threshold", type=float, default=0.65, help="Per-model minimum confidence."
    )
    return parser.parse_args()


def model_specs(raw_specs: list[str] | None) -> list[tuple[str, str]]:
    if not raw_specs:
        return list(DEFAULT_MODELS)
    parsed: list[tuple[str, str]] = []
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"--model must be slug=repo, got: {spec}")
        slug, repo = spec.split("=", 1)
        if not re.fullmatch(r"[a-z0-9_]+", slug):
            raise ValueError(f"invalid model slug: {slug}")
        if not repo.strip():
            raise ValueError(f"empty model repository in: {spec}")
        parsed.append((slug, repo.strip()))
    if len(parsed) not in (2, 3):
        raise ValueError("exactly two or three independent models are required")
    if len({slug for slug, _ in parsed}) != len(parsed):
        raise ValueError("model slugs must be unique")
    return parsed


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_id_sequence(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_manifest(input_dir: Path, verify_hashes: bool = True) -> dict[str, Any]:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("llm_judge_pool.parquet", "anchor_v6.parquet"):
        path = input_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"input not found: {path}")
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if verify_hashes and expected and sha256_file(path) != expected:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
    return manifest


def build_prompt(row: Any) -> str:
    product_parts = [
        f"başlık={row.title}",
        f"kategori={row.category}",
        f"marka={row.brand}",
        f"cinsiyet={row.gender}",
        f"yaş={row.age_group}",
    ]
    if row.attributes_compact:
        product_parts.append(f"özellik={row.attributes_compact}")
    return f"SORGU: {row.query}\nÜRÜN: " + " | ".join(product_parts)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def read_valid_part(
    path: Path, expected_ids: np.ndarray, expected_signature: str
) -> bool:
    if not path.is_file():
        return False
    try:
        part = pd.read_parquet(
            path,
            columns=[
                "id",
                "label",
                "p_relevant",
                "confidence",
                "run_signature",
            ],
        )
        ids = part["id"].astype("string").to_numpy()
    except Exception:
        return False
    return (
        len(ids) == len(expected_ids)
        and np.array_equal(ids, expected_ids)
        and part["label"].isin([0, 1]).all()
        and np.isfinite(part["p_relevant"]).all()
        and np.isfinite(part["confidence"]).all()
        and part["p_relevant"].between(0.0, 1.0).all()
        and part["confidence"].between(0.5, 1.0).all()
        and part["run_signature"].astype("string").eq(expected_signature).all()
    )


def extract_logprob(value: Any) -> float:
    if value is None:
        return float("nan")
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def binary_probability(logprob_zero: float, logprob_one: float) -> float:
    if not (math.isfinite(logprob_zero) and math.isfinite(logprob_one)):
        return float("nan")
    pivot = max(logprob_zero, logprob_one)
    p0 = math.exp(logprob_zero - pivot)
    p1 = math.exp(logprob_one - pivot)
    return p1 / (p0 + p1)


def logsumexp(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    pivot = max(finite)
    return pivot + math.log(sum(math.exp(value - pivot) for value in finite))


def parse_gpt_oss_final_label(decoded_completion: str) -> int:
    """Extract a binary label only from GPT-OSS's Harmony final channel."""
    matches = HARMONY_FINAL_PATTERN.findall(decoded_completion)
    if not matches:
        raise ValueError("GPT-OSS completion has no Harmony final channel")
    final_text = matches[-1].strip()
    label_match = re.fullmatch(r"([01])(?:[.!])?", final_text)
    if label_match is None:
        raise ValueError(f"invalid GPT-OSS final answer: {final_text[:120]!r}")
    return int(label_match.group(1))


def cascade_selection_mask(
    args: argparse.Namespace,
    specs: list[tuple[str, str]],
    full_pool: pd.DataFrame,
    model_index: int,
) -> np.ndarray:
    """Return the exact adaptive row mask for a cascade stage."""
    if model_index == 0:
        return np.ones(len(full_pool), dtype=bool)
    pool_ids = full_pool["id"].astype("string").to_numpy()
    alternative = full_pool["alternative_prediction"].to_numpy(dtype=np.int8)
    print("Cascade prefilter: Qwen checkpoints are being read...", flush=True)
    first_votes = load_votes(pool_ids, args.work_dir / "votes" / specs[0][0])
    first_labels = first_votes["label"].to_numpy(dtype=np.int8)
    first_confidence = first_votes["confidence"].to_numpy(dtype=np.float32)
    second_mask = (first_labels == alternative) & (
        first_confidence >= args.second_model_prefilter
    )
    if model_index == 1:
        return second_mask
    if model_index != 2 or len(specs) != 3:
        raise ValueError("invalid cascade stage")
    print("Cascade prefilter: Mistral checkpoints are being read...", flush=True)
    second_votes = load_votes(
        pool_ids[second_mask], args.work_dir / "votes" / specs[1][0]
    )
    second_labels = second_votes["label"].to_numpy(dtype=np.int8)
    second_confidence = second_votes["confidence"].to_numpy(dtype=np.float32)
    third_within_second = (second_labels == alternative[second_mask]) & (
        second_confidence >= args.third_model_prefilter
    )
    third_mask = np.zeros(len(full_pool), dtype=bool)
    third_mask[np.flatnonzero(second_mask)[third_within_second]] = True
    return third_mask


def judge_model(args: argparse.Namespace, specs: list[tuple[str, str]]) -> None:
    venv_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("ninja") is None:
        raise RuntimeError(
            "ninja executable not found. Install it inside the vLLM environment."
        )
    if not 0 <= args.model_index < len(specs):
        raise ValueError("--model-index is out of range")
    slug, repo = specs[args.model_index]
    print(
        f"Judge worker started | model={args.model_index + 1}/{len(specs)} | repo={repo}",
        flush=True,
    )
    print("Input bundle and hashes are being verified...", flush=True)
    input_manifest = load_manifest(args.input_dir, verify_hashes=True)
    pool_path = args.input_dir / "llm_judge_pool.parquet"
    full_pool = pd.read_parquet(pool_path, columns=POOL_COLUMNS)
    if full_pool["id"].duplicated().any():
        raise ValueError("pool contains duplicate ids")
    pool = full_pool
    if args.model_index > 0:
        selection_mask = cascade_selection_mask(args, specs, full_pool, args.model_index)
        pool = full_pool.loc[selection_mask].reset_index(drop=True)
        if pool.empty:
            raise RuntimeError(
                f"cascade prefilter selected zero rows for model {args.model_index + 1}"
            )
        print(
            f"Adaptive model-{args.model_index + 1} prefilter: "
            f"{len(pool):,}/{len(full_pool):,} "
            f"({len(pool) / len(full_pool):.1%})",
            flush=True,
        )

    vote_dir = args.work_dir / "votes" / slug
    vote_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80, flush=True)
    print(f"MODEL {args.model_index + 1}/{len(specs)}: {repo}", flush=True)
    print(
        f"Rows: {len(pool):,} | chunk: {args.chunk_size:,} | checkpoint: {vote_dir}",
        flush=True,
    )

    prompt_sha256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    pool_id_sha256 = hash_id_sequence(pool["id"].astype("string"))
    vllm_version = importlib.metadata.version("vllm")
    base_signature_payload = {
        "repository": repo,
        "vllm_version": vllm_version,
        "input_pool_sha256": input_manifest["files"]["llm_judge_pool.parquet"][
            "sha256"
        ],
        "judged_id_sequence_sha256": pool_id_sha256,
        "system_prompt_sha256": prompt_sha256,
        "model_index": args.model_index,
        "second_model_prefilter": (
            args.second_model_prefilter if args.model_index == 1 else None
        ),
        "third_model_prefilter": (
            args.third_model_prefilter if args.model_index == 2 else None
        ),
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "temperature": 0.0,
        "inference_protocol": (
            "harmony_final_v2" if "gpt-oss" in repo.lower() else "binary_logprobs_v1"
        ),
        "max_tokens": (
            args.gpt_oss_max_tokens if "gpt-oss" in repo.lower() else 1
        ),
    }

    chunks = math.ceil(len(pool) / args.chunk_size)
    expected_part_names = {f"part_{index:05d}.parquet" for index in range(chunks)}
    for stale_part in vote_dir.glob("part_*.parquet"):
        if stale_part.name not in expected_part_names:
            stale_part.unlink()

    config_path = vote_dir / "run_config.json"
    summary_path = vote_dir / "model_run.json"
    saved_config: dict[str, Any] | None = None
    for candidate_path in (config_path, summary_path):
        if candidate_path.is_file():
            try:
                saved_config = json.loads(candidate_path.read_text(encoding="utf-8"))
                break
            except (OSError, json.JSONDecodeError):
                saved_config = None
    saved_payload = (saved_config or {}).get("run_signature_payload", {})
    saved_invariants_match = all(
        saved_payload.get(key) == value for key, value in base_signature_payload.items()
    )

    if saved_invariants_match and not args.force:
        saved_signature = str(saved_config.get("run_signature", ""))
        all_parts_valid = bool(saved_signature)
        for chunk_index, start in enumerate(range(0, len(pool), args.chunk_size)):
            end = min(start + args.chunk_size, len(pool))
            part_path = vote_dir / f"part_{chunk_index:05d}.parquet"
            expected_ids = pool["id"].iloc[start:end].astype("string").to_numpy()
            if not read_valid_part(part_path, expected_ids, saved_signature):
                all_parts_valid = False
                break
        if all_parts_valid and summary_path.is_file():
            print(f"{slug}: all checkpoints complete; skipped before loading model weights")
            return
        model_revision = str(saved_payload["model_revision"])
        print(f"Resume pins saved model revision: {model_revision}")
    else:
        from huggingface_hub import model_info  # type: ignore

        print("Hugging Face model revision is being resolved...", flush=True)
        model_revision = str(model_info(repo).sha)
        print(f"Pinned model revision: {model_revision}", flush=True)

    # Import only inside the model subprocess.  Exiting this process releases
    # all CUDA allocations before the second model starts.
    from vllm import LLM, SamplingParams  # type: ignore

    is_mistral = "mistral" in repo.lower()
    is_gpt_oss = "gpt-oss" in repo.lower()
    llm_kwargs: dict[str, Any] = {
        "model": repo,
        "revision": model_revision,
        "tokenizer_revision": model_revision,
        "trust_remote_code": False,
        "dtype": "bfloat16" if is_mistral else "auto",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": min(args.max_num_seqs, 128) if is_mistral else args.max_num_seqs,
        "max_num_batched_tokens": 16_384 if is_mistral else 32_768,
        "seed": args.seed,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "generation_config": "vllm",
        "disable_log_stats": True,
    }
    if is_mistral:
        llm_kwargs.update(
            {
                "tokenizer_mode": "mistral",
                "config_format": "mistral",
                "load_format": "mistral",
                "mm_processor_cache_gb": 0,
            }
        )
    print(
        "Model weights are loading into the GPU. A first download can take several minutes...",
        flush=True,
    )
    llm = LLM(**llm_kwargs)
    print("Model loaded; tokenizer and sampling configuration are being prepared...", flush=True)
    tokenizer = llm.get_tokenizer()

    token_ids_by_label: dict[int, set[int]] = {0: set(), 1: set()}
    token_map: dict[int, int] = {}
    if is_gpt_oss:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=args.gpt_oss_max_tokens,
            min_tokens=1,
            detokenize=True,
            seed=args.seed,
        )
    else:
        for label in (0, 1):
            for variant in (str(label), f" {label}"):
                token_ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(token_ids) == 1 and tokenizer.decode(token_ids).strip() == str(label):
                    token_ids_by_label[label].add(int(token_ids[0]))
            if not token_ids_by_label[label]:
                raise RuntimeError(
                    f"{repo}: label {label} has no one-token variant; "
                    "refusing an unconstrained run"
                )
        overlap = token_ids_by_label[0] & token_ids_by_label[1]
        if overlap:
            raise RuntimeError(f"{repo}: 0/1 token sets overlap: {sorted(overlap)}")
        token_map = {
            token_id: label
            for label, token_ids in token_ids_by_label.items()
            for token_id in token_ids
        }

        sampling_signature = inspect.signature(SamplingParams)
        required_sampling_parameters = {"allowed_token_ids", "logprob_token_ids"}
        missing_sampling_parameters = required_sampling_parameters - set(
            sampling_signature.parameters
        )
        if missing_sampling_parameters:
            raise RuntimeError(
                "Installed vLLM lacks required constrained-logprob parameters: "
                f"{sorted(missing_sampling_parameters)}. Install vLLM 0.24.0."
            )
        allowed_ids = list(token_map)
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=1,
            min_tokens=1,
            logprobs=max(2, len(allowed_ids)),
            allowed_token_ids=allowed_ids,
            logprob_token_ids=allowed_ids,
            detokenize=False,
            seed=args.seed,
        )

    run_signature_payload = {
        **base_signature_payload,
        "model_revision": model_revision,
        "label_token_ids": {
            str(label): sorted(token_ids)
            for label, token_ids in token_ids_by_label.items()
        },
    }
    run_signature = hashlib.sha256(
        json.dumps(
            run_signature_payload, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    config_path.write_text(
        json.dumps(
            {
                "run_signature": run_signature,
                "run_signature_payload": run_signature_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if is_gpt_oss:
        print("GPT-OSS Harmony pilot is starting (up to 64 rows)...", flush=True)
        pilot_parts = []
        for alternative_label in (0, 1):
            candidates = pool[pool["alternative_prediction"] == alternative_label]
            if candidates.empty:
                raise RuntimeError(
                    f"GPT-OSS pilot has no alternative={alternative_label} rows"
                )
            pilot_parts.append(candidates.head(min(32, len(candidates))))
        pilot = pd.concat(pilot_parts, ignore_index=True)
        pilot_conversations = [
            [
                {"role": "system", "content": GPT_OSS_SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(row)},
            ]
            for row in pilot.itertuples(index=False)
        ]
        pilot_outputs = llm.chat(
            pilot_conversations,
            sampling_params=sampling,
            use_tqdm=False,
            chat_template_content_format="string",
        )
        pilot_labels = []
        for request_output in pilot_outputs:
            completion = request_output.outputs[0]
            decoded = tokenizer.decode(
                completion.token_ids, skip_special_tokens=False
            )
            pilot_labels.append(parse_gpt_oss_final_label(decoded))
        if len(set(pilot_labels)) < 2:
            raise RuntimeError(
                "GPT-OSS Harmony pilot collapsed to one class; refusing full run"
            )
        print(
            "GPT-OSS Harmony pilot OK | "
            f"rows={len(pilot_labels)} | positive={np.mean(pilot_labels):.3f}",
            flush=True,
        )

    for chunk_index, start in enumerate(range(0, len(pool), args.chunk_size)):
        end = min(start + args.chunk_size, len(pool))
        part_path = vote_dir / f"part_{chunk_index:05d}.parquet"
        expected_ids = pool["id"].iloc[start:end].astype("string").to_numpy()
        if not args.force and read_valid_part(
            part_path, expected_ids, run_signature
        ):
            print(f"[{chunk_index + 1}/{chunks}] checkpoint OK; skipping", flush=True)
            continue

        block = pool.iloc[start:end]
        conversations: list[list[dict[str, str]]] = []
        for row in block.itertuples(index=False):
            conversations.append(
                [
                    {
                        "role": "system",
                        "content": GPT_OSS_SYSTEM_PROMPT if is_gpt_oss else SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": build_prompt(row)},
                ]
            )

        outputs = llm.chat(
            conversations,
            sampling_params=sampling,
            use_tqdm=True,
            chat_template_content_format="string",
        )
        labels: list[int] = []
        p_relevant: list[float] = []
        chosen_confidence: list[float] = []
        margins: list[float] = []
        prompt_tokens: list[int] = []
        raw_text: list[str] = []
        block_anchor = block["anchor_prediction"].to_numpy(dtype=np.int8)
        harmony_fallbacks = 0
        for row_offset, request_output in enumerate(outputs):
            completion = request_output.outputs[0]
            if is_gpt_oss:
                decoded = tokenizer.decode(
                    completion.token_ids, skip_special_tokens=False
                )
                try:
                    label = parse_gpt_oss_final_label(decoded)
                    confidence = 1.0
                    lp0, lp1 = ((0.0, 50.0) if label == 1 else (50.0, 0.0))
                except ValueError:
                    # Missing/non-binary Harmony final is an abstention. Keep
                    # the anchor-side label so this row cannot confirm a flip,
                    # while allowing the rest of the deterministic chunk to
                    # finish and be audited.
                    label = int(block_anchor[row_offset])
                    confidence = 0.5
                    lp0, lp1 = (0.0, 0.0)
                    harmony_fallbacks += 1
                generated_label = label
                probability = float(label)
                raw_completion = decoded[-1000:]
            else:
                token_id = int(completion.token_ids[0]) if completion.token_ids else -1
                generated_label = token_map.get(token_id, -1)
                step_logprobs = completion.logprobs[0] if completion.logprobs else {}
                lp0 = logsumexp(
                    extract_logprob(step_logprobs.get(candidate_id))
                    for candidate_id in token_ids_by_label[0]
                )
                lp1 = logsumexp(
                    extract_logprob(step_logprobs.get(candidate_id))
                    for candidate_id in token_ids_by_label[1]
                )
                probability = binary_probability(lp0, lp1)
                label = int(probability >= 0.5) if math.isfinite(probability) else -1
                confidence = (
                    max(probability, 1.0 - probability)
                    if math.isfinite(probability)
                    else float("nan")
                )
                raw_completion = completion.text
            labels.append(label)
            p_relevant.append(probability)
            chosen_confidence.append(confidence)
            margins.append(abs(lp1 - lp0) if math.isfinite(lp0) and math.isfinite(lp1) else float("nan"))
            prompt_tokens.append(len(request_output.prompt_token_ids or []))
            raw_text.append(raw_completion)
            if generated_label < 0:
                raise RuntimeError(f"{repo}: generated a token outside the 0/1 set")

        if is_gpt_oss and harmony_fallbacks:
            print(
                f"  Harmony fallback (abstain->anchor) on "
                f"{harmony_fallbacks}/{len(outputs)} rows"
            )
            if chunk_index == 0 and harmony_fallbacks > 0.2 * len(outputs):
                raise RuntimeError(
                    "GPT-OSS Harmony final channel missing/non-binary on "
                    f"{harmony_fallbacks}/{len(outputs)} of the first chunk (>20%); "
                    "aborting before the full run."
                )

        part = pd.DataFrame(
            {
                "id": expected_ids,
                "label": np.asarray(labels, dtype=np.int8),
                "p_relevant": np.asarray(p_relevant, dtype=np.float32),
                "confidence": np.asarray(chosen_confidence, dtype=np.float32),
                "margin": np.asarray(margins, dtype=np.float32),
                "prompt_tokens": np.asarray(prompt_tokens, dtype=np.int16),
                "raw_text": pd.Series(raw_text, dtype="string"),
                "run_signature": pd.Series(
                    [run_signature] * len(expected_ids), dtype="string"
                ),
            }
        )
        if (
            (part["label"] < 0).any()
            or not np.isfinite(part["p_relevant"]).all()
            or not np.isfinite(part["margin"]).all()
        ):
            raise RuntimeError(f"{repo}: invalid constrained output in chunk {chunk_index}")
        if chunk_index == 0 and part["label"].nunique() < 2:
            print(f"WARNING: {repo}: first chunk produced only one class")
        atomic_parquet(part, part_path)
        print(
            f"[{chunk_index + 1}/{chunks}] saved {start:,}:{end:,} | "
            f"positive={part['label'].mean():.3f} | confidence={part['confidence'].mean():.3f}",
            flush=True,
        )

    summary = {
        "slug": slug,
        "repository": repo,
        "rows": int(len(pool)),
        "full_pool_rows": int(len(full_pool)),
        "second_model_prefilter": (
            args.second_model_prefilter if args.model_index == 1 else None
        ),
        "third_model_prefilter": (
            args.third_model_prefilter if args.model_index == 2 else None
        ),
        "chunks": chunks,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "model_revision": model_revision,
        "vllm_version": vllm_version,
        "prompt_sha256": prompt_sha256,
        "judged_id_sequence_sha256": pool_id_sha256,
        "run_signature": run_signature,
        "run_signature_payload": run_signature_payload,
        "label_token_ids": {
            str(label): sorted(token_ids) for label, token_ids in token_ids_by_label.items()
        },
        "system_prompt": SYSTEM_PROMPT,
    }
    (vote_dir / "model_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_votes(pool_ids: np.ndarray, vote_dir: Path) -> pd.DataFrame:
    parts = sorted(vote_dir.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no model checkpoints found in {vote_dir}")
    frame = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "id",
                    "label",
                    "p_relevant",
                    "confidence",
                    "run_signature",
                ],
            )
            for path in parts
        ],
        ignore_index=True,
    )
    ids = frame["id"].astype("string").to_numpy()
    if len(ids) != len(pool_ids) or not np.array_equal(ids, pool_ids):
        raise RuntimeError(f"model checkpoints do not exactly cover the pool: {vote_dir}")
    if (frame["label"] < 0).any() or not np.isfinite(frame["p_relevant"]).all():
        raise RuntimeError(f"invalid model decisions: {vote_dir}")
    signatures = frame["run_signature"].astype("string").dropna().unique().tolist()
    run_path = vote_dir / "model_run.json"
    if len(signatures) != 1 or not run_path.is_file():
        raise RuntimeError(f"missing or inconsistent run signature: {vote_dir}")
    run_summary = json.loads(run_path.read_text(encoding="utf-8"))
    if signatures[0] != run_summary.get("run_signature"):
        raise RuntimeError(f"checkpoint signature does not match model_run.json: {vote_dir}")
    return frame


def count_preserving_flips(
    pool: pd.DataFrame, eligible: np.ndarray, confidence: np.ndarray
) -> np.ndarray:
    flips = np.zeros(len(pool), dtype=bool)
    working = pd.DataFrame(
        {
            "pool_index": np.arange(len(pool), dtype=np.int32),
            "term_id": pool["term_id"].astype("string"),
            "anchor": pool["anchor_prediction"].to_numpy(dtype=np.uint8),
            "confidence": confidence.astype(np.float32, copy=False),
            "source_votes": pool["source_votes"].to_numpy(dtype=np.uint8),
            "priority": pool["priority"].to_numpy(dtype=np.float32),
            "eligible": eligible,
        }
    )
    working = working[working["eligible"]]
    if working.empty:
        return flips

    for _, group in working.groupby("term_id", sort=False):
        additions = group[group["anchor"] == 0]
        removals = group[group["anchor"] == 1]
        swap_count = min(len(additions), len(removals))
        if swap_count == 0:
            continue
        sort_columns = ["confidence", "source_votes", "priority"]
        additions = additions.sort_values(sort_columns, ascending=False, kind="stable")
        removals = removals.sort_values(sort_columns, ascending=False, kind="stable")
        flips[additions["pool_index"].iloc[:swap_count].to_numpy()] = True
        flips[removals["pool_index"].iloc[:swap_count].to_numpy()] = True
    return flips


def write_candidate(
    name: str,
    anchor: pd.DataFrame,
    pool: pd.DataFrame,
    flip_mask: np.ndarray,
    output_dir: Path,
    count_preserving: bool,
) -> dict[str, Any]:
    predictions = anchor["prediction"].to_numpy(dtype=np.uint8, copy=True)
    positions = pool["row_position"].to_numpy(dtype=np.int64)
    pool_ids = pool["id"].astype("string").to_numpy()
    anchor_ids = anchor["id"].astype("string").to_numpy()
    if not np.array_equal(anchor_ids[positions], pool_ids):
        raise RuntimeError("pool row positions do not align with the anchor")

    selected_positions = positions[flip_mask]
    predictions[selected_positions] = pool.loc[
        flip_mask, "alternative_prediction"
    ].to_numpy(dtype=np.uint8)
    if set(np.unique(predictions).tolist()) != {0, 1}:
        raise RuntimeError(f"{name}: predictions are not binary")

    if count_preserving and flip_mask.any():
        delta = (
            pool.loc[flip_mask, ["term_id", "anchor_prediction", "alternative_prediction"]]
            .assign(
                delta=lambda x: x["alternative_prediction"].astype(int)
                - x["anchor_prediction"].astype(int)
            )
            .groupby("term_id", sort=False)["delta"]
            .sum()
        )
        if not (delta == 0).all():
            raise RuntimeError(f"{name}: per-query positive counts changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.csv"
    temporary_path = output_path.with_suffix(".csv.tmp")
    pd.DataFrame({"id": anchor_ids, "prediction": predictions}).to_csv(
        temporary_path, index=False
    )
    os.replace(temporary_path, output_path)
    flipped_anchor = pool.loc[flip_mask, "anchor_prediction"].to_numpy(dtype=np.uint8)
    report = {
        "file": output_path.name,
        "rows": int(len(predictions)),
        "flips": int(flip_mask.sum()),
        "zero_to_one": int((flipped_anchor == 0).sum()),
        "one_to_zero": int((flipped_anchor == 1).sum()),
        "positive_count": int(predictions.sum()),
        "positive_rate": float(predictions.mean()),
        "count_preserving_per_query": count_preserving,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }
    return report


def finalize(args: argparse.Namespace, specs: list[tuple[str, str]]) -> None:
    manifest = load_manifest(args.input_dir, verify_hashes=True)
    pool = pd.read_parquet(args.input_dir / "llm_judge_pool.parquet")
    anchor = pd.read_parquet(args.input_dir / "anchor_v6.parquet")
    pool_ids = pool["id"].astype("string").to_numpy()
    first_votes = load_votes(pool_ids, args.work_dir / "votes" / specs[0][0])
    labels_0 = first_votes["label"].to_numpy(dtype=np.int8)
    confidence_0 = first_votes["confidence"].to_numpy(dtype=np.float32)
    alternative = pool["alternative_prediction"].to_numpy(dtype=np.int8)
    second_mask = (labels_0 == alternative) & (
        confidence_0 >= args.second_model_prefilter
    )
    second_votes = load_votes(
        pool_ids[second_mask], args.work_dir / "votes" / specs[1][0]
    )
    labels_1 = np.full(len(pool), -1, dtype=np.int8)
    confidence_1 = np.zeros(len(pool), dtype=np.float32)
    p1_1_full = np.full(len(pool), np.nan, dtype=np.float32)
    labels_1[second_mask] = second_votes["label"].to_numpy(dtype=np.int8)
    confidence_1[second_mask] = second_votes["confidence"].to_numpy(dtype=np.float32)
    p1_1_full[second_mask] = second_votes["p_relevant"].to_numpy(dtype=np.float32)
    joint_confidence = np.minimum(confidence_0, confidence_1)
    consensus_alternative = (labels_0 == alternative) & (labels_1 == alternative)

    labels_2: np.ndarray | None = None
    confidence_2: np.ndarray | None = None
    p1_2_full: np.ndarray | None = None
    third_mask: np.ndarray | None = None
    triple_alternative: np.ndarray | None = None
    triple_confidence: np.ndarray | None = None
    if len(specs) == 3:
        third_mask = consensus_alternative & (
            confidence_1 >= args.third_model_prefilter
        )
        third_votes = load_votes(
            pool_ids[third_mask], args.work_dir / "votes" / specs[2][0]
        )
        labels_2 = np.full(len(pool), -1, dtype=np.int8)
        confidence_2 = np.zeros(len(pool), dtype=np.float32)
        p1_2_full = np.full(len(pool), np.nan, dtype=np.float32)
        labels_2[third_mask] = third_votes["label"].to_numpy(dtype=np.int8)
        confidence_2[third_mask] = third_votes["confidence"].to_numpy(dtype=np.float32)
        p1_2_full[third_mask] = third_votes["p_relevant"].to_numpy(dtype=np.float32)
        triple_alternative = consensus_alternative & (labels_2 == alternative)
        triple_confidence = np.minimum(joint_confidence, confidence_2)

    thresholds = {
        "strict": float(args.strict_threshold),
        "medium": float(args.medium_threshold),
        "broad": 0.50,
    }
    output_dir = args.work_dir / "output"
    candidate_reports: dict[str, Any] = {}
    for level, threshold in thresholds.items():
        eligible = consensus_alternative & (joint_confidence >= threshold)
        free_name = (
            f"llm_qwen_mistral_{level}"
            if len(specs) == 3
            else f"llm_consensus_{level}"
        )
        candidate_reports[free_name] = write_candidate(
            free_name,
            anchor,
            pool,
            eligible,
            output_dir,
            count_preserving=False,
        )
        if level != "broad" and len(specs) == 2:
            count_mask = count_preserving_flips(pool, eligible, joint_confidence)
            count_name = f"llm_count_preserving_{level}"
            candidate_reports[count_name] = write_candidate(
                count_name,
                anchor,
                pool,
                count_mask,
                output_dir,
                count_preserving=True,
            )
        if len(specs) == 3:
            assert triple_alternative is not None and triple_confidence is not None
            triple_eligible = triple_alternative & (triple_confidence >= threshold)
            triple_name = f"llm_triple_consensus_{level}"
            candidate_reports[triple_name] = write_candidate(
                triple_name,
                anchor,
                pool,
                triple_eligible,
                output_dir,
                count_preserving=False,
            )
            if level != "broad":
                triple_count = count_preserving_flips(
                    pool, triple_eligible, triple_confidence
                )
                triple_count_name = f"llm_triple_count_preserving_{level}"
                candidate_reports[triple_count_name] = write_candidate(
                    triple_count_name,
                    anchor,
                    pool,
                    triple_count,
                    output_dir,
                    count_preserving=True,
                )

    p1_0 = first_votes["p_relevant"].to_numpy(dtype=np.float32)
    p1_1 = p1_1_full[second_mask]
    model_rows = [int(len(pool)), int(second_mask.sum())]
    positive_rates = [float(labels_0.mean()), float(labels_1[second_mask].mean())]
    mean_probabilities = [float(p1_0.mean()), float(p1_1.mean())]
    if len(specs) == 3:
        assert third_mask is not None and labels_2 is not None and p1_2_full is not None
        model_rows.append(int(third_mask.sum()))
        positive_rates.append(float(labels_2[third_mask].mean()))
        mean_probabilities.append(float(p1_2_full[third_mask].mean()))
    report = {
        "schema_version": 1,
        "input_manifest": manifest,
        "models": [
            {"slug": slug, "repository": repo} for slug, repo in specs
        ],
        "pool_rows": int(len(pool)),
        "model_rows_judged": model_rows,
        "second_model_prefilter": args.second_model_prefilter,
        "third_model_prefilter": args.third_model_prefilter if len(specs) == 3 else None,
        "model_positive_rates": positive_rates,
        "model_label_agreement_on_second_stage": float(
            (labels_0[second_mask] == labels_1[second_mask]).mean()
        ),
        "both_models_choose_alternative": int(consensus_alternative.sum()),
        "all_three_choose_alternative": (
            int(triple_alternative.sum()) if triple_alternative is not None else None
        ),
        "mean_relevance_probabilities": mean_probabilities,
        "mean_joint_confidence": float(joint_confidence[second_mask].mean()),
        "thresholds": thresholds,
        "candidates": candidate_reports,
        "kaggle_submission_called": False,
    }
    report_path = output_dir / "llm_judge_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("=" * 80)
    print("TAMAMLANDI — Kaggle submission çağrısı yapılmadı")
    print(f"Çıktı klasörü: {output_dir}")
    print(json.dumps(candidate_reports, ensure_ascii=False, indent=2))


def self_test(args: argparse.Namespace) -> None:
    # Pure finalisation primitive test: two terms, balanced swaps, deterministic
    # preference by confidence.  No model or GPU is needed.
    pool = pd.DataFrame(
        {
            "term_id": ["a", "a", "a", "a", "b", "b"],
            "anchor_prediction": [0, 0, 1, 1, 0, 1],
            "source_votes": [2, 1, 2, 1, 2, 2],
            "priority": [21, 11, 22, 12, 20, 20],
        }
    )
    eligible = np.ones(len(pool), dtype=bool)
    confidence = np.array([0.9, 0.8, 0.95, 0.7, 0.9, 0.9], dtype=np.float32)
    flips = count_preserving_flips(pool, eligible, confidence)
    assert flips.tolist() == [True, True, True, True, True, True]
    delta = np.where(pool["anchor_prediction"].to_numpy() == 0, 1, -1)
    assert (
        pd.Series(delta[flips]).groupby(pool.loc[flips, "term_id"].to_numpy()).sum() == 0
    ).all()
    print("self-test: OK")


def child_command(
    args: argparse.Namespace, specs: list[tuple[str, str]], stage: str, model_index: int = 0
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--input-dir",
        str(args.input_dir),
        "--work-dir",
        str(args.work_dir),
        "--model-index",
        str(model_index),
        "--chunk-size",
        str(args.chunk_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpt-oss-max-tokens",
        str(args.gpt_oss_max_tokens),
        "--seed",
        str(args.seed),
        "--second-model-prefilter",
        str(args.second_model_prefilter),
        "--third-model-prefilter",
        str(args.third_model_prefilter),
        "--strict-threshold",
        str(args.strict_threshold),
        "--medium-threshold",
        str(args.medium_threshold),
    ]
    for slug, repo in specs:
        command.extend(["--model", f"{slug}={repo}"])
    if args.force:
        command.append("--force")
    return command


def main() -> None:
    args = parse_args()
    specs = model_specs(args.model)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        load_manifest(args.input_dir, verify_hashes=True)
        for index in range(len(specs)):
            subprocess.run(child_command(args, specs, "judge", index), check=True)
        subprocess.run(child_command(args, specs, "finalize"), check=True)
    elif args.stage == "judge":
        judge_model(args, specs)
    elif args.stage == "finalize":
        finalize(args, specs)
    else:
        self_test(args)


if __name__ == "__main__":
    main()
