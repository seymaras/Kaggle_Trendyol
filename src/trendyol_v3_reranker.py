#!/usr/bin/env python3
"""Qwen3 yes/no and generic one-logit reranker training plus shard inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from trendyol_v3_core import normalize_category, normalize_text, set_global_seed
from trendyol_v3_validation import classification_report_dict, sigmoid, validate_oof_frame


Architecture = Literal["qwen_causal", "sequence_classifier"]
LossKind = Literal["bce", "focal"]

QWEN_SYSTEM_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
QWEN_ASSISTANT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
DEFAULT_RERANK_INSTRUCTION = (
    "Given a Turkish e-commerce search query, determine whether the product satisfies the "
    "shopper's intended product type and requested attributes. Treat a wrong product type, "
    "brand, model, gender, age group, size, color, or capacity as not relevant."
)
CONTROL_TOKEN_PATTERN = r"<\|(?:im_start|im_end|endoftext)\|>"


@dataclass(frozen=True)
class RerankerConfig:
    """Model loading, serialization and adapter configuration."""

    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    architecture: Architecture = "qwen_causal"
    revision: str | None = None
    trust_remote_code: bool = False
    initialize_sequence_head: bool = False
    max_length: int = 256
    query_max_tokens: int = 48
    product_view: Literal["short", "long", "short_ascii", "long_ascii"] = "long"
    instruction: str = DEFAULT_RERANK_INSTRUCTION
    use_lora: bool = True
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    attention: Literal["auto", "sdpa", "flash_attention_2", "eager"] = "auto"
    dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"

    def validate(self) -> None:
        """Validate model and sequence settings."""

        if not self.model_name:
            raise ValueError("model_name boş olamaz")
        if self.max_length < 64 or self.query_max_tokens < 8 or self.query_max_tokens >= self.max_length:
            raise ValueError("max_length/query_max_tokens geçersiz")
        if self.use_qlora and not self.use_lora:
            raise ValueError("QLoRA, use_lora=True gerektirir")
        if self.architecture == "qwen_causal" and self.initialize_sequence_head:
            raise ValueError("Qwen causal reranker classification head başlatmaz")
        if self.lora_r < 1 or self.lora_alpha < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA ayarları geçersiz")


@dataclass(frozen=True)
class TrainConfig:
    """Query-aware optimizer and loss settings."""

    epochs: int = 2
    micro_batch_size: int = 4
    effective_batch_size: int = 32
    queries_per_batch: int = 4
    positives_per_query: int = 2
    negatives_per_positive: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    loss_kind: LossKind = "bce"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.02
    pairwise_weight: float = 0.20
    listwise_weight: float = 0.00
    listwise_temperature: float = 1.0
    base_margin: float = 0.50
    critical_margin: float = 0.20
    distillation_weight: float = 0.25
    distillation_temperature: float = 2.0
    auxiliary_weight: float = 0.10
    multi_sample_dropout: int = 1
    use_fgm: bool = False
    fgm_epsilon: float = 0.5
    use_swa: bool = False
    swa_start_fraction: float = 0.75
    num_workers: int = 0
    seed: int = 42

    def validate(self) -> None:
        """Validate optimizer, sampler and loss bounds."""

        if self.epochs < 1 or self.micro_batch_size < 1 or self.effective_batch_size < 1:
            raise ValueError("Epoch/batch değerleri pozitif olmalıdır")
        if self.queries_per_batch < 1 or self.positives_per_query < 1 or self.negatives_per_positive < 1:
            raise ValueError("Query group sampler ayarları pozitif olmalıdır")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate/max_grad_norm pozitif olmalıdır")
        if not 0 <= self.label_smoothing < 0.5:
            raise ValueError("label_smoothing [0,0.5) aralığında olmalıdır")
        if self.loss_kind == "focal" and self.label_smoothing > 0:
            raise ValueError("Focal loss ve label smoothing aynı ablation'da kullanılmamalıdır")
        for value in (
            self.pairwise_weight, self.listwise_weight,
            self.distillation_weight, self.auxiliary_weight,
        ):
            if value < 0:
                raise ValueError("Loss ağırlıkları negatif olamaz")
        if self.listwise_temperature <= 0:
            raise ValueError("listwise_temperature pozitif olmalıdır")
        if not 0 < self.swa_start_fraction < 1:
            raise ValueError("swa_start_fraction (0,1) aralığında olmalıdır")
        if not 1 <= self.multi_sample_dropout <= 8:
            raise ValueError("multi_sample_dropout [1,8] aralığında olmalıdır")


@dataclass(frozen=True)
class InferenceConfig:
    """Bounded-memory shard inference settings."""

    batch_size: int = 32
    minimum_batch_size: int = 1
    shard_size: int = 50_000
    device: str = "cuda"
    seed: int = 42

    def validate(self) -> None:
        """Validate inference batch and shard sizes."""

        if self.batch_size < 1 or self.minimum_batch_size < 1:
            raise ValueError("Inference batch değerleri pozitif olmalıdır")
        if self.minimum_batch_size > self.batch_size:
            raise ValueError("minimum_batch_size batch_size değerinden büyük olamaz")
        if self.shard_size < 1:
            raise ValueError("shard_size pozitif olmalıdır")


def require_torch_stack() -> dict[str, Any]:
    """Import deep-learning dependencies with an actionable error."""

    try:
        import torch
        import transformers
        from torch.utils.data import DataLoader, Dataset, Sampler
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError("GPU stack eksik; requirements-gpu.txt kurun") from exc
    version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    if version < (4, 51):
        raise RuntimeError(f"Qwen3 için transformers>=4.51 gerekli; bulunan={transformers.__version__}")
    return {
        "torch": torch, "transformers": transformers, "DataLoader": DataLoader,
        "Dataset": Dataset, "Sampler": Sampler,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "get_cosine_schedule_with_warmup": get_cosine_schedule_with_warmup,
    }


def sanitize_model_text(value: object) -> str:
    """Remove model control-token injection while preserving Turkish text."""

    import re

    text = "" if value is None or pd.isna(value) else str(value)
    return re.sub(CONTROL_TOKEN_PATTERN, " ", text).strip()


def format_qwen_instruction(query: object, document: object, instruction: str = DEFAULT_RERANK_INSTRUCTION) -> str:
    """Format the exact official Qwen3 reranker user payload."""

    safe_instruction = sanitize_model_text(instruction)
    safe_query = sanitize_model_text(query)
    safe_document = sanitize_model_text(document)
    return f"<Instruct>: {safe_instruction}\n<Query>: {safe_query}\n<Document>: {safe_document}"


def product_view_column(view: str) -> str:
    """Map a configured product view to its canonical text column."""

    mapping = {
        "short": "item_text_short", "long": "item_text_long",
        "short_ascii": "item_text_short_ascii", "long_ascii": "item_text_long_ascii",
    }
    if view not in mapping:
        raise ValueError(f"Bilinmeyen product view: {view}")
    return mapping[view]


def build_selected_item_view(items: pd.DataFrame, view: str) -> pd.DataFrame:
    """Build only the requested normalized item view to bound full-catalog RAM."""

    required = {"item_id", "title", "category", "brand", "gender", "age_group", "attributes"}
    if missing := required - set(items.columns):
        raise ValueError(f"Selected item view kolonları eksik: {sorted(missing)}")
    product_view_column(view)
    title = items["title"].fillna("").map(normalize_text)
    category = items["category"].fillna("").map(normalize_category)
    brand = items["brand"].fillna("").map(normalize_text)
    text = "[TITLE] " + title + " [CATEGORY] " + category + " [BRAND] " + brand
    if view.startswith("long"):
        gender = items["gender"].fillna("").map(normalize_text)
        age = items["age_group"].fillna("").map(normalize_text)
        attributes = items["attributes"].fillna("").map(normalize_text)
        text = text + " [GENDER] " + gender + " [AGE] " + age + " [ATTRIBUTES] " + attributes
    if view.endswith("ascii"):
        text = text.map(lambda value: normalize_text(value, ascii_fold=True))
    return pd.DataFrame({"item_id": items["item_id"].astype("string"), "product_text": text})


def load_or_build_item_view_cache(
    items: pd.DataFrame,
    view: str,
    cache_path: Path,
) -> pd.DataFrame:
    """Reuse a fingerprinted query-independent full-catalog product-text cache."""

    required = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    if missing := set(required) - set(items.columns):
        raise ValueError(f"Item view cache kolonları eksik: {sorted(missing)}")
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".json")
    row_hashes = pd.util.hash_pandas_object(items[required].astype("string"), index=False).values
    input_hash = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    expected = {"view": view, "rows": len(items), "input_hash": input_hash}
    if cache_path.exists() and meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing == expected:
            cached = pd.read_parquet(cache_path)
            if (
                len(cached) == len(items)
                and cached["item_id"].astype(str).equals(items["item_id"].astype(str).reset_index(drop=True))
            ):
                return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = build_selected_item_view(items, view)
    temporary = cache_path.with_suffix(".tmp.parquet")
    prepared.to_parquet(temporary, index=False)
    os.replace(temporary, cache_path)
    temporary_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    temporary_meta.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_meta, meta_path)
    return prepared


def enrich_pair_frame(
    pairs: pd.DataFrame,
    terms: pd.DataFrame,
    items: pd.DataFrame,
    *,
    product_view: str = "long",
    prepared_item_view: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join query and deterministic query-independent product text without reordering."""

    required = {"term_id", "item_id"}
    if missing := required - set(pairs.columns):
        raise ValueError(f"Pair kolonları eksik: {sorted(missing)}")
    if not {"term_id", "query"}.issubset(terms.columns):
        raise ValueError("terms term_id,query içermeli")
    if "item_id" not in items:
        raise ValueError("items item_id içermeli")
    out = pairs.drop(columns=["query", "product_text"], errors="ignore").copy()
    out["_row_idx"] = np.arange(len(out), dtype=np.int64)
    item_views = (
        prepared_item_view
        if prepared_item_view is not None
        else build_selected_item_view(items, product_view)
    )
    if not {"item_id", "product_text"}.issubset(item_views.columns):
        raise ValueError("prepared_item_view item_id,product_text içermeli")
    out = out.merge(
        terms[["term_id", "query"]].drop_duplicates("term_id"),
        on="term_id", how="left", validate="many_to_one", sort=False,
    ).merge(
        item_views[["item_id", "product_text"]].drop_duplicates("item_id"),
        on="item_id", how="left", validate="many_to_one", sort=False,
    )
    out = out.sort_values("_row_idx").reset_index(drop=True)
    if out[["query", "product_text"]].isna().any().any() or len(out) != len(pairs):
        raise ValueError("Pair enrichment sırasında eksik query/product veya satır kaybı oluştu")
    return out


def _deterministic_hash(seed: int, epoch: int, value: object) -> int:
    """Return a stable 64-bit sampling key."""

    return int.from_bytes(hashlib.blake2b(f"{seed}:{epoch}:{value}".encode(), digest_size=8).digest(), "little")


class QueryUniformBatchSampler:
    """Query-uniform batches with deterministic positive rotation and hard negatives."""

    def __init__(self, frame: pd.DataFrame, config: TrainConfig, epoch: int = 0) -> None:
        """Index eligible query groups and initialize deterministic epoch state."""

        config.validate()
        required = {"term_id", "label"}
        if missing := required - set(frame.columns):
            raise ValueError(f"Sampler kolonları eksik: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.epoch = epoch
        self.groups = {
            str(term_id): np.asarray(indices, dtype=np.int64)
            for term_id, indices in self.frame.groupby("term_id", sort=False).indices.items()
        }
        valid_groups = []
        for term_id, indices in self.groups.items():
            labels = self.frame.iloc[indices]["label"].astype(int).to_numpy()
            if (labels == 1).any() and (labels == 0).any():
                valid_groups.append(term_id)
        if not valid_groups:
            raise ValueError("Sampler en az bir pozitif ve negatif içeren query grubu bulamadı")
        self.term_ids = valid_groups

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic query order and positive rotation."""

        if epoch < 0:
            raise ValueError("epoch negatif olamaz")
        self.epoch = epoch

    def __len__(self) -> int:
        """Return batches per epoch, one visit per eligible query."""

        return math.ceil(len(self.term_ids) / min(self.config.queries_per_batch, self.config.micro_batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        """Yield lists of row indices grouped by query."""

        ordered_terms = sorted(
            self.term_ids, key=lambda term_id: _deterministic_hash(self.config.seed, self.epoch, term_id)
        )
        batch: list[int] = []
        query_count = 0
        for term_id in ordered_terms:
            indices = self.groups[term_id]
            group = self.frame.iloc[indices]
            positive = indices[group["label"].astype(int).to_numpy() == 1]
            negative = indices[group["label"].astype(int).to_numpy() == 0]
            positive = positive[np.argsort([
                _deterministic_hash(self.config.seed, self.epoch, f"{term_id}:p:{index}") for index in positive
            ])]
            start = (self.epoch * self.config.positives_per_query) % len(positive)
            positive = np.roll(positive, -start)[: self.config.positives_per_query]
            hardness = pd.to_numeric(
                self.frame.iloc[negative].get("hardness_score", pd.Series(0.0, index=range(len(negative)))),
                errors="coerce",
            ).fillna(0.0).to_numpy()
            tie = np.asarray([
                _deterministic_hash(self.config.seed, self.epoch, f"{term_id}:n:{index}") for index in negative
            ], dtype=np.uint64)
            negative_order = np.lexsort((tie, -hardness))
            required_negative = min(len(negative), len(positive) * self.config.negatives_per_positive)
            chosen_negative = negative[negative_order[:required_negative]]
            batch.extend(int(value) for value in np.concatenate([positive, chosen_negative]))
            query_count += 1
            if query_count >= min(self.config.queries_per_batch, self.config.micro_batch_size):
                yield batch
                batch, query_count = [], 0
        if batch:
            yield batch


def _resolve_dtype(torch: Any, requested: str, device: str) -> Any:
    """Resolve safe dtype for the current accelerator."""

    if requested == "fp32" or not device.startswith("cuda"):
        return torch.float32
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Bu GPU bf16 desteklemiyor")
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _attention_implementation(config: RerankerConfig, torch: Any) -> str:
    """Choose Flash Attention only when package and hardware support are present."""

    if config.attention != "auto":
        return config.attention
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        try:
            import flash_attn  # noqa: F401

            if major >= 8:
                return "flash_attention_2"
        except ImportError:
            return "sdpa"
    return "sdpa"


class RerankerAdapter:
    """Unified Qwen causal or one-logit sequence-classification adapter."""

    def __init__(
        self,
        config: RerankerConfig,
        *,
        device: str = "cuda",
        checkpoint: Path | None = None,
        trainable: bool = True,
    ) -> None:
        """Load the correct pretrained scoring architecture or saved adapter."""

        config.validate()
        stack = require_torch_stack()
        self.torch = stack["torch"]
        self.config = config
        self.device = device
        self.dtype = _resolve_dtype(self.torch, config.dtype, device)
        checkpoint_model = checkpoint / "model" if checkpoint and (checkpoint / "model").exists() else checkpoint
        is_adapter_checkpoint = bool(
            checkpoint_model and (checkpoint_model / "adapter_config.json").exists()
        )
        source = config.model_name if is_adapter_checkpoint else str(checkpoint_model or config.model_name)
        tokenizer_source = str(
            checkpoint / "tokenizer"
            if checkpoint and (checkpoint / "tokenizer").exists()
            else source
        )
        self.tokenizer = stack["AutoTokenizer"].from_pretrained(
            tokenizer_source,
            revision=config.revision if source == config.model_name else None,
            trust_remote_code=config.trust_remote_code,
            padding_side="left" if config.architecture == "qwen_causal" else "right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: dict[str, Any] = {
            "revision": config.revision if source == config.model_name else None,
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": self.dtype,
            "attn_implementation": _attention_implementation(config, self.torch),
        }
        if config.use_qlora and source == config.model_name:
            if not device.startswith("cuda"):
                raise RuntimeError("QLoRA yalnız CUDA üzerinde desteklenir")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("QLoRA için bitsandbytes gereklidir") from exc
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )
            load_kwargs["device_map"] = {"": 0}
        model_class = (
            stack["AutoModelForCausalLM"]
            if config.architecture == "qwen_causal"
            else stack["AutoModelForSequenceClassification"]
        )
        if config.architecture == "sequence_classifier" and config.initialize_sequence_head and not checkpoint:
            load_kwargs["num_labels"] = 1
            load_kwargs["ignore_mismatched_sizes"] = True
        self.model = model_class.from_pretrained(source, **load_kwargs)
        if config.architecture == "sequence_classifier":
            if int(getattr(self.model.config, "num_labels", 0)) != 1:
                raise RuntimeError("Sequence reranker pretrained tek-logit head içermelidir; rastgele head açılmadı")
        if config.gradient_checkpointing and trainable:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        if is_adapter_checkpoint and checkpoint_model is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("LoRA checkpoint yüklemek için peft gereklidir") from exc
            self.model = PeftModel.from_pretrained(
                self.model, checkpoint_model, is_trainable=trainable
            )
        elif config.use_lora and source == config.model_name and trainable:
            self._attach_lora()
        if not config.use_qlora:
            self.model.to(device)
        self.prefix_tokens: list[int] = []
        self.suffix_tokens: list[int] = []
        self.yes_token_id: int | None = None
        self.no_token_id: int | None = None
        if config.architecture == "qwen_causal":
            self._configure_qwen_tokens()

    def _attach_lora(self) -> None:
        """Attach PEFT LoRA/QLoRA without replacing pretrained scoring heads."""

        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("LoRA için peft>=0.15 gereklidir") from exc
        if self.config.use_qlora:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=self.config.gradient_checkpointing
            )
        task_type = TaskType.CAUSAL_LM if self.config.architecture == "qwen_causal" else TaskType.SEQ_CLS
        kwargs: dict[str, Any] = {
            "task_type": task_type,
            "r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": "all-linear",
            "bias": "none",
        }
        if self.config.architecture == "sequence_classifier":
            kwargs["modules_to_save"] = ["classifier"]
        self.model = get_peft_model(self.model, LoraConfig(**kwargs))
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        if trainable <= 0:
            raise RuntimeError("LoRA sonrası trainable parametre bulunamadı")

    def _configure_qwen_tokens(self) -> None:
        """Validate official prefix/suffix and single-token yes/no outputs."""

        self.prefix_tokens = self.tokenizer.encode(QWEN_SYSTEM_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(QWEN_ASSISTANT_SUFFIX, add_special_tokens=False)
        yes_ids = self.tokenizer.encode("yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("no", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise RuntimeError(f"Qwen yes/no tek token olmalı; yes={yes_ids}, no={no_ids}")
        self.yes_token_id, self.no_token_id = int(yes_ids[0]), int(no_ids[0])
        if self.yes_token_id == self.no_token_id:
            raise RuntimeError("Qwen yes/no token id aynı olamaz")

    def tokenize(self, queries: Sequence[str], products: Sequence[str]) -> Mapping[str, Any]:
        """Tokenize pairs with architecture-correct padding and truncation."""

        if len(queries) != len(products) or not queries:
            raise ValueError("Tokenize query/product listeleri boş veya farklı uzunlukta")
        if self.config.architecture == "qwen_causal":
            payloads = [
                format_qwen_instruction(query, product, self.config.instruction)
                for query, product in zip(queries, products)
            ]
            budget = self.config.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
            if budget < 16:
                raise ValueError("Qwen prompt prefix/suffix sonrası token bütçesi yetersiz")
            encoded = self.tokenizer(
                payloads, padding=False, truncation=True, max_length=budget,
                add_special_tokens=False, return_attention_mask=False,
            )
            encoded["input_ids"] = [
                self.prefix_tokens + ids + self.suffix_tokens for ids in encoded["input_ids"]
            ]
            batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        else:
            safe_queries = [sanitize_model_text(value) for value in queries]
            safe_products = [sanitize_model_text(value) for value in products]
            # Bound query tokens separately so a verbose query cannot evict product fields.
            bounded_queries = []
            for query in safe_queries:
                ids = self.tokenizer.encode(
                    query, add_special_tokens=False, truncation=True, max_length=self.config.query_max_tokens
                )
                bounded_queries.append(self.tokenizer.decode(ids, skip_special_tokens=True))
            batch = self.tokenizer(
                bounded_queries, safe_products, padding=True, truncation="only_second",
                max_length=self.config.max_length, return_tensors="pt",
            )
        return {key: value.to(self.device) for key, value in batch.items()}

    def forward_logits(
        self,
        tokenized: Mapping[str, Any],
        *,
        need_representation: bool = False,
    ) -> tuple[Any, Any | None]:
        """Return one relevance logit per pair and optional hidden representation."""

        kwargs = dict(tokenized)
        kwargs["output_hidden_states"] = need_representation
        if self.config.architecture == "qwen_causal":
            kwargs["use_cache"] = False
            kwargs["logits_to_keep"] = 1
            output = self.model(**kwargs)
            vocab_logits = output.logits[:, -1, :]
            logits = vocab_logits[:, self.yes_token_id] - vocab_logits[:, self.no_token_id]
            representation = output.hidden_states[-1][:, -1, :] if need_representation else None
        else:
            output = self.model(**kwargs)
            if output.logits.ndim != 2 or output.logits.shape[1] != 1:
                raise RuntimeError(f"Sequence reranker logits shape geçersiz: {tuple(output.logits.shape)}")
            logits = output.logits[:, 0]
            representation = output.hidden_states[-1][:, 0, :] if need_representation else None
        if logits.ndim != 1 or logits.shape[0] != tokenized["input_ids"].shape[0]:
            raise RuntimeError("Reranker her pair için tam bir logit üretmedi")
        return logits, representation

    def save(self, output_dir: Path) -> None:
        """Save model/adapter, tokenizer and immutable configuration."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir / "model", safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir / "tokenizer")
        (output_dir / "reranker_config.json").write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def query_aware_loss(
    torch: Any,
    logits: Any,
    labels: Any,
    weights: Any,
    term_codes: Any,
    contradiction: Any,
    config: TrainConfig,
    *,
    teacher_probability: Any | None = None,
    auxiliary_logits: Any | None = None,
    auxiliary_targets: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    """Combine weighted pointwise, same-query pairwise, distillation and auxiliary losses."""

    labels = labels.float()
    weights = weights.float().clamp_min(1e-6)
    if config.loss_kind == "bce":
        target = labels * (1 - config.label_smoothing) + 0.5 * config.label_smoothing
        point = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    else:
        base = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        probability = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, probability, 1 - probability)
        point = (1 - pt).pow(config.focal_gamma) * base
    point_loss = (point * weights).sum() / weights.sum()
    pair_terms = []
    for code in torch.unique(term_codes):
        group = term_codes == code
        positive_idx = torch.nonzero(group & (labels > 0.5), as_tuple=False).flatten()
        negative_idx = torch.nonzero(group & (labels <= 0.5), as_tuple=False).flatten()
        if positive_idx.numel() == 0 or negative_idx.numel() == 0:
            continue
        pos = positive_idx.repeat_interleave(negative_idx.numel())
        neg = negative_idx.repeat(positive_idx.numel())
        margin = config.base_margin + config.critical_margin * contradiction[neg].float()
        pair_terms.append(torch.nn.functional.softplus(margin - (logits[pos] - logits[neg])).mean())
    pair_loss = torch.stack(pair_terms).mean() if pair_terms else logits.sum() * 0.0
    listwise_terms = []
    for code in torch.unique(term_codes):
        group = term_codes == code
        group_labels = labels[group]
        if group_labels.sum() <= 0 or (~(group_labels > 0.5)).sum() <= 0:
            continue
        target_distribution = group_labels / group_labels.sum().clamp_min(1.0)
        predicted_log_distribution = torch.nn.functional.log_softmax(
            logits[group] / config.listwise_temperature, dim=0
        )
        listwise_terms.append(-(target_distribution * predicted_log_distribution).sum())
    listwise_loss = torch.stack(listwise_terms).mean() if listwise_terms else logits.sum() * 0.0
    distill_loss = logits.sum() * 0.0
    if teacher_probability is not None and config.distillation_weight > 0:
        teacher = teacher_probability.float()
        mask = torch.isfinite(teacher)
        if mask.any():
            teacher = teacher[mask].clamp(1e-5, 1 - 1e-5)
            teacher_logit = torch.logit(teacher) / config.distillation_temperature
            student = logits[mask] / config.distillation_temperature
            distill_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    student, torch.sigmoid(teacher_logit), reduction="mean"
                ) * (config.distillation_temperature ** 2)
            )
    auxiliary_loss = logits.sum() * 0.0
    if auxiliary_logits is not None and auxiliary_targets is not None and config.auxiliary_weight > 0:
        target = auxiliary_targets.float()
        mask = target >= 0
        if mask.any():
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                auxiliary_logits, target.clamp(0, 1), reduction="none"
            )
            auxiliary_loss = raw[mask].mean()
    total = (
        point_loss + config.pairwise_weight * pair_loss
        + config.listwise_weight * listwise_loss
        + config.distillation_weight * distill_loss
        + config.auxiliary_weight * auxiliary_loss
    )
    parts = {
        "point": float(point_loss.detach().cpu()), "pairwise": float(pair_loss.detach().cpu()),
        "listwise": float(listwise_loss.detach().cpu()),
        "distillation": float(distill_loss.detach().cpu()), "auxiliary": float(auxiliary_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, parts


class FGM:
    """Fast gradient perturbation for trainable embedding parameters."""

    def __init__(self, model: Any, epsilon: float) -> None:
        """Bind a model and perturbation radius."""

        self.model = model
        self.epsilon = epsilon
        self.backup: dict[str, Any] = {}

    def attack(self) -> None:
        """Perturb embedding parameters along their normalized gradient."""

        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None and "embed" in name:
                norm = parameter.grad.norm()
                if norm.isfinite() and norm > 0:
                    self.backup[name] = parameter.data.clone()
                    parameter.data.add_(self.epsilon * parameter.grad / norm)

    def restore(self) -> None:
        """Restore every perturbed parameter exactly."""

        for name, parameter in self.model.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup.clear()


def _raw_batch(frame: pd.DataFrame, indices: Sequence[int], auxiliary_columns: Sequence[str]) -> dict[str, Any]:
    """Build one query-aware raw text batch from frame indices."""

    batch = frame.iloc[list(indices)]
    term_codes, _ = pd.factorize(batch["term_id"].astype(str), sort=True)
    teacher = pd.to_numeric(
        batch["teacher_probability"] if "teacher_probability" in batch else pd.Series(np.nan, index=batch.index),
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    auxiliary = np.column_stack([
        pd.to_numeric(batch[column], errors="coerce").fillna(-1).to_numpy(dtype=np.float32)
        for column in auxiliary_columns
    ]) if auxiliary_columns else np.empty((len(batch), 0), dtype=np.float32)
    return {
        "queries": batch["query"].fillna("").astype(str).tolist(),
        "products": batch["product_text"].fillna("").astype(str).tolist(),
        "labels": batch["label"].to_numpy(dtype=np.float32),
        "weights": pd.to_numeric(
            batch["sample_weight"] if "sample_weight" in batch else pd.Series(1.0, index=batch.index),
            errors="raise",
        ).to_numpy(dtype=np.float32),
        "term_codes": term_codes.astype(np.int64),
        "contradiction": pd.to_numeric(
            batch["contradiction_any"] if "contradiction_any" in batch else pd.Series(0, index=batch.index),
            errors="coerce",
        ).fillna(0).to_numpy(dtype=np.float32),
        "teacher": teacher,
        "auxiliary": auxiliary,
    }


def validate_training_frame(frame: pd.DataFrame) -> None:
    """Validate a fully enriched positive+negative reranker frame."""

    required = {"term_id", "item_id", "query", "product_text", "label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Reranker train kolonları eksik: {sorted(missing)}")
    if frame.duplicated(["term_id", "item_id"]).any():
        raise ValueError("Reranker train duplicate term-item içeriyor")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Reranker label yalnızca 0/1 olmalıdır")
    per_term = frame.assign(_label=labels).groupby("term_id")["_label"].agg(["min", "max"])
    if not ((per_term["min"] == 0) & (per_term["max"] == 1)).all():
        raise ValueError("Her training query en az bir pozitif ve bir negatif içermelidir")


def train_reranker(
    train_frame: pd.DataFrame,
    output_dir: Path,
    model_config: RerankerConfig,
    train_config: TrainConfig,
    *,
    device: str = "cuda",
    resume_checkpoint: Path | None = None,
    auxiliary_columns: Sequence[str] = (
        "wrong_category", "wrong_brand", "wrong_gender", "wrong_age_group",
        "wrong_color", "wrong_product_type", "contradiction_any",
    ),
) -> Path:
    """Train a query-aware reranker with deterministic checkpoint-per-epoch resume."""

    model_config.validate()
    train_config.validate()
    validate_training_frame(train_frame)
    set_global_seed(train_config.seed)
    stack = require_torch_stack()
    torch = stack["torch"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    auxiliary_columns = tuple(column for column in auxiliary_columns if column in train_frame)
    adapter = RerankerAdapter(
        model_config, device=device, checkpoint=resume_checkpoint, trainable=True
    )
    hidden_size = int(getattr(adapter.model.config, "hidden_size", 0))
    auxiliary_head = None
    if auxiliary_columns and train_config.auxiliary_weight > 0:
        if hidden_size <= 0:
            raise RuntimeError("Auxiliary heads için model hidden_size bulunamadı")
        auxiliary_head = torch.nn.Linear(hidden_size, len(auxiliary_columns)).to(device=device, dtype=adapter.dtype)
    parameters = [parameter for parameter in adapter.model.parameters() if parameter.requires_grad]
    if auxiliary_head is not None:
        parameters.extend(auxiliary_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    sampler = QueryUniformBatchSampler(train_frame, train_config)
    accumulation = max(1, math.ceil(train_config.effective_batch_size / train_config.micro_batch_size))
    total_steps = max(1, math.ceil(len(sampler) / accumulation) * train_config.epochs)
    scheduler = stack["get_cosine_schedule_with_warmup"](
        optimizer,
        num_warmup_steps=int(total_steps * train_config.warmup_ratio),
        num_training_steps=total_steps,
    )
    start_epoch, global_step = 0, 0
    if resume_checkpoint and (resume_checkpoint / "trainer_state.pt").exists():
        state = torch.load(resume_checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        if auxiliary_head is not None and state.get("auxiliary_head") is not None:
            auxiliary_head.load_state_dict(state["auxiliary_head"])
    scaler = torch.amp.GradScaler("cuda", enabled=adapter.dtype == torch.float16 and device.startswith("cuda"))
    if train_config.use_fgm and scaler.is_enabled():
        raise ValueError("FGM fp16 GradScaler ile güvenli değil; bf16 veya fp32 kullanın")
    fgm = FGM(adapter.model, train_config.fgm_epsilon) if train_config.use_fgm else None
    swa_model = None
    if train_config.use_swa:
        if model_config.use_qlora:
            raise ValueError("SWA ve QLoRA aynı deneyde desteklenmez")
        swa_model = torch.optim.swa_utils.AveragedModel(adapter.model)
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = device.startswith("cuda") and adapter.dtype in {torch.float16, torch.bfloat16}
    for epoch in range(start_epoch, train_config.epochs):
        sampler.set_epoch(epoch)
        adapter.model.train()
        if auxiliary_head is not None:
            auxiliary_head.train()
        running: list[dict[str, float]] = []
        for batch_index, indices in enumerate(sampler):
            raw = _raw_batch(train_frame, indices, auxiliary_columns)
            tokenized = adapter.tokenize(raw["queries"], raw["products"])
            tensors = {
                key: torch.as_tensor(value, device=device)
                for key, value in raw.items() if key not in {"queries", "products"}
            }
            with torch.autocast(
                device_type="cuda", dtype=adapter.dtype,
                enabled=autocast_enabled,
            ):
                dropout_outputs = [
                    adapter.forward_logits(
                        tokenized, need_representation=auxiliary_head is not None
                    )
                    for _ in range(train_config.multi_sample_dropout)
                ]
                logits = torch.stack([value[0] for value in dropout_outputs]).mean(dim=0)
                representation = (
                    torch.stack([value[1] for value in dropout_outputs]).mean(dim=0)
                    if auxiliary_head is not None else None
                )
                auxiliary_logits = auxiliary_head(representation) if auxiliary_head is not None else None
                loss, parts = query_aware_loss(
                    torch, logits, tensors["labels"], tensors["weights"], tensors["term_codes"],
                    tensors["contradiction"], train_config,
                    teacher_probability=tensors["teacher"],
                    auxiliary_logits=auxiliary_logits,
                    auxiliary_targets=tensors["auxiliary"],
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if fgm is not None:
                fgm.attack()
                with torch.autocast(device_type="cuda", dtype=adapter.dtype, enabled=autocast_enabled):
                    adversarial_logits, _ = adapter.forward_logits(tokenized, need_representation=False)
                    adversarial_loss, _ = query_aware_loss(
                        torch, adversarial_logits, tensors["labels"], tensors["weights"],
                        tensors["term_codes"], tensors["contradiction"], train_config,
                        teacher_probability=tensors["teacher"],
                    )
                scaler.scale(adversarial_loss / accumulation).backward()
                fgm.restore()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(sampler):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, train_config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if swa_model is not None and global_step >= int(total_steps * train_config.swa_start_fraction):
                    swa_model.update_parameters(adapter.model)
            running.append(parts)
        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            **{key: float(np.mean([row[key] for row in running])) for key in running[0]},
        }
        history.append(epoch_metrics)
        checkpoint = output_dir / f"checkpoint-epoch-{epoch:02d}"
        adapter.save(checkpoint)
        state = {
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "global_step": global_step,
            "auxiliary_head": auxiliary_head.state_dict() if auxiliary_head is not None else None,
            "auxiliary_columns": list(auxiliary_columns), "train_config": asdict(train_config),
        }
        torch.save(state, checkpoint / "trainer_state.pt")
        (checkpoint / "checkpoint_manifest.json").write_text(
            json.dumps({
                "model_config": asdict(model_config),
                "train_resume_signature": _resume_signature(train_config),
                "ordered_pair_hash": _ordered_pair_hash(train_frame),
                "epoch": epoch,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
    final_path = output_dir / "final"
    if swa_model is not None and swa_model.n_averaged > 0:
        adapter.model.load_state_dict(swa_model.module.state_dict())
    adapter.save(final_path)
    if auxiliary_head is not None:
        torch.save({"state_dict": auxiliary_head.state_dict(), "columns": list(auxiliary_columns)}, final_path / "auxiliary_head.pt")
    manifest = {
        "model_config": asdict(model_config), "train_config": asdict(train_config),
        "rows": len(train_frame), "terms": int(train_frame["term_id"].nunique()),
        "positive_rows": int(train_frame["label"].sum()),
        "negative_rows": int(train_frame["label"].eq(0).sum()),
        "ordered_pair_hash": _ordered_pair_hash(train_frame),
        "completed_at": time.time(),
    }
    (final_path / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return final_path


def train_with_oom_fallback(
    train_frame: pd.DataFrame,
    output_dir: Path,
    model_config: RerankerConfig,
    train_config: TrainConfig,
    *,
    device: str = "cuda",
) -> Path:
    """Restart deterministically with a smaller microbatch after CUDA OOM."""

    current = train_config
    output_dir = Path(output_dir)
    final_manifest = output_dir / "final" / "run_manifest.json"
    if final_manifest.exists():
        existing = json.loads(final_manifest.read_text(encoding="utf-8"))
        if (
            existing.get("model_config") == asdict(model_config)
            and existing.get("train_config") == asdict(train_config)
            and existing.get("ordered_pair_hash") == _ordered_pair_hash(train_frame)
        ):
            return output_dir / "final"
    while True:
        all_checkpoints = sorted(
            output_dir.glob("checkpoint-epoch-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        compatible = []
        for checkpoint in all_checkpoints:
            manifest_path = checkpoint / "checkpoint_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("model_config") == asdict(model_config)
                and manifest.get("train_resume_signature") == _resume_signature(current)
                and manifest.get("ordered_pair_hash") == _ordered_pair_hash(train_frame)
            ):
                compatible.append(checkpoint)
        if all_checkpoints and not compatible:
            raise ValueError(
                f"{output_dir} altında mevcut checkpoint'ler farklı config/veriye ait; "
                "yeni experiment_id/output_dir kullanın"
            )
        resume = compatible[-1] if compatible else None
        try:
            return train_reranker(
                train_frame, output_dir, model_config, current,
                device=device, resume_checkpoint=resume,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current.micro_batch_size <= 1:
                raise
            stack = require_torch_stack()
            torch = stack["torch"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            current = replace(current, micro_batch_size=max(1, current.micro_batch_size // 2))
            (Path(output_dir) / "oom_fallback.json").write_text(
                json.dumps({"new_micro_batch_size": current.micro_batch_size, "error": str(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )


def _ordered_pair_hash(frame: pd.DataFrame) -> str:
    """Hash ordered pair IDs for cache and shard reuse validation."""

    columns = [column for column in ("id", "term_id", "item_id") if column in frame]
    if not {"term_id", "item_id"}.issubset(columns):
        raise ValueError("Pair hash term_id,item_id gerektirir")
    hashes = pd.util.hash_pandas_object(frame[columns].astype("string"), index=False).values
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def _resume_signature(config: TrainConfig) -> dict[str, Any]:
    """Return checkpoint-compatible training settings, excluding OOM microbatch size."""

    payload = asdict(config)
    payload.pop("micro_batch_size", None)
    return payload


def score_frame_adaptive(
    frame: pd.DataFrame,
    adapter: RerankerAdapter,
    inference: InferenceConfig,
) -> tuple[np.ndarray, int]:
    """Score an enriched frame, retrying the same range with halved batches on OOM."""

    inference.validate()
    required = {"query", "product_text"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Inference frame kolonları eksik: {sorted(missing)}")
    torch = adapter.torch
    adapter.model.eval()
    batch_size = inference.batch_size
    while True:
        try:
            chunks: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(frame), batch_size):
                    batch = frame.iloc[start:start + batch_size]
                    tokenized = adapter.tokenize(
                        batch["query"].fillna("").astype(str).tolist(),
                        batch["product_text"].fillna("").astype(str).tolist(),
                    )
                    enabled = inference.device.startswith("cuda") and adapter.dtype in {torch.float16, torch.bfloat16}
                    with torch.autocast(device_type="cuda", dtype=adapter.dtype, enabled=enabled):
                        logits, _ = adapter.forward_logits(tokenized)
                    chunks.append(logits.float().cpu().numpy())
            result = np.concatenate(chunks).astype(np.float32) if chunks else np.empty(0, dtype=np.float32)
            if len(result) != len(frame) or not np.isfinite(result).all():
                raise RuntimeError("Inference eksik veya NaN/Inf logit üretti")
            return result, batch_size
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= inference.minimum_batch_size:
                raise
            batch_size = max(inference.minimum_batch_size, batch_size // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def score_oof_frame(
    frame: pd.DataFrame,
    adapter: RerankerAdapter,
    inference: InferenceConfig,
    *,
    fold: int,
    model_key: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one fixed outer validation universe and retain raw logits."""

    logits, used_batch = score_frame_adaptive(frame, adapter, inference)
    output = frame[[column for column in ("pair_uid", "id", "term_id", "item_id", "label", "label_source") if column in frame]].copy()
    if "pair_uid" not in output:
        output["pair_uid"] = [
            hashlib.blake2b(f"{fold}|{term}|{item}".encode(), digest_size=12).hexdigest()
            for term, item in zip(frame["term_id"], frame["item_id"])
        ]
    output["fold"] = np.int16(fold)
    output["seed"] = np.int32(seed)
    output["model_key"] = model_key
    output["raw_score_kind"] = "yes_no_logit" if adapter.config.architecture == "qwen_causal" else "classification_logit"
    output["raw_logit"] = logits
    output["raw_probability"] = sigmoid(logits).astype(np.float32)
    validate_oof_frame(output)
    metrics = classification_report_dict(
        output["label"].to_numpy(), (output["raw_probability"] >= 0.5).astype(np.int8),
        probability=output["raw_probability"].to_numpy(), threshold=0.5,
    )
    metrics.update({"fold": fold, "seed": seed, "model_key": model_key, "inference_batch_size": used_batch})
    return output, metrics


def _model_fingerprint(model_path: Path, model_config: RerankerConfig) -> str:
    """Fingerprint configuration plus local model file metadata."""

    entries = []
    for path in sorted(Path(model_path).rglob("*")):
        if path.is_file():
            stat = path.stat()
            entries.append((str(path.relative_to(model_path)), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps({"config": asdict(model_config), "files": entries}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _valid_shard(path: Path, done_path: Path, expected: Mapping[str, Any]) -> bool:
    """Accept a cached shard only when all fingerprints and row bounds match."""

    if not path.exists() or not done_path.exists():
        return False
    try:
        manifest = json.loads(done_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in expected.items()):
            return False
        frame = pd.read_parquet(path, columns=["_row_idx", "id", "raw_logit", "probability_raw"])
        return (
            len(frame) == expected["rows"]
            and int(frame["_row_idx"].iloc[0]) == expected["start"]
            and int(frame["_row_idx"].iloc[-1]) == expected["stop"] - 1
            and np.isfinite(frame[["raw_logit", "probability_raw"]].to_numpy()).all()
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def chunked_pair_inference(
    pairs_path: Path,
    terms: pd.DataFrame,
    items: pd.DataFrame,
    model_path: Path,
    output_dir: Path,
    model_config: RerankerConfig,
    inference: InferenceConfig,
    *,
    model_key: str,
    item_view_cache_path: Path | None = None,
) -> Path:
    """Score CSV pairs in restartable ordered shards and stream a final Parquet."""

    inference.validate()
    set_global_seed(inference.seed)
    pairs_path, output_dir = Path(pairs_path), Path(output_dir)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pair CSV bulunamadı: {pairs_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "score_shards"
    shard_dir.mkdir(exist_ok=True)
    adapter = RerankerAdapter(model_config, device=inference.device, checkpoint=model_path, trainable=False)
    fingerprint = _model_fingerprint(model_path, model_config)
    prepared_item_view = (
        load_or_build_item_view_cache(items, model_config.product_view, item_view_cache_path)
        if item_view_cache_path is not None
        else build_selected_item_view(items, model_config.product_view)
    )
    shard_paths: list[Path] = []
    row_offset = 0
    for shard_index, raw in enumerate(pd.read_csv(pairs_path, dtype="string", chunksize=inference.shard_size)):
        required = {"id", "term_id", "item_id"}
        if missing := required - set(raw.columns):
            raise ValueError(f"Inference pair kolonları eksik: {sorted(missing)}")
        raw["_row_idx"] = np.arange(row_offset, row_offset + len(raw), dtype=np.int64)
        row_offset += len(raw)
        pair_hash = _ordered_pair_hash(raw)
        shard_path = shard_dir / f"part-{shard_index:05d}.parquet"
        done_path = shard_dir / f"part-{shard_index:05d}.done.json"
        expected = {
            "rows": len(raw), "start": int(raw["_row_idx"].iloc[0]),
            "stop": int(raw["_row_idx"].iloc[-1]) + 1, "pair_hash": pair_hash,
            "model_fingerprint": fingerprint, "model_key": model_key,
        }
        shard_paths.append(shard_path)
        if _valid_shard(shard_path, done_path, expected):
            continue
        enriched = enrich_pair_frame(
            raw.drop(columns="_row_idx"), terms, items,
            product_view=model_config.product_view,
            prepared_item_view=prepared_item_view,
        )
        enriched["_row_idx"] = raw["_row_idx"].to_numpy()
        logits, used_batch = score_frame_adaptive(enriched, adapter, inference)
        result = enriched[["_row_idx", "id", "term_id", "item_id"]].copy()
        result["raw_logit"] = logits
        result["probability_raw"] = sigmoid(logits).astype(np.float32)
        result["model_key"] = model_key
        result["model_fingerprint"] = fingerprint
        tmp_path = shard_path.with_suffix(".tmp.parquet")
        result.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, shard_path)
        manifest = {**expected, "used_batch_size": used_batch, "created_at": time.time()}
        tmp_done = done_path.with_suffix(".tmp.json")
        tmp_done.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_done, done_path)
    if not shard_paths:
        raise ValueError("Inference pair CSV boş")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Final shard birleştirme için pyarrow gereklidir") from exc
    final_path = output_dir / f"{model_key}_test_probabilities.parquet"
    temp_final = final_path.with_suffix(".tmp.parquet")
    writer = None
    expected_row = 0
    try:
        for shard_path in shard_paths:
            table = pq.read_table(shard_path)
            rows = table.column("_row_idx").to_numpy()
            if len(rows) == 0 or int(rows[0]) != expected_row or not np.array_equal(rows, np.arange(expected_row, expected_row + len(rows))):
                raise RuntimeError(f"Shard row order bozuk: {shard_path}")
            expected_row += len(rows)
            if writer is None:
                writer = pq.ParquetWriter(temp_final, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temp_final, final_path)
    (output_dir / f"{model_key}_inference_manifest.json").write_text(
        json.dumps({"rows": expected_row, "shards": len(shard_paths), "model_fingerprint": fingerprint}, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_path


def benchmark_inference(
    frame: pd.DataFrame,
    adapter: RerankerAdapter,
    inference: InferenceConfig,
    *,
    total_pairs: int,
    folds: int = 1,
) -> dict[str, float | int]:
    """Benchmark a stratified sample and estimate runtime with a 15% guard factor."""

    if len(frame) < 1 or total_pairs < 1 or folds < 1:
        raise ValueError("Benchmark frame/total_pairs/folds pozitif olmalıdır")
    start = time.perf_counter()
    _, used_batch = score_frame_adaptive(frame, adapter, inference)
    elapsed = max(time.perf_counter() - start, 1e-6)
    rate = len(frame) / elapsed
    return {
        "benchmark_rows": len(frame), "seconds": elapsed,
        "pairs_per_second": rate, "used_batch_size": used_batch,
        "estimated_hours": total_pairs * folds / rate / 3600 * 1.15,
    }


def parse_args() -> argparse.Namespace:
    """Parse train, OOF-score, test-infer or benchmark commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "score-oof", "infer", "benchmark"], required=True)
    parser.add_argument("--input", type=Path, required=True, help="Enriched train/OOF Parquet or pair CSV")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3/reranker"))
    parser.add_argument("--model-name", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--architecture", choices=["qwen_causal", "sequence_classifier"], default="qwen_causal")
    parser.add_argument("--model-key", default="qwen3_reranker_06b")
    parser.add_argument("--product-view", choices=["short", "long", "short_ascii", "long_ascii"], default="long")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--item-view-cache", type=Path)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--pairwise-weight", type=float, default=0.20)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Execute the selected reranker workflow."""

    args = parse_args()
    dtype = "bf16" if args.bf16 else "fp16" if args.fp16 else "auto"
    model_config = RerankerConfig(
        model_name=args.model_name, architecture=args.architecture,
        max_length=args.max_length, product_view=args.product_view,
        use_lora=args.lora, use_qlora=args.qlora, dtype=dtype,
    )
    train_config = TrainConfig(
        epochs=args.epochs, micro_batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        learning_rate=args.learning_rate, pairwise_weight=args.pairwise_weight, seed=args.seed,
    )
    inference = InferenceConfig(
        batch_size=args.inference_batch_size, shard_size=args.shard_size,
        device=args.device, seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        frame = pd.read_parquet(args.input)
        path = train_with_oom_fallback(frame, args.output_dir, model_config, train_config, device=args.device)
        print(path)
        return
    if args.model_path is None:
        raise ValueError("score-oof/infer/benchmark modunda --model-path zorunludur")
    if args.mode == "infer":
        terms = pd.read_csv(args.data_dir / "terms.csv", dtype="string")
        items = pd.read_csv(args.data_dir / "items.csv", dtype="string")
        path = chunked_pair_inference(
            args.input, terms, items, args.model_path, args.output_dir,
            model_config, inference, model_key=args.model_key,
            item_view_cache_path=args.item_view_cache,
        )
        print(path)
        return
    frame = pd.read_parquet(args.input)
    adapter = RerankerAdapter(model_config, device=args.device, checkpoint=args.model_path, trainable=False)
    if args.mode == "benchmark":
        report = benchmark_inference(frame, adapter, inference, total_pairs=3_359_679)
        print(json.dumps(report, indent=2))
        return
    oof, metrics = score_oof_frame(
        frame, adapter, inference, fold=args.fold, model_key=args.model_key, seed=args.seed
    )
    oof.to_parquet(args.output_dir / f"{args.model_key}_fold{args.fold}_oof.parquet", index=False)
    (args.output_dir / f"{args.model_key}_fold{args.fold}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
