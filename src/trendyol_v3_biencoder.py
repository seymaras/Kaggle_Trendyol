#!/usr/bin/env python3
"""Instruction-aware bi-encoder training, embedding caches and FAISS mining."""

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
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from trendyol_v3_core import set_global_seed
from trendyol_v3_reranker import QueryUniformBatchSampler, TrainConfig, require_torch_stack


DEFAULT_RETRIEVAL_INSTRUCTION = (
    "Given a Turkish e-commerce search query, retrieve products that satisfy the shopper's "
    "intended product type and requested attributes"
)


@dataclass(frozen=True)
class BiEncoderConfig:
    """Bi-encoder architecture, pooling and parameter-efficient training settings."""

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str | None = None
    trust_remote_code: bool = False
    pooling: Literal["auto", "last_token", "cls", "mean"] = "auto"
    query_instruction: str = DEFAULT_RETRIEVAL_INSTRUCTION
    max_length: int = 256
    output_dimension: int = 1024
    matryoshka_dimensions: tuple[int, ...] = (1024, 512, 256)
    temperature: float = 0.05
    use_lora: bool = True
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"

    def validate(self) -> None:
        """Validate dimensions, temperature and adapter settings."""

        if not self.model_name or self.max_length < 32:
            raise ValueError("Bi-encoder model_name/max_length geçersiz")
        if self.output_dimension < 32 or not self.matryoshka_dimensions:
            raise ValueError("Embedding dimensions geçersiz")
        if max(self.matryoshka_dimensions) > self.output_dimension or min(self.matryoshka_dimensions) < 32:
            raise ValueError("Matryoshka dimensions output_dimension sınırları dışında")
        if not 0 < self.temperature <= 1:
            raise ValueError("temperature (0,1] aralığında olmalıdır")
        if self.use_qlora and not self.use_lora:
            raise ValueError("QLoRA use_lora=True gerektirir")


@dataclass(frozen=True)
class EmbeddingCacheSpec:
    """Immutable cache identity for ordered text embeddings."""

    role: Literal["query", "document"]
    id_column: str
    text_column: str
    rows: int
    dimension: int
    ordered_input_hash: str
    model_fingerprint: str


def _resolve_pooling(config: BiEncoderConfig) -> str:
    """Resolve official pooling defaults for supported model families."""

    if config.pooling != "auto":
        return config.pooling
    lowered = config.model_name.lower()
    if "qwen3-embedding" in lowered:
        return "last_token"
    if "bge-m3" in lowered:
        return "cls"
    return "mean"


def _resolve_dtype(torch: Any, config: BiEncoderConfig, device: str) -> Any:
    """Resolve accelerator-safe model dtype."""

    if config.dtype == "fp32" or not device.startswith("cuda"):
        return torch.float32
    if config.dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Bu GPU bf16 desteklemiyor")
        return torch.bfloat16
    if config.dtype == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class BiEncoderAdapter:
    """Shared query/document encoder with role-specific formatting and pooling."""

    def __init__(
        self,
        config: BiEncoderConfig,
        *,
        device: str = "cuda",
        checkpoint: Path | None = None,
        trainable: bool = False,
    ) -> None:
        """Load a base/full/adapter encoder with role-correct tokenizer padding."""

        config.validate()
        stack = require_torch_stack()
        torch = stack["torch"]
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Bi-encoder için transformers gereklidir") from exc
        self.torch = torch
        self.config = config
        self.device = device
        self.dtype = _resolve_dtype(torch, config, device)
        checkpoint_model = checkpoint / "model" if checkpoint and (checkpoint / "model").exists() else checkpoint
        is_adapter = bool(checkpoint_model and (checkpoint_model / "adapter_config.json").exists())
        source = config.model_name if is_adapter else str(checkpoint_model or config.model_name)
        self.checkpoint_path = Path(checkpoint) if checkpoint is not None else None
        tokenizer_source = str(
            checkpoint / "tokenizer" if checkpoint and (checkpoint / "tokenizer").exists() else source
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=config.revision if source == config.model_name else None,
            trust_remote_code=config.trust_remote_code,
            padding_side="left" if _resolve_pooling(config) == "last_token" else "right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: dict[str, Any] = {
            "revision": config.revision if source == config.model_name else None,
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": self.dtype,
            "attn_implementation": "sdpa",
        }
        if config.use_qlora and source == config.model_name:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("QLoRA için bitsandbytes gereklidir") from exc
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.dtype,
            )
            load_kwargs["device_map"] = {"": 0}
        self.model = AutoModel.from_pretrained(source, **load_kwargs)
        if is_adapter and checkpoint_model is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Bi-encoder adapter yüklemek için peft gereklidir") from exc
            self.model = PeftModel.from_pretrained(self.model, checkpoint_model, is_trainable=trainable)
        elif config.use_lora and trainable and source == config.model_name:
            self._attach_lora()
        if config.gradient_checkpointing and trainable:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        if not config.use_qlora:
            self.model.to(device)
        hidden = int(getattr(self.model.config, "hidden_size", 0))
        if hidden < config.output_dimension:
            raise ValueError(f"Model hidden_size={hidden}, output_dimension={config.output_dimension}")

    def _attach_lora(self) -> None:
        """Attach LoRA/QLoRA to all linear backbone projections."""

        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("Bi-encoder LoRA için peft>=0.15 gereklidir") from exc
        if self.config.use_qlora:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=self.config.gradient_checkpointing
            )
        lora = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules="all-linear",
            bias="none",
        )
        self.model = get_peft_model(self.model, lora)
        if sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad) <= 0:
            raise RuntimeError("Bi-encoder LoRA trainable parametre üretmedi")

    def format_texts(self, texts: Sequence[str], role: str) -> list[str]:
        """Apply Qwen query instruction while leaving documents instruction-free."""

        if role not in {"query", "document"}:
            raise ValueError("role query veya document olmalıdır")
        clean = [str(value).replace("<|im_start|>", " ").replace("<|im_end|>", " ") for value in texts]
        if role == "query" and "qwen3-embedding" in self.config.model_name.lower():
            return [f"Instruct: {self.config.query_instruction}\nQuery:{text}" for text in clean]
        return clean

    def tokenize(self, texts: Sequence[str], role: str) -> Mapping[str, Any]:
        """Tokenize a query or document batch with dynamic padding."""

        if not texts:
            raise ValueError("Boş embedding batch")
        formatted = self.format_texts(texts, role)
        batch = self.tokenizer(
            formatted, padding=True, truncation=True,
            max_length=self.config.max_length, return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in batch.items()}

    def forward_embeddings(self, tokenized: Mapping[str, Any], dimension: int | None = None) -> Any:
        """Pool, truncate and L2-normalize model hidden states."""

        torch = self.torch
        output = self.model(**tokenized, return_dict=True)
        hidden = output.last_hidden_state
        attention = tokenized["attention_mask"]
        pooling = _resolve_pooling(self.config)
        if pooling == "last_token":
            left_padded = bool(torch.all(attention[:, -1] == 1))
            if left_padded:
                pooled = hidden[:, -1]
            else:
                sequence_lengths = attention.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), sequence_lengths]
        elif pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        dimension = dimension or self.config.output_dimension
        if dimension < 32 or dimension > pooled.shape[1]:
            raise ValueError(f"İstenen embedding dimension geçersiz: {dimension}")
        return torch.nn.functional.normalize(pooled[:, :dimension].float(), p=2, dim=1)

    def encode(
        self,
        texts: Sequence[str],
        role: str,
        *,
        batch_size: int = 32,
        minimum_batch_size: int = 1,
        dimension: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Encode texts with same-range OOM retry and return the successful batch size."""

        if batch_size < 1 or minimum_batch_size < 1 or minimum_batch_size > batch_size:
            raise ValueError("Embedding batch sınırları geçersiz")
        torch = self.torch
        self.model.eval()
        current = batch_size
        while True:
            try:
                outputs: list[np.ndarray] = []
                with torch.inference_mode():
                    for start in range(0, len(texts), current):
                        tokenized = self.tokenize(texts[start:start + current], role)
                        enabled = self.device.startswith("cuda") and self.dtype in {torch.float16, torch.bfloat16}
                        with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=enabled):
                            embeddings = self.forward_embeddings(tokenized, dimension)
                        outputs.append(embeddings.cpu().numpy().astype(np.float32))
                result = np.concatenate(outputs) if outputs else np.empty((0, dimension or self.config.output_dimension), dtype=np.float32)
                if len(result) != len(texts) or not np.isfinite(result).all():
                    raise RuntimeError("Embedding çıktısı eksik veya NaN/Inf")
                return result, current
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or current <= minimum_batch_size:
                    raise
                current = max(minimum_batch_size, current // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    def save(self, output_dir: Path) -> None:
        """Save encoder/adapter, tokenizer and configuration."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir / "model", safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir / "tokenizer")
        (output_dir / "biencoder_config.json").write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def multi_positive_infonce(
    torch: Any,
    query_embeddings: Any,
    document_embeddings: Any,
    positive_mask: Any,
    allowed_mask: Any,
    *,
    temperature: float,
) -> Any:
    """Average each positive's log-probability with masked false negatives."""

    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("InfoNCE embeddings 2D olmalıdır")
    if positive_mask.shape != allowed_mask.shape or positive_mask.shape != (
        query_embeddings.shape[0], document_embeddings.shape[0]
    ):
        raise ValueError("InfoNCE mask shape uyumsuz")
    if not torch.all(positive_mask.sum(dim=1) > 0):
        raise ValueError("Her query en az bir pozitif dokümana sahip olmalıdır")
    if torch.any(positive_mask & ~allowed_mask):
        raise ValueError("Pozitifler allowed_mask dışında bırakılamaz")
    scores = query_embeddings @ document_embeddings.T / temperature
    scores = scores.masked_fill(~allowed_mask, torch.finfo(scores.dtype).min)
    log_probability = torch.nn.functional.log_softmax(scores, dim=1)
    per_query = []
    for index in range(scores.shape[0]):
        per_query.append(-log_probability[index][positive_mask[index]].mean())
    return torch.stack(per_query).mean()


def matryoshka_multi_positive_loss(
    torch: Any,
    query_embeddings: Any,
    document_embeddings: Any,
    positive_mask: Any,
    allowed_mask: Any,
    dimensions: Sequence[int],
    temperature: float,
) -> Any:
    """Average multi-positive InfoNCE across normalized Matryoshka slices."""

    losses = []
    for dimension in dimensions:
        if dimension > query_embeddings.shape[1] or dimension > document_embeddings.shape[1]:
            raise ValueError(f"Matryoshka dimension={dimension} hidden boyuttan büyük")
        query_slice = torch.nn.functional.normalize(query_embeddings[:, :dimension], p=2, dim=1)
        document_slice = torch.nn.functional.normalize(document_embeddings[:, :dimension], p=2, dim=1)
        losses.append(multi_positive_infonce(
            torch, query_slice, document_slice, positive_mask, allowed_mask,
            temperature=temperature,
        ))
    return torch.stack(losses).mean()


def _contrastive_batch(
    frame: pd.DataFrame,
    indices: Sequence[int],
    known_positive_items: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Build unique-query texts, documents and multi-positive/allowed masks."""

    batch = frame.iloc[list(indices)].reset_index(drop=True)
    term_ids = list(dict.fromkeys(batch["term_id"].astype(str)))
    query_texts = [str(batch.loc[batch["term_id"].astype(str).eq(term_id), "query"].iloc[0]) for term_id in term_ids]
    item_ids = batch["item_id"].astype(str).tolist()
    positive_mask = np.zeros((len(term_ids), len(batch)), dtype=bool)
    allowed_mask = np.ones_like(positive_mask)
    suspect = batch.get("triage_status", pd.Series("train", index=batch.index)).astype(str).ne("train").to_numpy()
    for row, term_id in enumerate(term_ids):
        positive_mask[row] = np.asarray([item_id in known_positive_items[term_id] for item_id in item_ids])
        same_query = batch["term_id"].astype(str).eq(term_id).to_numpy()
        allowed_mask[row, same_query & suspect] = False
        allowed_mask[row, positive_mask[row]] = True
    if not positive_mask.any(axis=1).all():
        raise ValueError("Contrastive batch pozitif coverage eksik")
    return {
        "queries": query_texts,
        "documents": batch["product_text"].fillna("").astype(str).tolist(),
        "positive_mask": positive_mask,
        "allowed_mask": allowed_mask,
    }


def validate_biencoder_training_frame(frame: pd.DataFrame) -> None:
    """Validate enriched contrastive rows and group label coverage."""

    required = {"term_id", "item_id", "query", "product_text", "label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Bi-encoder train kolonları eksik: {sorted(missing)}")
    if frame.duplicated(["term_id", "item_id"]).any():
        raise ValueError("Bi-encoder duplicate term-item içeriyor")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    grouped = frame.assign(_label=labels).groupby("term_id")["_label"].agg(["min", "max"])
    if not ((grouped["min"] == 0) & (grouped["max"] == 1)).all():
        raise ValueError("Her bi-encoder query grubu pozitif ve negatif içermeli")


def train_biencoder(
    frame: pd.DataFrame,
    output_dir: Path,
    model_config: BiEncoderConfig,
    train_config: TrainConfig,
    *,
    device: str = "cuda",
    resume_checkpoint: Path | None = None,
) -> Path:
    """Fine-tune a shared encoder with query-uniform multi-positive InfoNCE."""

    model_config.validate()
    train_config.validate()
    validate_biencoder_training_frame(frame)
    set_global_seed(train_config.seed)
    stack = require_torch_stack()
    torch = stack["torch"]
    adapter = BiEncoderAdapter(
        model_config, device=device, checkpoint=resume_checkpoint, trainable=True
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapter.model.parameters() if parameter.requires_grad],
        lr=train_config.learning_rate, weight_decay=train_config.weight_decay,
    )
    sampler = QueryUniformBatchSampler(frame, train_config)
    accumulation = max(1, math.ceil(train_config.effective_batch_size / train_config.micro_batch_size))
    total_steps = max(1, math.ceil(len(sampler) / accumulation) * train_config.epochs)
    scheduler = stack["get_cosine_schedule_with_warmup"](
        optimizer,
        num_warmup_steps=int(total_steps * train_config.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=adapter.dtype == torch.float16 and device.startswith("cuda")
    )
    start_epoch = 0
    if resume_checkpoint and (resume_checkpoint / "trainer_state.pt").exists():
        state = torch.load(
            resume_checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
    known = frame.loc[frame["label"].eq(1)].groupby("term_id")["item_id"].agg(
        lambda values: set(values.astype(str))
    ).to_dict()
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, train_config.epochs):
        sampler.set_epoch(epoch)
        adapter.model.train()
        losses: list[float] = []
        for batch_index, indices in enumerate(sampler):
            raw = _contrastive_batch(frame, indices, known)
            query_tokens = adapter.tokenize(raw["queries"], "query")
            document_tokens = adapter.tokenize(raw["documents"], "document")
            enabled = device.startswith("cuda") and adapter.dtype in {torch.float16, torch.bfloat16}
            with torch.autocast(device_type="cuda", dtype=adapter.dtype, enabled=enabled):
                query_embeddings = adapter.forward_embeddings(query_tokens, model_config.output_dimension)
                document_embeddings = adapter.forward_embeddings(document_tokens, model_config.output_dimension)
                loss = matryoshka_multi_positive_loss(
                    torch, query_embeddings, document_embeddings,
                    torch.as_tensor(raw["positive_mask"], device=device),
                    torch.as_tensor(raw["allowed_mask"], device=device),
                    model_config.matryoshka_dimensions, model_config.temperature,
                )
            scaler.scale(loss / accumulation).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(sampler):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), train_config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
        checkpoint = output_dir / f"checkpoint-epoch-{epoch:02d}"
        adapter.save(checkpoint)
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }, checkpoint / "trainer_state.pt")
        (checkpoint / "checkpoint_manifest.json").write_text(
            json.dumps({
                "model_config": asdict(model_config),
                "train_resume_signature": _resume_signature(train_config),
                "ordered_pair_hash": _ordered_pair_hash(frame),
                "epoch": epoch,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        history.append({"epoch": epoch, "contrastive_loss": float(np.mean(losses)), "batches": len(losses)})
        pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
    final = output_dir / "final"
    adapter.save(final)
    (final / "run_manifest.json").write_text(
        json.dumps({
            "model_config": asdict(model_config), "train_config": asdict(train_config),
            "rows": len(frame), "terms": int(frame["term_id"].nunique()),
            "ordered_pair_hash": _ordered_pair_hash(frame), "completed_at": time.time(),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final


def train_biencoder_with_oom_fallback(
    frame: pd.DataFrame,
    output_dir: Path,
    model_config: BiEncoderConfig,
    train_config: TrainConfig,
    *,
    device: str = "cuda",
) -> Path:
    """Retry a deterministic bi-encoder run with fewer query groups after OOM."""

    current = train_config
    output_dir = Path(output_dir)
    final_manifest = output_dir / "final" / "run_manifest.json"
    if final_manifest.exists():
        existing = json.loads(final_manifest.read_text(encoding="utf-8"))
        if (
            existing.get("model_config") == asdict(model_config)
            and existing.get("train_config") == asdict(train_config)
            and existing.get("ordered_pair_hash") == _ordered_pair_hash(frame)
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
                and manifest.get("ordered_pair_hash") == _ordered_pair_hash(frame)
            ):
                compatible.append(checkpoint)
        if all_checkpoints and not compatible:
            raise ValueError(
                f"{output_dir} checkpoint'leri farklı bi-encoder config/verisine ait; "
                "yeni output_dir kullanın"
            )
        resume = compatible[-1] if compatible else None
        try:
            return train_biencoder(
                frame, output_dir, model_config, current,
                device=device, resume_checkpoint=resume,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current.micro_batch_size <= 1:
                raise
            current = replace(current, micro_batch_size=max(1, current.micro_batch_size // 2))
            gc.collect()
            try:
                torch = require_torch_stack()["torch"]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError:
                raise exc
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "oom_fallback.json").write_text(
                json.dumps({"new_micro_batch_size": current.micro_batch_size, "error": str(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )


def _ordered_pair_hash(frame: pd.DataFrame) -> str:
    """Hash ordered term/item pairs for training-resume cache safety."""

    if not {"term_id", "item_id"}.issubset(frame.columns):
        raise ValueError("Training hash term_id,item_id gerektirir")
    values = pd.util.hash_pandas_object(
        frame[["term_id", "item_id"]].astype("string"), index=False
    ).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def _resume_signature(config: TrainConfig) -> dict[str, Any]:
    """Return resume-critical settings while allowing OOM microbatch changes."""

    payload = asdict(config)
    payload.pop("micro_batch_size", None)
    return payload


def _ordered_input_hash(frame: pd.DataFrame, id_column: str, text_column: str) -> str:
    """Hash ordered cache IDs and text content."""

    if not {id_column, text_column}.issubset(frame.columns):
        raise ValueError("Embedding cache id/text kolonları eksik")
    values = pd.util.hash_pandas_object(frame[[id_column, text_column]].astype("string"), index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def _model_fingerprint(adapter: BiEncoderAdapter) -> str:
    """Fingerprint encoder config, parameter count and local checkpoint files."""

    count = sum(parameter.numel() for parameter in adapter.model.parameters())
    file_metadata = []
    if adapter.checkpoint_path and adapter.checkpoint_path.exists():
        for path in sorted(adapter.checkpoint_path.rglob("*")):
            if path.is_file():
                stat = path.stat()
                file_metadata.append((
                    str(path.relative_to(adapter.checkpoint_path)), stat.st_size, stat.st_mtime_ns
                ))
    payload = json.dumps({
        "config": asdict(adapter.config), "parameters": count, "files": file_metadata,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def encode_to_memmap_cache(
    frame: pd.DataFrame,
    adapter: BiEncoderAdapter,
    output_dir: Path,
    *,
    id_column: str,
    text_column: str,
    role: Literal["query", "document"],
    batch_size: int = 32,
    chunk_size: int = 20_000,
) -> tuple[Path, Path]:
    """Encode ordered rows into restart-safe float16 memmap shards and an ID map."""

    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError(f"Embedding cache {id_column} boş veya duplicate")
    if chunk_size < 1:
        raise ValueError("chunk_size pozitif olmalıdır")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dimension = adapter.config.output_dimension
    data_path = output_dir / f"{role}_embeddings.f16.npy"
    map_path = output_dir / f"{role}_map.parquet"
    meta_path = output_dir / f"{role}_embeddings.json"
    spec = EmbeddingCacheSpec(
        role=role, id_column=id_column, text_column=text_column, rows=len(frame), dimension=dimension,
        ordered_input_hash=_ordered_input_hash(frame, id_column, text_column),
        model_fingerprint=_model_fingerprint(adapter),
    )
    if data_path.exists() and map_path.exists() and meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in asdict(spec).items()):
            cached = np.load(data_path, mmap_mode="r")
            if cached.shape == (len(frame), dimension):
                return data_path, map_path
    temp_path = data_path.with_suffix(".tmp.npy")
    memmap = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float16, shape=(len(frame), dimension))
    used_batches: list[int] = []
    for start in range(0, len(frame), chunk_size):
        texts = frame[text_column].iloc[start:start + chunk_size].fillna("").astype(str).tolist()
        values, used = adapter.encode(texts, role, batch_size=batch_size, dimension=dimension)
        memmap[start:start + len(values)] = values.astype(np.float16)
        memmap.flush()
        used_batches.append(used)
    del memmap
    os.replace(temp_path, data_path)
    mapping = frame[[id_column]].copy().reset_index(drop=True)
    mapping.insert(0, "embedding_row", np.arange(len(mapping), dtype=np.int64))
    mapping["text_hash"] = frame[text_column].fillna("").astype(str).map(
        lambda value: hashlib.blake2b(value.encode(), digest_size=12).hexdigest()
    )
    mapping.to_parquet(map_path, index=False)
    meta_path.write_text(
        json.dumps({**asdict(spec), "minimum_used_batch_size": min(used_batches or [batch_size])}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return data_path, map_path


def mine_faiss_hard_negatives(
    query_embeddings_path: Path,
    query_map_path: Path,
    item_embeddings_path: Path,
    item_map_path: Path,
    positives: pd.DataFrame,
    output_path: Path,
    *,
    search_dimension: int = 256,
    top_k: int = 500,
    output_per_query: int = 100,
    query_batch_size: int = 256,
    miner_model_hash: str = "base_biencoder",
    miner_train_term_hash: str = "catalog_only",
) -> pd.DataFrame:
    """Mine normalized inner-product candidates, excluding every known positive."""

    if search_dimension < 32 or top_k < output_per_query or output_per_query < 1:
        raise ValueError("FAISS dimension/top_k/output_per_query geçersiz")
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Hard-negative mining için faiss gereklidir") from exc
    query_embeddings = np.load(query_embeddings_path, mmap_mode="r")
    item_embeddings = np.load(item_embeddings_path, mmap_mode="r")
    queries = pd.read_parquet(query_map_path)
    items = pd.read_parquet(item_map_path)
    if query_embeddings.shape[0] != len(queries) or item_embeddings.shape[0] != len(items):
        raise ValueError("Embedding cache ve ID map satır sayıları uyuşmuyor")
    if search_dimension > query_embeddings.shape[1] or search_dimension > item_embeddings.shape[1]:
        raise ValueError("search_dimension embedding boyutundan büyük")
    query_id_col = next((column for column in ("term_id", "query_id") if column in queries), None)
    item_id_col = next((column for column in ("item_id", "product_id") if column in items), None)
    if query_id_col is None or item_id_col is None:
        raise ValueError("Query/item map canonical ID kolonu içermiyor")
    known = positives.groupby("term_id")["item_id"].agg(lambda values: set(values.astype(str))).to_dict()
    item_matrix = np.asarray(item_embeddings[:, :search_dimension], dtype=np.float32)
    norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
    item_matrix /= np.maximum(norms, 1e-12)
    index = faiss.IndexFlatIP(search_dimension)
    index.add(item_matrix)
    records: list[dict[str, Any]] = []
    for start in range(0, len(queries), query_batch_size):
        query_matrix = np.asarray(query_embeddings[start:start + query_batch_size, :search_dimension], dtype=np.float32)
        query_matrix /= np.maximum(np.linalg.norm(query_matrix, axis=1, keepdims=True), 1e-12)
        scores, positions = index.search(query_matrix, min(top_k, len(items)))
        for local_row, (row_scores, row_positions) in enumerate(zip(scores, positions)):
            term_id = str(queries.iloc[start + local_row][query_id_col])
            selected = 0
            for rank, (score, position) in enumerate(zip(row_scores, row_positions), start=1):
                if position < 0:
                    continue
                item_id = str(items.iloc[int(position)][item_id_col])
                if item_id in known.get(term_id, set()):
                    continue
                records.append({
                    "term_id": term_id, "item_id": item_id,
                    "negative_type": "bi_encoder_high_score",
                    "bi_encoder_score": np.float32(score),
                    "hardness_score": np.float32(np.clip((float(score) + 1.0) / 2.0, 0.0, 1.0)),
                    "retrieval_rank": np.int32(rank),
                    "miner_model_hash": miner_model_hash,
                    "miner_train_term_hash": miner_train_term_hash,
                })
                selected += 1
                if selected >= output_per_query:
                    break
    result = pd.DataFrame.from_records(records)
    if result.empty or result.duplicated(["term_id", "item_id"]).any():
        raise RuntimeError("FAISS miner boş veya duplicate çıktı üretti")
    positive_index = pd.MultiIndex.from_frame(positives[["term_id", "item_id"]].astype(str))
    result_index = pd.MultiIndex.from_frame(result[["term_id", "item_id"]].astype(str))
    if result_index.isin(positive_index).any():
        raise RuntimeError("FAISS miner known-positive collision üretti")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    """Parse bi-encoder train, encode or mine commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "encode", "mine"], required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3/biencoder"))
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--role", choices=["query", "document"])
    parser.add_argument("--id-column")
    parser.add_argument("--text-column")
    parser.add_argument("--query-embeddings", type=Path)
    parser.add_argument("--query-map", type=Path)
    parser.add_argument("--item-embeddings", type=Path)
    parser.add_argument("--item-map", type=Path)
    parser.add_argument("--positives", type=Path)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Execute the selected bi-encoder workflow."""

    args = parse_args()
    config = BiEncoderConfig(
        model_name=args.model_name, output_dimension=args.dimension,
        matryoshka_dimensions=tuple(dict.fromkeys(
            value for value in (args.dimension, 512, 256) if value <= args.dimension
        )),
        use_lora=args.lora, use_qlora=args.qlora,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        if args.input is None:
            raise ValueError("train modunda --input enriched Parquet zorunludur")
        frame = pd.read_parquet(args.input)
        train_config = TrainConfig(
            epochs=args.epochs, micro_batch_size=args.batch_size,
            queries_per_batch=args.batch_size, effective_batch_size=max(args.batch_size, 16),
            label_smoothing=0.0, pairwise_weight=0.0, seed=args.seed,
        )
        print(train_biencoder_with_oom_fallback(frame, args.output_dir, config, train_config, device=args.device))
        return
    if args.mode == "mine":
        required_paths = [args.query_embeddings, args.query_map, args.item_embeddings, args.item_map, args.positives]
        if any(path is None for path in required_paths):
            raise ValueError("mine modu query/item embedding+map ve positives yollarını gerektirir")
        positives = pd.read_csv(args.positives, dtype="string") if args.positives.suffix == ".csv" else pd.read_parquet(args.positives)
        result = mine_faiss_hard_negatives(
            args.query_embeddings, args.query_map, args.item_embeddings, args.item_map,
            positives, args.output_dir / "bi_encoder_hard_negatives.parquet",
        )
        print(f"Mined rows: {len(result):,}")
        return
    if args.input is None or args.role is None or not args.id_column or not args.text_column:
        raise ValueError("encode modu --input --role --id-column --text-column gerektirir")
    frame = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input, dtype="string")
    adapter = BiEncoderAdapter(config, device=args.device, checkpoint=args.model_path, trainable=False)
    paths = encode_to_memmap_cache(
        frame, adapter, args.output_dir, id_column=args.id_column,
        text_column=args.text_column, role=args.role, batch_size=args.batch_size,
    )
    print(paths)


if __name__ == "__main__":
    main()
