#!/usr/bin/env python3
"""Train and shard-infer BGE/XLM-R cross-encoders on Trendyol pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import experiment_dir, stable_term_folds, validate_probability_frame, write_config
from text_features import normalize_category_path, normalize_text, select_attributes

ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
TEXT_COLUMNS = ITEM_COLUMNS[1:]
TRAIN_REQUIRED = {"term_id", "item_id", "query", "label", "sample_weight"}


def require_deep_stack():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback,
            Trainer, TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("GPU bağımlılıkları gerekli: pip install -r requirements-gpu.txt") from exc
    return (
        torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer,
        EarlyStoppingCallback, Trainer, TrainingArguments,
    )


def product_text(frame: pd.DataFrame) -> pd.Series:
    attributes = [select_attributes(a, q) for a, q in zip(frame["attributes"], frame["query"])]
    return (
        "[T] " + frame["title"].fillna("").map(normalize_text)
        + " [C] " + frame["category"].fillna("").map(normalize_category_path)
        + " [B] " + frame["brand"].fillna("").map(normalize_text)
        + " [G] " + frame["gender"].fillna("").map(normalize_text)
        + " [AGE] " + frame["age_group"].fillna("").map(normalize_text)
        + " [A] " + pd.Series(attributes, index=frame.index)
    )


def refresh_catalog_text(frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Restore raw structured fields lost during lexical mining normalization."""
    items_path, terms_path = data_dir / "items.csv", data_dir / "terms.csv"
    if not items_path.exists() or not terms_path.exists():
        raise FileNotFoundError("Raw items.csv ve terms.csv eğitim sırasında gereklidir")
    result = frame.copy()
    result["item_id"] = result["item_id"].astype("string")
    result["term_id"] = result["term_id"].astype("string")
    items = pd.read_csv(items_path, usecols=ITEM_COLUMNS, dtype="string").rename(
        columns={column: f"_raw_{column}" for column in TEXT_COLUMNS}
    )
    terms = pd.read_csv(terms_path, usecols=["term_id", "query"], dtype="string").rename(
        columns={"query": "_raw_query"}
    )
    result = result.merge(
        items, on="item_id", how="left", validate="many_to_one", sort=False,
        indicator="_item_merge",
    )
    if result["_item_merge"].ne("both").any():
        raise ValueError("items.csv içinde bulunamayan item_id var")
    result = result.drop(columns="_item_merge")
    result = result.merge(
        terms, on="term_id", how="left", validate="many_to_one", sort=False,
        indicator="_term_merge",
    )
    if result["_term_merge"].ne("both").any():
        raise ValueError("terms.csv içinde bulunamayan term_id var")
    result = result.drop(columns="_term_merge")
    for column in TEXT_COLUMNS:
        raw = f"_raw_{column}"
        result[column] = result.pop(raw).fillna("")
    result["query"] = result.pop("_raw_query").fillna("")
    return result


def validate_training_frame(frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    missing = TRAIN_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Eğitim kolonları eksik: {sorted(missing)}")
    frame = frame.copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int8")
    frame["sample_weight"] = pd.to_numeric(frame["sample_weight"], errors="raise").astype("float32")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label yalnızca 0/1 olmalıdır")
    if not np.isfinite(frame["sample_weight"]).all() or frame["sample_weight"].le(0).any():
        raise ValueError("sample_weight sonlu ve pozitif olmalıdır")
    if frame.duplicated(["term_id", "item_id"]).any():
        raise ValueError("Duplicate (term_id,item_id) eğitim çifti bulundu")
    known_path = data_dir / "training_pairs.csv"
    if known_path.exists():
        known = pd.read_csv(known_path, usecols=["term_id", "item_id"], dtype="string")
        known_index = pd.MultiIndex.from_frame(known)
        negative_index = pd.MultiIndex.from_frame(frame.loc[frame["label"].eq(0), ["term_id", "item_id"]])
        collisions = int(negative_index.isin(known_index).sum())
        if collisions:
            raise ValueError(f"Bilinen pozitif-negatif collision bulundu: {collisions:,}")
    return frame


def make_dataset_class(torch, Dataset):
    class PairDataset(Dataset):
        def __init__(self, frame, tokenizer, max_length, labelled=True):
            self.query = frame["query"].fillna("").map(normalize_text).tolist()
            self.product = product_text(frame).tolist()
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.labels = frame["label"].astype(float).to_numpy() if labelled else None
            self.weights = frame.get("sample_weight", pd.Series(1.0, index=frame.index)).astype(float).to_numpy()
        def __len__(self):
            return len(self.query)
        def __getitem__(self, index):
            encoded = self.tokenizer(
                self.query[index], self.product[index], truncation="only_second",
                max_length=self.max_length,
            )
            if self.labels is not None:
                encoded["labels"] = self.labels[index]
                encoded["sample_weight"] = self.weights[index]
            return encoded
    return PairDataset


def make_weighted_trainer(Trainer, torch):
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("sample_weight").float()
            labels = inputs.pop("labels").float()
            outputs = model(**inputs)
            logits = outputs.logits.view(-1)
            per_example = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.view(-1), reduction="none"
            )
            loss = (per_example * weights.view(-1)).sum() / weights.sum().clamp_min(1e-6)
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


def train(args, out: Path):
    (
        torch, _, Dataset, AutoModel, AutoTokenizer, EarlyStoppingCallback,
        Trainer, TrainingArguments,
    ) = require_deep_stack()
    frame = pd.read_parquet(args.train)
    frame = validate_training_frame(frame, args.data_dir)
    if args.refresh_catalog_text:
        frame = refresh_catalog_text(frame, args.data_dir)
    elif missing := set(TEXT_COLUMNS) - set(frame.columns):
        raise ValueError(
            f"--no-refresh-catalog-text için parquet içinde alanlar eksik: {sorted(missing)}"
        )
    if args.max_train_rows and len(frame) > args.max_train_rows:
        frame = frame.sample(args.max_train_rows, random_state=args.seed).reset_index(drop=True)
    if "split" in frame.columns:
        invalid_splits = set(frame["split"].dropna().astype(str).unique()) - {"train", "valid"}
        if invalid_splits:
            raise ValueError(f"Geçersiz split değerleri: {sorted(invalid_splits)}")
        validation_mask = frame["split"].astype(str).eq("valid")
    else:
        folds = stable_term_folds(frame["term_id"], 5, args.seed).set_index("term_id")["fold"]
        validation_mask = frame["term_id"].map(folds).eq(0)
    if not validation_mask.any() or validation_mask.all():
        raise ValueError("Train ve validation split'leri ikisi de dolu olmalıdır")
    if frame.loc[validation_mask, "term_id"].isin(frame.loc[~validation_mask, "term_id"]).any():
        raise RuntimeError("Validation term_id eğitim fold'una sızdı")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(
        args.model_name, num_labels=1, ignore_mismatched_sizes=True,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
    PairDataset = make_dataset_class(torch, Dataset)
    train_dataset = PairDataset(frame[~validation_mask].reset_index(drop=True), tokenizer, args.max_length)
    valid_dataset = PairDataset(frame[validation_mask].reset_index(drop=True), tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=max(1, args.effective_batch_size // args.batch_size),
        learning_rate=args.learning_rate, weight_decay=0.01, warmup_ratio=0.05,
        lr_scheduler_type="cosine", max_grad_norm=1.0,
        bf16=args.bf16, fp16=args.fp16, eval_strategy="steps", save_strategy="steps",
        eval_steps=args.eval_steps, save_steps=args.eval_steps, save_total_limit=2,
        logging_steps=50, load_best_model_at_end=True, metric_for_best_model="eval_macro_f1",
        greater_is_better=True, remove_unused_columns=False, report_to="none", seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        dataloader_pin_memory=torch.cuda.is_available(),
        tf32=args.tf32, optim=args.optim,
    )
    WeightedTrainer = make_weighted_trainer(Trainer, torch)

    def compute_metrics(evaluation):
        """Select a validation threshold and report PU-aware Macro-F1."""

        logits = np.asarray(evaluation.predictions).reshape(-1)
        labels = np.asarray(evaluation.label_ids).reshape(-1).astype(np.int8)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        best = (-1.0, 0.5, 0.0)
        for threshold in np.linspace(0.02, 0.98, 193):
            pred = probability >= threshold
            tp = np.sum(pred & (labels == 1)); fp = np.sum(pred & (labels == 0))
            fn = np.sum(~pred & (labels == 1)); tn = np.sum(~pred & (labels == 0))
            f1_pos = 2 * tp / max(1, 2 * tp + fp + fn)
            f1_neg = 2 * tn / max(1, 2 * tn + fp + fn)
            macro = float((f1_pos + f1_neg) / 2)
            if macro > best[0]:
                best = (macro, float(threshold), float(pred.mean()))
        return {"macro_f1": best[0], "best_threshold": best[1], "positive_rate": best[2]}
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    trainer = WeightedTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=valid_dataset, processing_class=tokenizer, callbacks=callbacks,
        compute_metrics=compute_metrics,
    )
    resume = args.resume_checkpoint
    if resume == "auto":
        checkpoints = sorted(
            (out / "checkpoints").glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        resume = str(checkpoints[-1]) if checkpoints else None
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(out / "model")
    tokenizer.save_pretrained(out / "model")
    validation_prediction = trainer.predict(valid_dataset)
    validation_logits = np.asarray(validation_prediction.predictions).reshape(-1)
    validation_frame = frame.loc[validation_mask, ["term_id", "item_id", "label"]].reset_index(drop=True)
    validation_frame["raw_logit"] = validation_logits.astype(np.float32)
    validation_frame["probability"] = (
        1.0 / (1.0 + np.exp(-np.clip(validation_logits, -30, 30)))
    ).astype(np.float32)
    validation_frame.to_parquet(out / "validation_probabilities.parquet", index=False)
    summary = {
        "rows": len(frame),
        "train_rows": len(train_dataset),
        "validation_rows": len(valid_dataset),
        "train_terms": int(frame.loc[~validation_mask, "term_id"].nunique()),
        "validation_terms": int(frame.loc[validation_mask, "term_id"].nunique()),
        "positive_rows": int(frame["label"].sum()),
        "negative_rows": int(frame["label"].eq(0).sum()),
        "best_eval_macro_f1": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "validation_macro_f1": validation_prediction.metrics.get("test_macro_f1"),
        "validation_threshold": validation_prediction.metrics.get("test_best_threshold"),
        "validation_positive_rate": validation_prediction.metrics.get("test_positive_rate"),
        "train_runtime": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
    }
    (out / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    )


def infer(args, out: Path):
    torch, DataLoader, Dataset, AutoModel, AutoTokenizer, _, _, _ = require_deep_stack()
    model_path = args.model_path or out / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(args.device)
    model.eval()
    PairDataset = make_dataset_class(torch, Dataset)
    if args.pairs:
        pairs = pd.read_parquet(args.pairs) if args.pairs.suffix == ".parquet" else pd.read_csv(args.pairs, dtype="string")
        pairs = pairs[["term_id", "item_id"]].drop_duplicates().reset_index(drop=True)
        pairs.insert(0, "id", pd.Series([f"PAIR_{index:09d}" for index in range(len(pairs))], dtype="string"))
    else:
        pairs = pd.read_csv(args.data_dir / "submission_pairs.csv", dtype="string")
    terms = pd.read_csv(args.data_dir / "terms.csv", usecols=["term_id", "query"], dtype="string")
    items = pd.read_csv(args.data_dir / "items.csv", usecols=ITEM_COLUMNS, dtype="string")
    shard_dir = out / "probability_shards"
    shard_dir.mkdir(exist_ok=True)
    shard_paths = []
    for shard, start in enumerate(range(0, len(pairs), args.shard_size)):
        path = shard_dir / f"part-{shard:05d}.parquet"
        shard_paths.append(path)
        if path.exists():
            continue
        chunk = pairs.iloc[start:start + args.shard_size].merge(terms, on="term_id", validate="many_to_one").merge(items, on="item_id", validate="many_to_one")
        dataset = PairDataset(chunk, tokenizer, args.max_length, labelled=False)
        loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=lambda rows: tokenizer.pad(rows, return_tensors="pt"))
        logits = []
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(args.device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16 and args.device.startswith("cuda")):
                    logits.append(model(**batch).logits.view(-1).float().cpu().numpy())
        raw = np.concatenate(logits)
        result = chunk[["id", "term_id", "item_id"]].copy()
        result["raw_logit"] = raw.astype(np.float32)
        result["probability"] = (1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))).astype(np.float32)
        result.to_parquet(path, index=False)
    scores = pd.concat([pd.read_parquet(path) for path in shard_paths], ignore_index=True)
    validate_probability_frame(scores, len(pairs))
    scores.to_parquet(out / "probabilities.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "train-infer"], default="train-infer")
    parser.add_argument("--model-name", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--pairs", type=Path, help="Optional term_id,item_id CSV/Parquet to score instead of submission_pairs")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--experiment-id", default="bge_cross_encoder")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--refresh-catalog-text", action=argparse.BooleanOptionalAction, default=True,
        help="items.csv/terms.csv alanlarını yeniden birleştir; hazır zengin parquet için kapatılabilir",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=1.5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-rows", type=int, default=0, help="Yalnız smoke test için; 0=tam veri")
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if "train" in args.mode and args.train is None:
        parser.error("--train eğitim modunda zorunludur")
    out = experiment_dir(args.artifacts_dir, args.experiment_id)
    write_config(out / "config.json", vars(args))
    if args.mode in {"train", "train-infer"}:
        train(args, out)
    if args.mode in {"infer", "train-infer"}:
        infer(args, out)


if __name__ == "__main__":
    main()
