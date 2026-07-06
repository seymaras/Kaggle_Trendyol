#!/usr/bin/env python3
"""Train and shard-infer BGE/XLM-R cross-encoders on Trendyol pairs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import experiment_dir, stable_term_folds, validate_probability_frame, write_config
from text_features import normalize_category_path, normalize_text, select_attributes

ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]


def require_deep_stack():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("GPU bağımlılıkları gerekli: pip install -r requirements-gpu.txt") from exc
    return torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def pair_text(frame: pd.DataFrame) -> pd.Series:
    attributes = [select_attributes(a, q) for a, q in zip(frame["attributes"], frame["query"])]
    return (
        "[Q] " + frame["query"].fillna("").map(normalize_text)
        + " [T] " + frame["title"].fillna("").map(normalize_text)
        + " [C] " + frame["category"].fillna("").map(normalize_category_path)
        + " [B] " + frame["brand"].fillna("").map(normalize_text)
        + " [A] " + pd.Series(attributes, index=frame.index)
    )


def make_dataset_class(torch, Dataset):
    class PairDataset(Dataset):
        def __init__(self, frame, tokenizer, max_length, labelled=True):
            self.text = pair_text(frame).tolist()
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.labels = frame["label"].astype(float).to_numpy() if labelled else None
            self.weights = frame.get("sample_weight", pd.Series(1.0, index=frame.index)).astype(float).to_numpy()
        def __len__(self):
            return len(self.text)
        def __getitem__(self, index):
            encoded = self.tokenizer(self.text[index], truncation=True, max_length=self.max_length)
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
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.view(-1), weight=weights.view(-1))
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


def train(args, out: Path):
    torch, _, Dataset, AutoModel, AutoTokenizer, Trainer, TrainingArguments = require_deep_stack()
    frame = pd.read_parquet(args.train)
    folds = stable_term_folds(frame["term_id"], 5, args.seed).set_index("term_id")["fold"]
    validation_mask = frame["term_id"].map(folds).eq(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, num_labels=1, ignore_mismatched_sizes=True)
    PairDataset = make_dataset_class(torch, Dataset)
    train_dataset = PairDataset(frame[~validation_mask].reset_index(drop=True), tokenizer, args.max_length)
    valid_dataset = PairDataset(frame[validation_mask].reset_index(drop=True), tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=max(1, args.effective_batch_size // args.batch_size),
        learning_rate=args.learning_rate, weight_decay=0.01, warmup_ratio=0.05,
        bf16=args.bf16, fp16=args.fp16, eval_strategy="steps", save_strategy="steps",
        eval_steps=args.eval_steps, save_steps=args.eval_steps, save_total_limit=2,
        logging_steps=50, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        remove_unused_columns=False, report_to="none", seed=args.seed,
    )
    WeightedTrainer = make_weighted_trainer(Trainer, torch)
    trainer = WeightedTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset, tokenizer=tokenizer)
    trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    trainer.save_model(out / "model")
    tokenizer.save_pretrained(out / "model")


def infer(args, out: Path):
    torch, DataLoader, Dataset, AutoModel, AutoTokenizer, _, _ = require_deep_stack()
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=1.5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-steps", type=int, default=1000)
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
