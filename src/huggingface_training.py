from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import validate_split_frame


MODEL_NAME = "FacebookAI/xlm-roberta-base"
OUTPUT_DIR = Path("results/huggingface/xlm_roberta_lora")
MERGED_DIR = Path("results/huggingface/xlm_roberta_merged")
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_ID = {label: index for index, label in ID_TO_LABEL.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for --train. Use the dry-run or install CUDA "
            "on a GPU machine."
        )
    return torch.device("cuda")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-RoBERTa with LoRA on Hinglish sentiment data."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val-path", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--merged-dir", type=Path, default=MERGED_DIR)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run GPU training. Without this flag, validate and exit.",
    )
    args = parser.parse_args(argv)
    args.dry_run = not args.train
    return args


def build_dataset(train_path: Path, val_path: Path) -> tuple[Dataset, Dataset]:
    train_frame = pd.read_csv(train_path, keep_default_na=False)
    val_frame = pd.read_csv(val_path, keep_default_na=False)
    validate_split_frame(train_frame, "train")
    validate_split_frame(val_frame, "validation")

    train_frame = train_frame[["uid", "text_clean", "label"]].copy()
    val_frame = val_frame[["uid", "text_clean", "label"]].copy()
    train_frame["text_clean"] = train_frame["text_clean"].astype(str)
    val_frame["text_clean"] = val_frame["text_clean"].astype(str)
    train_frame["label"] = train_frame["label"].astype(int)
    val_frame["label"] = val_frame["label"].astype(int)
    return Dataset.from_pandas(train_frame, preserve_index=False), Dataset.from_pandas(
        val_frame, preserve_index=False
    )


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["query", "key", "value", "dense"],
    )


def build_lora_model(
    model_name: str,
    num_labels: int,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model = get_peft_model(base_model, build_lora_config())
    model.print_trainable_parameters()
    return tokenizer, model


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions = np.asarray(eval_prediction.predictions)
    labels = np.asarray(eval_prediction.label_ids)
    predicted_labels = np.argmax(predictions, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "macro_f1": float(f1_score(labels, predicted_labels, average="macro")),
        "weighted_f1": float(
            f1_score(labels, predicted_labels, average="weighted")
        ),
    }


def merge_and_export(adapter_dir: Path, merged_dir: Path) -> Path:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")
    base_model_name = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    ).get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"Missing base_model_name_or_path in {adapter_dir}")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        str(base_model_name),
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_name))
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir


def train(args: argparse.Namespace) -> dict[str, object]:
    device = require_cuda()
    set_seed(args.seed)
    train_dataset, val_dataset = build_dataset(args.train_path, args.val_path)
    tokenizer, model = build_lora_model(args.model_name, num_labels=3)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return tokenizer(
            batch["text_clean"],
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    train_dataset = train_dataset.remove_columns(["uid", "text_clean"])
    val_dataset = val_dataset.remove_columns(["uid", "text_clean"])
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=device.type == "cuda",
        logging_steps=25,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    merged_dir = merge_and_export(
        args.output_dir,
        args.merged_dir,
    )
    return {
        "adapter_dir": str(args.output_dir),
        "merged_dir": str(merged_dir),
        "metrics": metrics,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    build_dataset(args.train_path, args.val_path)
    if args.dry_run:
        print(json.dumps({
            "status": "dry-run",
            "model": args.model_name,
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "device": "CUDA required for --train",
        }, indent=2))
        return {"status": "dry-run"}
    return train(args)


if __name__ == "__main__":
    main()
