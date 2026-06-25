from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from train.dataset import jsonl_record_to_bio


DEFAULT_EPOCHS = 3.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_WARMUP_RATIO = 0.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: float
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    auto_find_batch_size: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a legal PII token-classification model.")
    parser.add_argument("--train-file", default="data/generated/legal_ner.jsonl")
    parser.add_argument("--dev-file", default="data/generated/legal_ner.dev.jsonl")
    parser.add_argument("--base-model", default="uer/chinese_roberta_L-2_H-128")
    parser.add_argument("--output-dir", default="models/legal-ner")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument(
        "--preset",
        choices=["manual", "auto"],
        default="manual",
        help="manual preserves explicit/default hyperparameters; auto adapts them before training starts.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. Use cuda to fail fast when GPU is unavailable.",
    )
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install ML dependencies with: pip install -r requirements-ml.txt") from exc

    device_message = resolve_device(args.device, torch)
    print(f"training device: {device_message}")

    train_records = _load_records(Path(args.train_file))
    dev_records = _load_records(Path(args.dev_file)) if Path(args.dev_file).exists() else train_records[: max(1, len(train_records) // 10)]
    train_config = resolve_training_config(
        preset=args.preset,
        train_count=len(train_records),
        base_model=args.base_model,
        device_message=device_message,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
    )
    print(
        "training config: "
        f"preset={args.preset}, train_count={len(train_records)}, "
        f"epochs={train_config.epochs}, batch_size={train_config.batch_size}, "
        f"learning_rate={train_config.learning_rate}, warmup_ratio={train_config.warmup_ratio}, "
        f"auto_find_batch_size={train_config.auto_find_batch_size}"
    )
    labels = _collect_labels(train_records + dev_records)
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    train_dataset = Dataset.from_list([_to_dataset_item(record) for record in train_records])
    dev_dataset = Dataset.from_list([_to_dataset_item(record) for record in dev_records])

    def tokenize_and_align(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=256,
        )
        aligned_labels = []
        for batch_index, labels_for_example in enumerate(batch["labels"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            previous_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label2id[labels_for_example[word_id]])
                else:
                    label_ids.append(label2id[labels_for_example[word_id]])
                previous_word_id = word_id
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    train_tokenized = train_dataset.map(tokenize_and_align, batched=True)
    dev_tokenized = dev_dataset.map(tokenize_and_align, batched=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    training_args = build_training_arguments(
        TrainingArguments,
        output_dir=args.output_dir,
        batch_size=train_config.batch_size,
        epochs=train_config.epochs,
        device=args.device,
        learning_rate=train_config.learning_rate,
        warmup_ratio=train_config.warmup_ratio,
        auto_find_batch_size=train_config.auto_find_batch_size,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, "labels.json").write_text(
        json.dumps(id2label, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _to_dataset_item(record: dict[str, Any]) -> dict[str, Any]:
    example = jsonl_record_to_bio(record)
    return {"tokens": example.tokens, "labels": example.labels}


def _collect_labels(records: list[dict[str, Any]]) -> list[str]:
    labels = {"O"}
    for record in records:
        labels.update(jsonl_record_to_bio(record).labels)
    return sorted(labels, key=lambda label: (label != "O", label))


def build_training_arguments(
    training_arguments_cls: type,
    output_dir: str,
    batch_size: int,
    epochs: float,
    device: str = "auto",
    learning_rate: float = DEFAULT_LEARNING_RATE,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    auto_find_batch_size: bool = False,
) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "logging_steps": 50,
    }
    strategy_arg = _evaluation_strategy_arg(training_arguments_cls)
    if strategy_arg:
        kwargs[strategy_arg] = "epoch"
    device_kwargs = _device_kwargs(training_arguments_cls, device)
    kwargs.update(device_kwargs)
    optional_kwargs = _supported_optional_kwargs(
        training_arguments_cls,
        {
            "warmup_ratio": warmup_ratio,
            "auto_find_batch_size": auto_find_batch_size,
        },
    )
    kwargs.update(optional_kwargs)
    return training_arguments_cls(**kwargs)


def resolve_training_config(
    preset: str,
    train_count: int,
    base_model: str,
    device_message: str,
    epochs: float | None,
    batch_size: int | None,
    learning_rate: float | None,
    warmup_ratio: float | None,
) -> TrainingConfig:
    if preset == "auto":
        inferred = _auto_training_config(train_count, base_model, device_message)
    else:
        inferred = TrainingConfig(
            epochs=DEFAULT_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
            learning_rate=DEFAULT_LEARNING_RATE,
            warmup_ratio=DEFAULT_WARMUP_RATIO,
        )
    return TrainingConfig(
        epochs=epochs if epochs is not None else inferred.epochs,
        batch_size=batch_size if batch_size is not None else inferred.batch_size,
        learning_rate=learning_rate if learning_rate is not None else inferred.learning_rate,
        warmup_ratio=warmup_ratio if warmup_ratio is not None else inferred.warmup_ratio,
        auto_find_batch_size=inferred.auto_find_batch_size,
    )


def _auto_training_config(train_count: int, base_model: str, device_message: str) -> TrainingConfig:
    model_size = _model_size(base_model)
    cuda = "cuda" in device_message.lower()
    if model_size == "small":
        return TrainingConfig(
            epochs=_epochs_for_size(train_count, below_10k=12, below_50k=8, above_50k=5),
            batch_size=32 if cuda else 16,
            learning_rate=3e-5,
            warmup_ratio=0.05,
            auto_find_batch_size=cuda,
        )
    if model_size == "medium":
        return TrainingConfig(
            epochs=_epochs_for_size(train_count, below_10k=8, below_50k=6, above_50k=4),
            batch_size=16 if cuda else 8,
            learning_rate=2e-5,
            warmup_ratio=0.06,
            auto_find_batch_size=cuda,
        )
    return TrainingConfig(
        epochs=_epochs_for_size(train_count, below_10k=6, below_50k=5, above_50k=3),
        batch_size=8 if cuda else 4,
        learning_rate=2e-5,
        warmup_ratio=0.06,
        auto_find_batch_size=cuda,
    )


def _model_size(base_model: str) -> str:
    lowered = base_model.lower()
    if "l-2_h-128" in lowered:
        return "small"
    if "l-4_h-512" in lowered:
        return "medium"
    return "base"


def _epochs_for_size(train_count: int, below_10k: float, below_50k: float, above_50k: float) -> float:
    if train_count < 10_000:
        return below_10k
    if train_count < 50_000:
        return below_50k
    return above_50k


def resolve_device(device: str, torch_module: Any) -> str:
    if device == "cuda":
        if not torch_module.cuda.is_available():
            raise SystemExit(
                "CUDA is not available in this Python environment. "
                "Install a CUDA-enabled PyTorch build, then rerun with --device cuda."
            )
        return f"cuda: {torch_module.cuda.get_device_name(0)}"
    if device == "cpu":
        return "cpu"
    if torch_module.cuda.is_available():
        return f"auto -> cuda: {torch_module.cuda.get_device_name(0)}"
    return "auto -> cpu"


def _device_kwargs(training_arguments_cls: type, device: str) -> dict[str, Any]:
    if device != "cpu":
        return {}
    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    if "use_cpu" in parameters:
        return {"use_cpu": True}
    if "no_cuda" in parameters:
        return {"no_cuda": True}
    return {}


def _supported_optional_kwargs(
    training_arguments_cls: type,
    candidates: dict[str, Any],
) -> dict[str, Any]:
    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    return {key: value for key, value in candidates.items() if key in parameters}


def _evaluation_strategy_arg(training_arguments_cls: type) -> str | None:
    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" in parameters:
        return "eval_strategy"
    if "evaluation_strategy" in parameters:
        return "evaluation_strategy"
    return None


if __name__ == "__main__":
    main()
