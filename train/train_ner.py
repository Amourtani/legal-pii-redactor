from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from train.dataset import jsonl_record_to_bio


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a legal PII token-classification model.")
    parser.add_argument("--train-file", default="data/generated/legal_ner.jsonl")
    parser.add_argument("--dev-file", default="data/generated/legal_ner.dev.jsonl")
    parser.add_argument("--base-model", default="uer/chinese_roberta_L-2_H-128")
    parser.add_argument("--output-dir", default="models/legal-ner")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    try:
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

    train_records = _load_records(Path(args.train_file))
    dev_records = _load_records(Path(args.dev_file)) if Path(args.dev_file).exists() else train_records[: max(1, len(train_records) // 10)]
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
        batch_size=args.batch_size,
        epochs=args.epochs,
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
) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": 3e-5,
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
    return training_arguments_cls(**kwargs)


def _evaluation_strategy_arg(training_arguments_cls: type) -> str | None:
    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" in parameters:
        return "eval_strategy"
    if "evaluation_strategy" in parameters:
        return "evaluation_strategy"
    return None


if __name__ == "__main__":
    main()
