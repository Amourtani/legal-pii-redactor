from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from train.dataset import jsonl_record_to_bio
from train.quality import format_issue
from train.synthetic import LEGAL_LABELS


@dataclass(frozen=True, slots=True)
class ValidationErrorItem:
    line: int
    message: str


@dataclass(slots=True)
class ValidationReport:
    record_count: int = 0
    entity_count: int = 0
    errors: list[ValidationErrorItem] = field(default_factory=list)
    label_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_dataset(path: str | Path) -> ValidationReport:
    report = ValidationReport()
    allowed = set(LEGAL_LABELS)
    file_path = Path(path)

    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        report.record_count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            report.errors.append(ValidationErrorItem(line_number, f"invalid JSON: {exc}"))
            continue
        _validate_record(record, line_number, allowed, report)

    return report


def _validate_record(
    record: dict[str, Any],
    line_number: int,
    allowed: set[str],
    report: ValidationReport,
) -> None:
    text = record.get("text")
    entities = record.get("entities")
    if not isinstance(text, str) or not text:
        report.errors.append(ValidationErrorItem(line_number, "text must be a non-empty string"))
        return
    if not isinstance(entities, list):
        report.errors.append(ValidationErrorItem(line_number, "entities must be a list"))
        return

    for index, entity in enumerate(entities):
        _validate_entity(text, entity, index, line_number, allowed, report)

    try:
        jsonl_record_to_bio(record)
    except Exception as exc:
        report.errors.append(ValidationErrorItem(line_number, f"invalid BIO offsets: {exc}"))


def _validate_entity(
    text: str,
    entity: dict[str, Any],
    index: int,
    line_number: int,
    allowed: set[str],
    report: ValidationReport,
) -> None:
    if "text" in entity:
        report.errors.append(
            ValidationErrorItem(line_number, f"entity {index} must use normalized start/end offsets")
        )
        return
    try:
        start = entity["start"]
        end = entity["end"]
        label = entity["label"]
    except KeyError as exc:
        report.errors.append(ValidationErrorItem(line_number, f"entity {index} missing key: {exc}"))
        return
    if not isinstance(start, int) or not isinstance(end, int):
        report.errors.append(ValidationErrorItem(line_number, f"entity {index} start/end must be integers"))
        return
    if label not in allowed:
        report.errors.append(ValidationErrorItem(line_number, f"entity {index} unknown label: {label}"))
        return
    if not (0 <= start < end <= len(text)):
        report.errors.append(ValidationErrorItem(line_number, f"entity {index} span out of range"))
        return

    span = text[start:end]
    report.entity_count += 1
    report.label_counts[label] = report.label_counts.get(label, 0) + 1
    message = _format_issue(label, span)
    if message:
        report.errors.append(ValidationErrorItem(line_number, f"entity {index} {message}: {span!r}"))


def _format_issue(label: str, span: str) -> str | None:
    return format_issue(label, span)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalized legal NER JSONL.")
    parser.add_argument("path")
    args = parser.parse_args()

    report = validate_dataset(args.path)
    print(f"records={report.record_count}")
    print(f"entities={report.entity_count}")
    print("label_counts=" + json.dumps(report.label_counts, ensure_ascii=False, sort_keys=True))
    if report.errors:
        print(f"errors={len(report.errors)}")
        for error in report.errors[:100]:
            print(f"line {error.line}: {error.message}")
        raise SystemExit(1)
    print("errors=0")


if __name__ == "__main__":
    main()
