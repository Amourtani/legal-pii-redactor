from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class AnnotationError(ValueError):
    pass


@dataclass(slots=True)
class BioExample:
    tokens: list[str]
    labels: list[str]


def jsonl_record_to_bio(record: dict[str, Any]) -> BioExample:
    text = record.get("text")
    entities = record.get("entities", [])
    if not isinstance(text, str):
        raise AnnotationError("record.text must be a string")

    labels = ["O"] * len(text)
    occupied = [False] * len(text)

    for entity in sorted(entities, key=lambda item: (item["start"], item["end"])):
        start = int(entity["start"])
        end = int(entity["end"])
        label = str(entity["label"])
        if start < 0 or end <= start or end > len(text):
            raise AnnotationError(f"invalid entity span: {entity}")
        if any(occupied[index] for index in range(start, end)):
            raise AnnotationError(f"overlapping entity span: {entity}")
        for index in range(start, end):
            occupied[index] = True
            prefix = "B" if index == start else "I"
            labels[index] = f"{prefix}-{label}"

    return BioExample(tokens=list(text), labels=labels)

