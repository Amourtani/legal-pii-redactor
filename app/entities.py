from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Entity:
    start: int
    end: int
    label: str
    text: str
    source: str
    score: float = 1.0
    replacement: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Entity") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(slots=True)
class DesensitizationResult:
    masked_text: str
    entities: list[Entity]
    need_review: bool = False
    risk_level: str = "low"

