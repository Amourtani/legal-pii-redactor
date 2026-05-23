from __future__ import annotations

from typing import Protocol

from app.entities import DesensitizationResult, Entity
from app.masker import mask_entities
from app.rules import RuleDetector


class NerDetector(Protocol):
    def detect(self, text: str) -> list[Entity]:
        ...


class NullNerDetector:
    def detect(self, text: str) -> list[Entity]:
        return []


class Desensitizer:
    def __init__(
        self,
        rule_detector: RuleDetector | None = None,
        ner_detector: NerDetector | None = None,
    ) -> None:
        self.rule_detector = rule_detector or RuleDetector()
        self.ner_detector = ner_detector or NullNerDetector()

    def desensitize(
        self,
        text: str,
        mode: str = "case_display",
        strict_level: str = "medium",
    ) -> DesensitizationResult:
        entities = self.rule_detector.detect(text)
        entities.extend(self.ner_detector.detect(text))
        result = mask_entities(text, entities)
        if mode == "contract_generation":
            result.need_review = result.need_review or strict_level == "high"
        return result

