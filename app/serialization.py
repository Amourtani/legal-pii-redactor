from __future__ import annotations

from typing import Any

from app.entities import DesensitizationResult, Entity


def result_to_dict(result: DesensitizationResult) -> dict[str, Any]:
    return {
        "masked_text": result.masked_text,
        "risk_level": result.risk_level,
        "need_review": result.need_review,
        "entities": [_entity_to_dict(entity) for entity in result.entities],
    }


def _entity_to_dict(entity: Entity) -> dict[str, Any]:
    return {
        "start": entity.start,
        "end": entity.end,
        "label": entity.label,
        "text": entity.text,
        "source": entity.source,
        "score": entity.score,
        "replacement": entity.replacement,
    }

