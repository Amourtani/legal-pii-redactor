from __future__ import annotations

from app.entities import Entity


LABEL_PRIORITY = {
    "ID_CARD": 100,
    "BANK_CARD": 95,
    "BANK_ACCOUNT": 95,
    "EMAIL": 90,
    "PHONE": 90,
    "IP": 85,
    "URL": 85,
    "PLATE": 80,
    "ADDRESS": 75,
    "NAME": 72,
    "PERSON_MINOR": 75,
    "PERSON_PARTY": 70,
    "PERSON_VICTIM": 70,
    "PERSON_WITNESS": 70,
    "PERSON_AGENT": 65,
    "ORG_PARTY": 55,
    "ORG_LAWFIRM": 50,
    "CASE_NUMBER": 40,
    "BUSINESS_SECRET": 40,
}


def merge_entities(entities: list[Entity]) -> list[Entity]:
    candidates = [
        entity
        for entity in entities
        if entity.start >= 0 and entity.end > entity.start and entity.text
    ]
    ranked = sorted(
        candidates,
        key=lambda entity: (
            -LABEL_PRIORITY.get(entity.label, 10),
            -entity.length,
            entity.start,
        ),
    )

    kept: list[Entity] = []
    for entity in ranked:
        if any(entity.overlaps(existing) for existing in kept):
            continue
        kept.append(entity)

    return sorted(kept, key=lambda entity: entity.start)
