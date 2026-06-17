from __future__ import annotations

import re

from app.entities import DesensitizationResult, Entity
from app.span import merge_entities


def mask_entities(text: str, entities: list[Entity]) -> DesensitizationResult:
    merged = merge_entities(entities)
    output: list[str] = []
    cursor = 0
    masked_entities: list[Entity] = []

    for entity in merged:
        output.append(text[cursor : entity.start])
        replacement = mask_value(entity.label, entity.text)
        entity.replacement = replacement
        output.append(replacement)
        masked_entities.append(entity)
        cursor = entity.end

    output.append(text[cursor:])
    risk_level = _risk_level(masked_entities)
    need_review = any(entity.label == "BUSINESS_SECRET" or entity.score < 0.65 for entity in masked_entities)
    return DesensitizationResult(
        masked_text="".join(output),
        entities=masked_entities,
        need_review=need_review,
        risk_level=risk_level,
    )


def mask_value(label: str, value: str) -> str:
    if label in {"PERSON_PARTY", "PERSON_VICTIM", "PERSON_WITNESS", "PERSON_AGENT", "PERSON_MINOR", "LAWYER", "JUDGE", "NAME"}:
        return _mask_person(value)
    if label == "PHONE":
        return _mask_phone(value)
    if label == "ID_CARD":
        return _mask_id_card(value)
    if label == "EMAIL":
        return _mask_email(value)
    if label in {"BANK_CARD", "BANK_ACCOUNT"}:
        return _mask_bank_card(value)
    if label == "PLATE":
        return value[:2] + "*" * max(len(value) - 2, 1)
    if label == "ADDRESS":
        return _mask_address(value)
    if label in {"IP", "URL"}:
        return value[:4] + "***" if len(value) > 4 else "***"
    if label == "CASE_NUMBER":
        return value
    if label.startswith("ORG"):
        return _mask_org(value)
    return "***"


def _mask_person(value: str) -> str:
    if len(value) <= 1:
        return "*"
    return value[0] + "*" * (len(value) - 1)


def _mask_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) < 11:
        return "***"

    digit_index = 0
    output = []
    for char in value:
        if not char.isdigit():
            output.append(char)
            continue
        digit_index += 1
        output.append("*" if 4 <= digit_index <= 7 else char)
    return "".join(output)


def _mask_id_card(value: str) -> str:
    return value[:6] + "********" + value[-4:] if len(value) >= 18 else "***"


def _mask_email(value: str) -> str:
    if "@" not in value:
        return "***"
    local, domain = value.split("@", 1)
    prefix = local[:1] if local else "*"
    return f"{prefix}***@{domain}"


def _mask_bank_card(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) < 8:
        return "***"
    return digits[:4] + " **** **** " + digits[-4:]


def _mask_address(value: str) -> str:
    last_admin_end = 0
    for match in re.finditer(r"[省市区县州]", value):
        last_admin_end = match.end()
    if last_admin_end >= 2:
        return value[:last_admin_end] + "****"
    return value[:2] + "****" if len(value) > 2 else "****"


def _mask_org(value: str) -> str:
    suffixes = ["有限公司", "股份有限公司", "公司", "律所", "律师事务所", "中心", "委员会"]
    for suffix in suffixes:
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[:2] + "某" + suffix
    return value[:2] + "****" if len(value) > 2 else "****"


def _risk_level(entities: list[Entity]) -> str:
    high_labels = {"ID_CARD", "BANK_CARD", "PERSON_MINOR", "HEALTH", "BUSINESS_SECRET"}
    medium_labels = {"PHONE", "EMAIL", "ADDRESS", "PERSON_PARTY", "PERSON_VICTIM", "PERSON_WITNESS"}
    if any(entity.label in high_labels for entity in entities):
        return "high"
    if any(entity.label in medium_labels for entity in entities):
        return "medium"
    return "low"
