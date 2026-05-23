from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from train.dataset import jsonl_record_to_bio
from train.quality import BOUNDARY_PUNCTUATION, FIELD_PREFIXES, format_issue


LEGAL_LABELS = [
    "PERSON_PARTY",
    "PERSON_VICTIM",
    "PERSON_WITNESS",
    "PERSON_MINOR",
    "PERSON_AGENT",
    "LAWYER",
    "JUDGE",
    "ORG_PARTY",
    "ORG_LAWFIRM",
    "ADDRESS",
    "ID_CARD",
    "PHONE",
    "EMAIL",
    "BANK_CARD",
    "BANK_ACCOUNT",
    "PLATE",
    "HEALTH",
    "BUSINESS_SECRET",
    "CONTRACT_PROJECT",
]


class GeneratedDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GeneratedRecordError:
    line: int
    message: str
    raw: str


def build_generation_prompt(count: int, mode: str) -> str:
    labels = "、".join(LEGAL_LABELS)
    return f"""你是法律行业文本脱敏数据生成器。
请生成 {count} 条用于法律行业文本脱敏 NER 训练的中文样本，场景模式为 {mode}。

输出必须是 JSONL，每行一个 JSON 对象，不要输出解释文字。每个对象格式：
{{"text":"证人王五称电话为13900139000。","entities":[{{"text":"王五","label":"PERSON_WITNESS"}},{{"text":"13900139000","label":"PHONE"}}]}}

要求：
1. 不要输出 start/end，实体只输出 text 和 label，由程序自动计算下标。
2. 实体 text 必须是原文中完整且连续出现的敏感片段，不要包含角色词、标点或说明词。
3. 标签只能从以下集合选择：{labels}。
4. 混合合同、案例展示、咨询记录、裁判摘要、证据材料等法律文本。
5. 加入一部分负样本，避免把普通法律术语误识别为隐私。
6. 如果同一个实体文本在原文中出现多次，必须加入 occurrence 字段，1 表示第 1 次出现，2 表示第 2 次出现。
7. 人名只标姓名本身，例如标“王五”，不要标“证人王五”或“证人”。
8. 电话、邮箱、身份证、银行卡、车牌必须只标实际号码，不要包含“电话：”“邮箱：”“身份证号：”等前缀。
9. 脱敏召回优先：ID_CARD 可以是 18 位身份证样式，不强制校验位正确；PHONE 可以是手机号或固定电话；PLATE 可以包含中点，例如“京A·88888”。
10. 可参考这些格式：110101199001011237、110101198001011234、13800138000、0755-86001234、jingli@example.com、京A12345、浙A·B5678。
"""


def parse_generated_records(payload: str) -> list[dict[str, Any]]:
    records, errors = parse_generated_records_lenient(payload)
    if errors:
        first = errors[0]
        raise GeneratedDataError(f"line {first.line}: {first.message}")
    return records


def parse_generated_records_lenient(payload: str) -> tuple[list[dict[str, Any]], list[GeneratedRecordError]]:
    cleaned = _strip_code_fence(payload)
    records = []
    errors: list[GeneratedRecordError] = []
    for line_number, line in enumerate(cleaned.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw_record = json.loads(line)
            records.append(normalize_generated_record(raw_record))
        except (json.JSONDecodeError, GeneratedDataError, KeyError, TypeError, ValueError) as exc:
            errors.append(GeneratedRecordError(line=line_number, message=str(exc), raw=line))
    return records, errors


def normalize_generated_record(record: dict[str, Any]) -> dict[str, Any]:
    text = record.get("text")
    raw_entities = record.get("entities", [])
    if not isinstance(text, str) or not text:
        raise GeneratedDataError("record.text must be a non-empty string")
    if not isinstance(raw_entities, list):
        raise GeneratedDataError("record.entities must be a list")

    entities = []
    for index, entity in enumerate(raw_entities):
        entities.append(_normalize_entity(text, entity, index))

    normalized = {"text": text, "entities": sorted(entities, key=lambda item: (item["start"], item["end"]))}
    try:
        jsonl_record_to_bio(normalized)
    except Exception as exc:
        raise GeneratedDataError(f"invalid normalized annotation: {exc}") from exc
    return normalized


def _normalize_entity(text: str, entity: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(entity, dict):
        raise GeneratedDataError(f"entity {index} must be an object")
    label = entity.get("label")
    if label not in LEGAL_LABELS:
        raise GeneratedDataError(f"entity {index} has unknown label: {label}")

    if "text" in entity:
        span_text = entity["text"]
        if not isinstance(span_text, str) or not span_text:
            raise GeneratedDataError(f"entity {index} text must be a non-empty string")
        has_occurrence = "occurrence" in entity
        occurrence = int(entity.get("occurrence", 1))
        start = _find_unique_or_occurrence(text, span_text, occurrence, has_occurrence, index)
        end = start + len(span_text)
    elif {"start", "end"} <= set(entity):
        start = int(entity["start"])
        end = int(entity["end"])
        span_text = text[start:end] if 0 <= start < end <= len(text) else ""
        raise GeneratedDataError(
            f"entity {index} uses start/end for {span_text!r}; regenerate with entity text only"
        )
    else:
        raise GeneratedDataError(f"entity {index} must contain text and label")

    _validate_span_boundary(span_text, label, index)
    _validate_span_format(span_text, label, index)
    return {"start": start, "end": end, "label": label}


def _find_unique_or_occurrence(
    text: str,
    span_text: str,
    occurrence: int,
    has_occurrence: bool,
    index: int,
) -> int:
    starts = []
    cursor = 0
    while True:
        found = text.find(span_text, cursor)
        if found == -1:
            break
        starts.append(found)
        cursor = found + 1

    if not starts:
        raise GeneratedDataError(f"entity {index} text {span_text!r} not found in record.text")
    if occurrence < 1:
        raise GeneratedDataError(f"entity {index} occurrence must be >= 1")
    if occurrence > len(starts) and len(starts) == 1:
        occurrence = 1
    elif occurrence > len(starts):
        raise GeneratedDataError(
            f"entity {index} occurrence {occurrence} exceeds {len(starts)} matches for {span_text!r}"
        )
    if len(starts) > 1 and not has_occurrence:
        raise GeneratedDataError(
            f"entity {index} text {span_text!r} appears {len(starts)} times; add occurrence"
        )
    return starts[occurrence - 1]


def _validate_span_boundary(span_text: str, label: str, index: int) -> None:
    if span_text[0] in BOUNDARY_PUNCTUATION or span_text[-1] in BOUNDARY_PUNCTUATION:
        raise GeneratedDataError(f"entity {index} span {span_text!r} includes boundary punctuation")
    if any(span_text.startswith(prefix) for prefix in FIELD_PREFIXES):
        raise GeneratedDataError(f"entity {index} span {span_text!r} includes a field prefix")


def _validate_span_format(span_text: str, label: str, index: int) -> None:
    issue = format_issue(label, span_text)
    if issue:
        raise GeneratedDataError(f"entity {index} {issue}: {span_text!r}")


def _strip_code_fence(payload: str) -> str:
    text = payload.strip()
    lines = [line.strip() for line in text.splitlines()]
    if lines and lines[0].startswith("```"):
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text
