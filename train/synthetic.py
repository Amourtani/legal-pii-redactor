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


DIVERSITY_PROFILES = [
    (
        "庭审/询问摘录",
        "使用问答、陈述、追问、笔录式表达；可出现法官、律师、证人、被害人、当事人之间的短句互动。",
        "重点覆盖 PERSON_PARTY、PERSON_WITNESS、PERSON_VICTIM、LAWYER、JUDGE、ADDRESS、PHONE。",
    ),
    (
        "合同条款/履约往来",
        "使用合同条款、补充协议、付款通知、交付验收、违约沟通等表达；句式可偏正式或半结构化。",
        "重点覆盖 ORG_PARTY、PERSON_AGENT、CONTRACT_PROJECT、BANK_ACCOUNT、EMAIL、PHONE、ADDRESS。",
    ),
    (
        "案例展示/裁判摘要",
        "使用案情简介、争议焦点、法院认定、裁判结果、公开展示改写等表达；避免每条都以“原告/被告”开头。",
        "重点覆盖 PERSON_PARTY、ORG_PARTY、ORG_LAWFIRM、LAWYER、JUDGE、ID_CARD、PLATE。",
    ),
    (
        "证据材料/附件清单",
        "使用证据目录、聊天记录摘要、转账凭证、物流单、车辆信息、病历材料等表达；可使用编号和短行文本。",
        "重点覆盖 BANK_CARD、BANK_ACCOUNT、PLATE、PHONE、EMAIL、HEALTH、BUSINESS_SECRET。",
    ),
    (
        "法律咨询/委托记录",
        "使用咨询记录、委托登记、回访备注、律师工作日志等表达；可出现口语化描述和省略句。",
        "重点覆盖 PERSON_PARTY、PERSON_AGENT、LAWYER、ORG_LAWFIRM、PHONE、ADDRESS、ID_CARD。",
    ),
    (
        "企业合规/商业秘密",
        "使用尽调纪要、合规审查、竞业限制、算法/客户名单/报价方案等商业场景；文本要像内部法律审阅材料。",
        "重点覆盖 ORG_PARTY、BUSINESS_SECRET、CONTRACT_PROJECT、PERSON_AGENT、EMAIL、BANK_ACCOUNT。",
    ),
    (
        "执行/调解/送达材料",
        "使用执行通知、和解协议、调解笔录、送达地址确认、财产线索等表达；可包含多个自然人和机构。",
        "重点覆盖 PERSON_PARTY、ADDRESS、BANK_ACCOUNT、PHONE、ID_CARD、ORG_PARTY。",
    ),
    (
        "负样本/边界样本",
        "生成部分没有敏感实体或只有普通法律术语的文本；也可包含像法条号、案由、金额、日期等不应标注的内容。",
        "重点训练不要把普通名词、角色词、法院通用称谓、合同章节标题误标为隐私。",
    ),
]


class GeneratedDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GeneratedRecordError:
    line: int
    message: str
    raw: str


def build_generation_prompt(count: int, mode: str, batch_index: int = 0) -> str:
    labels = "、".join(LEGAL_LABELS)
    profile_offset = max(batch_index - 1, 0)
    profile_name, profile_style, profile_labels = DIVERSITY_PROFILES[
        profile_offset % len(DIVERSITY_PROFILES)
    ]
    return f"""你是法律行业文本脱敏数据生成器。
请生成 {count} 条用于法律行业文本脱敏 NER 训练的中文样本，场景模式为 {mode}。

输出必须是 JSONL，每行一个 JSON 对象，不要输出解释文字。每个对象格式：
{{"text":"证人王五称电话为13900139000。","entities":[{{"text":"王五","label":"PERSON_WITNESS"}},{{"text":"13900139000","label":"PHONE"}}]}}

本批次多样性画像：
- 画像名称：{profile_name}
- 文体要求：{profile_style}
- 实体侧重：{profile_labels}

要求：
1. 不要输出 start/end，实体只输出 text 和 label，由程序自动计算下标。
2. 实体 text 必须是原文中完整且连续出现的敏感片段，不要包含角色词、标点或说明词。
3. 标签只能从以下集合选择：{labels}。
4. 每条 text 都要像真实法律业务文本，不要像模板填空；不要连续使用相同开头、相同句式或相同实体顺序。
5. 同一批内至少混合 5 种表达形态：长段落、短句、编号条目、咨询/对话、记录摘要、条款、通知、证据说明、裁判摘要。
6. 文本长度要拉开：约三分之一为 15-35 字短文本，三分之一为 36-80 字中等文本，三分之一为 80-180 字长文本。
7. 实体密度要变化：包含少量无实体负样本、单实体样本、2-4 个实体样本，以及少量 5 个以上实体样本。
8. 加入 15%-25% 负样本，避免把普通法律术语、案由、法条号、金额、日期、角色词误识别为隐私。
9. 人名、机构名、项目名、地址、邮箱、账号、车牌要明显变化；不要反复使用“张三、李四、王五、北京某公司”等固定模板。
10. 可以使用真实业务中的含糊表达，例如“尾号8899账户”“朝阳区某小区”“王律师”“陈某某”，但实体 text 仍必须是原文中的连续片段。
11. 不同实体组合要交错出现，不要总是“姓名 + 身份证 + 电话 + 地址”的顺序。
12. 如果同一个实体文本在原文中出现多次，必须加入 occurrence 字段，1 表示第 1 次出现，2 表示第 2 次出现。
13. 人名只标姓名本身，例如标“王五”，不要标“证人王五”或“证人”。
14. 律师、法官如果文本写成“王律师”“赵法官”，可以整体标为 LAWYER/JUDGE；如果写成“律师王磊”，只标“王磊”为 LAWYER。
15. 电话、邮箱、身份证、银行卡、车牌必须只标实际号码，不要包含“电话：”“邮箱：”“身份证号：”等前缀。
16. 脱敏召回优先：ID_CARD 可以是 18 位身份证样式，不强制校验位正确；PHONE 可以是手机号或固定电话；PLATE 可以包含中点，例如“京A·88888”。
17. 可参考这些格式，但不要集中复用这些值：110101199001011237、110101198001011234、13800138000、0755-86001234、jingli@example.com、京A12345、浙A·B5678。
18. 输出的每行 JSON 必须能被 json.loads 解析；不要输出 Markdown 代码块、序号、解释或多余字段。
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
