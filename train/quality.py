from __future__ import annotations

import re


PHONE_RE = re.compile(
    r"(?:\+?86[- ]?)?1[3-9]\d[- .]?\d{4}[- .]?\d{4}|"
    r"0\d{2,3}[- ]?\d{7,8}(?:-\d{1,6})?"
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IP_RE = re.compile(
    r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
)
URL_RE = re.compile(r"https?://[^\s，。；；、)）]+")
ID_CARD_RE = re.compile(r"\d{17}[\dXx]")
BANK_ACCOUNT_RE = re.compile(r"(?:\d[ -]?){8,30}")
PLATE_RE = re.compile(r"[\u4e00-\u9fa5][A-Z][·.]?[A-Z0-9]{5,6}")

BOUNDARY_PUNCTUATION = set("，。；：、（）()《》“”\"'‘’")
FIELD_PREFIXES = [
    "电话",
    "邮箱",
    "身份证号",
    "身份证号码",
    "银行卡号",
    "银行账号",
    "账号",
    "访问地址",
    "链接",
    "网址",
]
ADDRESS_PREFIXES = ["住址", "地址", "家庭住址", "注册地", "地点", "地点在"]


def format_issue(label: str, span: str) -> str | None:
    if not span:
        return "span is empty"
    if span[0] in BOUNDARY_PUNCTUATION or span[-1] in BOUNDARY_PUNCTUATION:
        return "span includes boundary punctuation"
    if any(span.startswith(prefix) for prefix in FIELD_PREFIXES):
        return "span includes a field prefix"
    if label.startswith("PERSON") or label in {"NAME", "LAWYER", "JUDGE"}:
        person_issue = _person_issue(label, span)
        if person_issue:
            return person_issue
    if label == "ADDRESS" and any(span.startswith(prefix) for prefix in ADDRESS_PREFIXES):
        return "ADDRESS span includes a field prefix"
    if label == "PHONE" and not PHONE_RE.fullmatch(span):
        return "PHONE span does not look like a phone number"
    if label == "EMAIL" and not EMAIL_RE.fullmatch(span):
        return "EMAIL span does not look like an email"
    if label == "IP" and not IP_RE.fullmatch(span):
        return "IP span does not look like an IPv4 address"
    if label == "URL" and not URL_RE.fullmatch(span):
        return "URL span does not look like a URL"
    if label == "ID_CARD" and not ID_CARD_RE.fullmatch(span):
        return "ID_CARD span does not look like an ID card candidate"
    if label in {"BANK_CARD", "BANK_ACCOUNT"} and not _looks_like_bank_account(span):
        return f"{label} span does not look like a bank account candidate"
    if label == "PLATE" and not PLATE_RE.fullmatch(span):
        return "PLATE span does not look like a plate number"
    return None


def _looks_like_bank_account(span: str) -> bool:
    digits = re.sub(r"\D", "", span)
    return 8 <= len(digits) <= 30 and bool(BANK_ACCOUNT_RE.fullmatch(span))


def _person_issue(label: str, span: str) -> str | None:
    if re.search(r"\d|[A-Za-z]", span):
        return "PERSON span has suspicious boundary"
    role_only = {
        "原告",
        "被告",
        "证人",
        "律师",
        "辩护人",
        "辩护律师",
        "法官",
        "审判长",
        "当事人",
        "客户",
        "患者",
        "受害人",
        "被害人",
        "代理人",
    }
    if span in role_only:
        return "PERSON span has suspicious boundary"
    suspicious_starts = set("人户者告师官长员方诉害审辩律被原上代")
    if span[0] in suspicious_starts:
        return "PERSON span has suspicious boundary"
    if label == "LAWYER" and "律师" in span and not span.endswith("律师"):
        return "PERSON span has suspicious boundary"
    if label == "JUDGE" and "法官" in span and not span.endswith("法官"):
        return "PERSON span has suspicious boundary"
    return None
