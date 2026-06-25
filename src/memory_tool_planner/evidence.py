from __future__ import annotations

import re
from typing import Any


SCOPED_LOCOMO_EVIDENCE_RE = re.compile(r"^locomo_(\d+):(.+)$")
LOCOMO_CONVERSATION_RE = re.compile(r"(?:^|_)locomo_(\d+)(?:_|$)")


def scoped_locomo_evidence_id(evidence_id: Any, conversation_index: int | str | None) -> str:
    text = str(evidence_id or "").strip()
    if not text:
        return ""
    if SCOPED_LOCOMO_EVIDENCE_RE.match(text):
        return text
    if conversation_index is None:
        return text
    return f"locomo_{int(conversation_index)}:{text}"


def scoped_locomo_evidence_ids(evidence_ids: Any, conversation_index: int | str | None) -> list[str]:
    if not isinstance(evidence_ids, list):
        evidence_ids = [evidence_ids]
    return [
        scoped
        for item in evidence_ids
        if (scoped := scoped_locomo_evidence_id(item, conversation_index))
    ]


def infer_locomo_conversation_index(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("locomo_"):
            parts = text.split("_", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
        match = LOCOMO_CONVERSATION_RE.search(text)
        if match:
            return int(match.group(1))
        return None
    if isinstance(value, list):
        for item in value:
            found = infer_locomo_conversation_index(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for key in [
            "conversation_index",
            "conversation_id",
            "memory_id",
            "uid",
            "participants",
            "memcell_ids",
        ]:
            found = infer_locomo_conversation_index(value.get(key))
            if found is not None:
                return found
    return None


def row_conversation_index(row: dict[str, Any]) -> int | None:
    for key in [
        "conversation_index",
        "conversation_id",
        "memory_id",
        "uid",
        "entity",
        "participants",
        "memcell_ids",
    ]:
        found = infer_locomo_conversation_index(row.get(key))
        if found is not None:
            return found
    return None


def with_scoped_row_evidence(row: dict[str, Any]) -> dict[str, Any]:
    conversation_index = row_conversation_index(row)
    if conversation_index is None:
        return row
    evidence_ids = row.get("evidence_ids")
    if not isinstance(evidence_ids, list):
        return row
    scoped = scoped_locomo_evidence_ids(evidence_ids, conversation_index)
    if scoped == evidence_ids and row.get("conversation_index") is not None:
        return row
    updated = dict(row)
    updated["conversation_index"] = conversation_index
    updated["conversation_id"] = row.get("conversation_id") or f"locomo_{conversation_index}"
    updated["evidence_ids"] = scoped
    return updated
