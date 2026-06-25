from __future__ import annotations

from typing import Any


def evidence_ids(memory: dict[str, Any]) -> list[str]:
    values = memory.get("evidence_ids", [])
    return values if isinstance(values, list) else []


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def memory_to_text(tool_name: str, memory: dict[str, Any], max_chars: int | None = None) -> str:
    mid = memory.get("memory_id", "")
    evidence = ",".join(evidence_ids(memory))
    if tool_name == "search_core_memory":
        body = memory.get("content", "")
    else:
        body = memory.get("event", "")
    return f"- [{tool_name} {mid} evidence={evidence}] {_truncate(str(body), max_chars)}"


def build_memory_context(tool_results: dict[str, Any], token_budget: int = 0, max_memory_chars: int | None = None) -> dict[str, Any]:
    lines: list[str] = []
    structured: list[dict[str, Any]] = []
    seen: set[str] = set()
    priority = {"search_episodic_memory": 0, "search_core_memory": 1}
    sorted_results = sorted(tool_results.get("tool_results", []), key=lambda item: priority.get(item.get("name", ""), 99))
    for result in sorted_results:
        if result.get("status") != "ok":
            continue
        tool_name = result.get("name", "")
        for memory in result.get("result", {}).get("memories", []):
            key = memory.get("memory_id") or "|".join(evidence_ids(memory))
            if key in seen:
                continue
            seen.add(key)
            line = memory_to_text(tool_name, memory, max_chars=max_memory_chars)
            tentative = "\n".join(lines + [line])
            if token_budget > 0 and len(tentative.split()) > token_budget:
                continue
            lines.append(line)
            structured.append(memory)
    return {"memory_context": "\n".join(lines), "memories": structured}
