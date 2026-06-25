from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .schema import CORE_FIELDS, TOOL_NAMES

DEFAULT_TOP_K = 5


@dataclass
class ValidationResult:
    valid: bool
    parsed: dict[str, Any] | None
    errors: list[dict[str, Any]]


def _error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details or {}}


def parse_json(raw: str | dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if isinstance(raw, dict):
        return raw, []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [_error("INVALID_JSON", str(exc))]
    if not isinstance(parsed, dict):
        return None, [_error("INVALID_JSON", "planner output must be a JSON object")]
    return parsed, []


def normalize_executable_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Fill executor-owned fields that are intentionally hidden from the model."""
    tool_calls = plan.get("tool_calls")
    if not isinstance(tool_calls, list):
        return plan
    normalized = dict(plan)
    normalized_calls: list[Any] = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            normalized_calls.append(call)
            continue
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else call.get("arguments")
        if isinstance(args, dict):
            normalized_args = dict(args)
            normalized_args["top_k"] = DEFAULT_TOP_K
        else:
            normalized_args = args
        normalized_calls.append(
            {
                "call_id": f"call_{index + 1}",
                "name": call.get("name"),
                "arguments": normalized_args,
            }
        )
    normalized["tool_calls"] = normalized_calls
    return normalized


def validate_planner_output(raw: str | dict[str, Any], max_tool_calls: int = 4, max_calls_per_tool: int = 2) -> ValidationResult:
    parsed, errors = parse_json(raw)
    if parsed is None:
        return ValidationResult(False, None, errors)
    parsed = normalize_executable_plan(parsed)
    tool_calls = parsed.get("tool_calls")
    if not isinstance(tool_calls, list):
        return ValidationResult(False, parsed, [_error("INVALID_ARGUMENTS", "top-level tool_calls must be an array")])
    if len(tool_calls) > max_tool_calls:
        errors.append(_error("TOO_MANY_CALLS", f"tool_calls length must be <= {max_tool_calls}", {"count": len(tool_calls)}))

    seen_call_ids: set[str] = set()
    seen_queries: set[tuple[str, str]] = set()
    counts_by_tool: dict[str, int] = {}
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            errors.append(_error("INVALID_ARGUMENTS", "tool call must be an object", {"index": index}))
            continue
        call_id = call.get("call_id")
        name = call.get("name")
        args = call.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            errors.append(_error("INVALID_ARGUMENTS", "call_id is required", {"index": index}))
        elif call_id in seen_call_ids:
            errors.append(_error("DUPLICATE_CALL", "call_id must be unique", {"call_id": call_id}))
        else:
            seen_call_ids.add(call_id)
        if name not in TOOL_NAMES:
            errors.append(_error("UNKNOWN_TOOL", "unknown tool name", {"name": name}))
            continue
        counts_by_tool[name] = counts_by_tool.get(name, 0) + 1
        if counts_by_tool[name] > max_calls_per_tool:
            errors.append(_error("TOO_MANY_CALLS", f"{name} can be called at most {max_calls_per_tool} times", {"name": name}))
        if not isinstance(args, dict):
            errors.append(_error("INVALID_ARGUMENTS", "arguments must be an object", {"name": name}))
            continue
        queries = _argument_queries(args)
        if not queries:
            errors.append(_error("INVALID_ARGUMENTS", "queries must be non-empty", {"name": name}))
        for query in queries:
            query_key = (name, query.strip())
            if query_key in seen_queries:
                errors.append(_error("DUPLICATE_CALL", "same tool cannot repeat the same query", {"name": name, "query": query}))
            seen_queries.add(query_key)
        top_k = args.get("top_k")
        if top_k != 5:
            errors.append(_error("INVALID_ARGUMENTS", "top_k must be exactly 5", {"name": name, "value": top_k}))
        if name == "search_core_memory":
            _validate_core(args, errors)
        elif name == "search_episodic_memory":
            _validate_episodic(args, errors)
    return ValidationResult(not errors, parsed, errors)


def _argument_queries(args: dict[str, Any]) -> list[str]:
    queries = args.get("queries")
    if not isinstance(queries, list):
        return []
    return [query.strip() for query in queries if isinstance(query, str) and query.strip()]


def _validate_core(args: dict[str, Any], errors: list[dict[str, Any]]) -> None:
    queries = args.get("queries")
    if not isinstance(queries, list) or not queries or not all(isinstance(query, str) and query.strip() for query in queries):
        errors.append(_error("INVALID_ARGUMENTS", "queries must be a non-empty string array", {"name": "search_core_memory"}))
    fields = args.get("fields")
    if not isinstance(fields, list) or not fields or not all(field in CORE_FIELDS for field in fields):
        errors.append(_error("INVALID_ARGUMENTS", "fields must be a non-empty core field array", {"fields": fields}))


def _validate_episodic(args: dict[str, Any], errors: list[dict[str, Any]]) -> None:
    queries = args.get("queries")
    if not isinstance(queries, list) or not queries or not all(isinstance(query, str) and query.strip() for query in queries):
        errors.append(_error("INVALID_ARGUMENTS", "queries must be a non-empty string array", {"name": "search_episodic_memory"}))
