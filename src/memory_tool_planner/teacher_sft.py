from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .context import build_memory_context
from .evidence import row_conversation_index
from .locomo_planner_data import read_jsonl, retrieved_evidence, write_jsonl
from .openai_protocol import OpenAIProtocolClient
from .planner import validate_planner_output
from .schema import CORE_FIELDS
from .retrieval import MemoryToolExecutor
from .training_data import DEFAULT_CURRENT_DATE, DEFAULT_TIMEZONE, PLANNER_PROMPT_TEMPLATE, planner_input_payload


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("teacher response is not a JSON object")


def canonicalize_teacher_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize common teacher deviations into the model-visible planner schema."""
    tool_calls = plan.get("tool_calls")
    if not isinstance(tool_calls, list):
        return plan
    field_aliases = {
        "allergies": "allergy",
        "constraints": "constraint",
        "facts": "identity",
        "fact": "identity",
        "goals": "goal",
        "identities": "identity",
        "preferences": "preference",
        "relationships": "relationship",
    }
    normalized_calls: list[Any] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            normalized_calls.append(call)
            continue
        normalized = dict(call)
        if "name" not in normalized and isinstance(normalized.get("tool_name"), str):
            normalized["name"] = normalized["tool_name"]
        normalized.pop("tool_name", None)
        args = normalized.get("arguments")
        if isinstance(args, dict):
            normalized_args = dict(args)
            query = normalized_args.pop("query", None)
            if "queries" not in normalized_args and isinstance(query, str) and query.strip():
                normalized_args["queries"] = [query.strip()]
            if normalized.get("name") == "search_core_memory":
                fields = normalized_args.get("fields", [])
                if not isinstance(fields, list):
                    fields = []
                canonical_fields = []
                for field in fields:
                    if not isinstance(field, str):
                        continue
                    value = field_aliases.get(field.strip().lower(), field.strip().lower())
                    if value in CORE_FIELDS and value not in canonical_fields:
                        canonical_fields.append(value)
                normalized_args["fields"] = canonical_fields or sorted(CORE_FIELDS)
            normalized["arguments"] = normalized_args
        normalized_calls.append(normalized)
    output = dict(plan)
    output["tool_calls"] = normalized_calls
    return output


def build_teacher_prompt(
    row: dict[str, Any],
    *,
    current_date: str = DEFAULT_CURRENT_DATE,
    timezone: str = DEFAULT_TIMEZONE,
) -> str:
    payload = planner_input_payload(row, current_date=current_date, timezone=timezone)
    return PLANNER_PROMPT_TEMPLATE.format(planner_input=json.dumps(payload, ensure_ascii=False, indent=2))


def evidence_score(gold_evidence: list[Any], tool_results: dict[str, Any]) -> float:
    gold = {str(item) for item in gold_evidence if item is not None}
    if not gold:
        return 0.0
    overlap = gold & retrieved_evidence(tool_results)
    return len(overlap) / len(gold)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scoped_conversation_index(row: dict[str, Any], executor: MemoryToolExecutor) -> int | None:
    conversation_index = _optional_int(row.get("conversation_index"))
    if conversation_index is None:
        return None
    if not any(row_conversation_index(memory) is not None for memory in executor.memories):
        return None
    return conversation_index


def score_teacher_candidate(
    row: dict[str, Any],
    raw_output: str,
    executor: MemoryToolExecutor,
    *,
    token_budget: int = 0,
    max_memory_chars: int | None = None,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {"raw_output": raw_output}
    try:
        parsed = canonicalize_teacher_plan(parse_json_object(raw_output))
    except Exception as exc:
        candidate.update({"valid": False, "errors": [{"code": "INVALID_JSON", "message": str(exc), "details": {}}], "score": -1.0})
        return candidate
    validation = validate_planner_output(parsed)
    candidate["parsed"] = validation.parsed
    candidate["valid"] = validation.valid
    candidate["errors"] = validation.errors
    if not validation.valid:
        candidate["score"] = -1.0
        return candidate
    tool_results = executor.execute_plan(
        validation.parsed or {"tool_calls": []},
        conversation_index=_scoped_conversation_index(row, executor),
    )
    memory_context = build_memory_context(tool_results, token_budget=token_budget, max_memory_chars=max_memory_chars)
    score = evidence_score(row.get("evidence", []), tool_results)
    candidate.update(
        {
            "score": score,
            "tool_results": tool_results,
            "memory_context": memory_context,
            "retrieval_eval": {
                "gold_evidence": [str(item) for item in row.get("evidence", []) if item is not None],
                "retrieved_evidence": sorted(retrieved_evidence(tool_results)),
                "evidence_score": score,
            },
        }
    )
    return candidate


def generate_teacher_sft_data(
    planner_data_path: str | Path,
    output_path: str | Path,
    stores_path: str | Path,
    client: OpenAIProtocolClient,
    *,
    candidates_per_sample: int = 4,
    limit: int | None = None,
    temperature: float = 0.4,
    current_date: str = DEFAULT_CURRENT_DATE,
    timezone: str = DEFAULT_TIMEZONE,
    token_budget: int = 0,
    max_memory_chars: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(planner_data_path)
    if limit is not None:
        rows = rows[:limit]
    executor = MemoryToolExecutor.from_jsonl(stores_path)
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    model_name = client.settings.chat_model
    for index, row in enumerate(rows):
        prompt = build_teacher_prompt(row, current_date=current_date, timezone=timezone)
        candidates = []
        for _ in range(max(1, candidates_per_sample)):
            raw_output = client.chat([{"role": "user", "content": prompt}], temperature=temperature)
            candidates.append(
                score_teacher_candidate(
                    row,
                    raw_output,
                    executor,
                    token_budget=token_budget,
                    max_memory_chars=max_memory_chars,
                )
            )
        valid_candidates = [item for item in candidates if item.get("valid")]
        if not valid_candidates:
            failures.append({"index": index, "question": row.get("question", ""), "candidate_errors": candidates})
            continue
        selected = max(valid_candidates, key=lambda item: float(item.get("score", -1.0)))
        output_row = dict(row)
        output_row["planner_target"] = selected.get("parsed", {"tool_calls": []})
        output_row["tool_results"] = selected.get("tool_results", {"tool_results": []})
        output_row["memory_context"] = selected.get("memory_context", {})
        output_row["retrieval_eval"] = selected.get("retrieval_eval", {})
        output_row["label_source"] = "teacher_distilled"
        output_row["teacher_model"] = model_name
        output_row["teacher_selected"] = {
            "candidate_index": candidates.index(selected),
            "score": selected.get("score", 0.0),
            "raw_output": selected.get("raw_output", ""),
        }
        output_row["teacher_candidates"] = [
            {
                "valid": item.get("valid", False),
                "score": item.get("score", -1.0),
                "errors": item.get("errors", []),
                "raw_output": item.get("raw_output", ""),
            }
            for item in candidates
        ]
        output_rows.append(output_row)
    if failures:
        raise RuntimeError(f"teacher failed to produce valid planner labels for {len(failures)} samples; first failure: {failures[0]}")
    write_jsonl(output_path, output_rows)
    scores = [float(row.get("retrieval_eval", {}).get("evidence_score", 0.0)) for row in output_rows]
    return {
        "output": str(output_path),
        "samples": len(output_rows),
        "teacher_model": model_name,
        "candidates_per_sample": max(1, candidates_per_sample),
        "avg_evidence_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "label_source": "teacher_distilled",
    }
