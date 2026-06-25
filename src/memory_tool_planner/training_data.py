from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .locomo_planner_data import read_jsonl, write_jsonl


DEFAULT_CURRENT_DATE = "2026-05-08"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DATA_SOURCE = "locomo_memory_tool_planner"
ABILITY = "memory_tool_planning"


PLANNER_PROMPT_TEMPLATE = """You are the Memory Tool Planner.

Your job is to decide which memory tools should be called before a frozen answer
model answers the user question. You must not answer the user question yourself.

Output rules:
1. Return JSON only.
2. The top-level JSON object must contain only "tool_calls".
3. Use only these tool names and argument fields:
   - search_core_memory: arguments = {{"queries": string[], "fields": string[]}}
   - search_episodic_memory: arguments = {{"queries": string[]}}
4. Do not invent memory content, evidence ids, or final answers.
5. Use an empty tool_calls array when no personalized memory is needed.
6. Do not output call_id, purpose, or top_k. The executor assigns those runtime fields.
7. If planner_input.person or planner_input.persons is non-empty, include the exact person name in each related core or episodic query.
8. Each tool call object must use the key "name"; never use "tool_name".
9. For search_core_memory, fields must be selected only from: allergy, constraint, goal, identity, preference, relationship.

Valid output shape example:
{{"tool_calls":[{{"name":"search_episodic_memory","arguments":{{"queries":["Melanie sunrise painting"]}}}}]}}

Planner input:
{planner_input}

JSON output:
"""


def question_id(row: dict[str, Any], index: int = 0) -> str:
    existing = row.get("question_id")
    if existing:
        return str(existing)
    conv = row.get("conversation_index", 0)
    qa = row.get("qa_index", index)
    return f"locomo_{conv}_qa{qa}"


def planner_input_payload(row: dict[str, Any], *, current_date: str = DEFAULT_CURRENT_DATE, timezone: str = DEFAULT_TIMEZONE) -> dict[str, Any]:
    planner_input = row.get("planner_input") if isinstance(row.get("planner_input"), dict) else {}
    del timezone
    return {
        "question": row.get("question", planner_input.get("question", "")),
        "person": row.get("person", planner_input.get("person", "")),
        "persons": row.get("persons", planner_input.get("persons", [])),
        "current_date": row.get("current_date", current_date),
    }


def planner_prompt(row: dict[str, Any], *, current_date: str = DEFAULT_CURRENT_DATE, timezone: str = DEFAULT_TIMEZONE) -> str:
    payload = planner_input_payload(row, current_date=current_date, timezone=timezone)
    return PLANNER_PROMPT_TEMPLATE.format(planner_input=json.dumps(payload, ensure_ascii=False, indent=2))


def model_visible_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {"tool_calls": []}
    tool_calls = plan.get("tool_calls")
    if not isinstance(tool_calls, list):
        return {"tool_calls": []}
    visible_calls: list[Any] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            visible_calls.append(call)
            continue
        visible_call = {key: value for key, value in call.items() if key not in {"call_id", "purpose"}}
        args = visible_call.get("arguments")
        if isinstance(args, dict):
            visible_call["arguments"] = {key: value for key, value in args.items() if key != "top_k"}
        visible_calls.append(visible_call)
    return {"tool_calls": visible_calls}


def planner_target_text(row: dict[str, Any]) -> str:
    require_teacher_label(row)
    target = row.get("planner_target", {"tool_calls": []})
    return json.dumps(model_visible_plan(target), ensure_ascii=False, separators=(",", ":"))


def require_teacher_label(row: dict[str, Any]) -> None:
    if row.get("label_source") != "teacher_distilled":
        raise ValueError("SFT data requires teacher_distilled planner labels; run generate-teacher-sft-data first")
    if "planner_target" not in row:
        raise ValueError("SFT data requires planner_target from teacher distillation")


def common_extra_info(row: dict[str, Any], index: int, *, stores_path: str | None = None) -> dict[str, Any]:
    extra = {
        "question_id": question_id(row, index),
        "sample_id": row.get("sample_id"),
        "conversation_index": row.get("conversation_index"),
        "qa_index": row.get("qa_index"),
        "split": row.get("split", "all"),
        "category": row.get("category"),
        "evidence": row.get("evidence", []),
    }
    if stores_path:
        extra["stores_path"] = stores_path
    return extra


def sft_row(row: dict[str, Any], index: int, *, current_date: str = DEFAULT_CURRENT_DATE, timezone: str = DEFAULT_TIMEZONE) -> dict[str, Any]:
    require_teacher_label(row)
    prompt = planner_prompt(row, current_date=current_date, timezone=timezone)
    response = planner_target_text(row)
    return {
        "data_source": DATA_SOURCE,
        "ability": ABILITY,
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "enable_thinking": False,
        "input": planner_input_payload(row, current_date=current_date, timezone=timezone),
        "target": model_visible_plan(row.get("planner_target", {"tool_calls": []})),
        "extra_info": common_extra_info(row, index),
    }


def rl_row(
    row: dict[str, Any],
    index: int,
    *,
    stores_path: str | None = None,
    current_date: str = DEFAULT_CURRENT_DATE,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    ground_truth = {
        "question_id": question_id(row, index),
        "question": row.get("question", ""),
        "gold_answer": row.get("answer"),
        "gold_evidence_ids": row.get("evidence", []),
        "category": row.get("category"),
        "conversation_index": row.get("conversation_index"),
        "qa_index": row.get("qa_index"),
        "split": row.get("split", "all"),
        "teacher_planner_target": row.get("planner_target", {"tool_calls": []}),
    }
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": planner_prompt(row, current_date=current_date, timezone=timezone)}],
        "ability": ABILITY,
        "reward_model": {
            "style": "custom",
            "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
        },
        "extra_info": common_extra_info(row, index, stores_path=stores_path),
    }


def write_rows(path: str | Path, rows: list[dict[str, Any]], output_format: str = "auto") -> None:
    output = Path(path)
    fmt = output_format
    if fmt == "auto":
        fmt = "parquet" if output.suffix == ".parquet" else "jsonl"
    if fmt == "jsonl":
        write_jsonl(output, rows)
        return
    if fmt != "parquet":
        raise ValueError("output_format must be one of: auto, jsonl, parquet")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Writing parquet requires pyarrow. Install it in the training environment.") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)


def export_sft_data(
    planner_data_path: str | Path,
    output_path: str | Path,
    *,
    output_format: str = "auto",
    current_date: str = DEFAULT_CURRENT_DATE,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    rows = read_jsonl(planner_data_path)
    exported = [sft_row(row, index, current_date=current_date, timezone=timezone) for index, row in enumerate(rows)]
    write_rows(output_path, exported, output_format=output_format)
    return {"output": str(output_path), "format": output_format, "samples": len(exported), "type": "sft"}


def export_rl_data(
    planner_data_path: str | Path,
    output_path: str | Path,
    *,
    stores_path: str | Path | None = None,
    output_format: str = "auto",
    current_date: str = DEFAULT_CURRENT_DATE,
    timezone: str = DEFAULT_TIMEZONE,
    drop_empty_evidence: bool = False,
) -> dict[str, Any]:
    rows = read_jsonl(planner_data_path)
    if drop_empty_evidence:
        rows = [row for row in rows if row.get("evidence")]
    stores = str(stores_path) if stores_path else None
    exported = [rl_row(row, index, stores_path=stores, current_date=current_date, timezone=timezone) for index, row in enumerate(rows)]
    write_rows(output_path, exported, output_format=output_format)
    return {
        "output": str(output_path),
        "format": output_format,
        "samples": len(exported),
        "type": "rl",
        "drop_empty_evidence": drop_empty_evidence,
    }
