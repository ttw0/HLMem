from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .answer_eval import write_json, write_jsonl
from .locomo_planner_data import read_jsonl
from .retrieval import MemoryToolExecutor
from .reward import evaluate_planner_output, summarize_reward_rows
from .schema import CORE_FIELDS


BASELINE_NAMES = {"no_memory", "naive_rag", "oracle_planner"}


def no_memory_plan(row: dict[str, Any]) -> dict[str, Any]:
    del row
    return {"tool_calls": []}


def naive_rag_plan(row: dict[str, Any]) -> dict[str, Any]:
    query = str(row.get("question", "")).strip()
    person = str(row.get("person", "")).strip()
    if person and person not in query:
        query = f"{person} {query}".strip()
    if not query:
        return {"tool_calls": []}
    return {
        "tool_calls": [
            {
                "name": "search_core_memory",
                "arguments": {
                    "queries": [query],
                    "fields": sorted(CORE_FIELDS),
                },
            },
            {
                "name": "search_episodic_memory",
                "arguments": {
                    "queries": [query],
                },
            },
        ]
    }


def _build_evidence_index(executor: MemoryToolExecutor) -> dict[str, list[dict[str, Any]]]:
    by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for memory in executor.memories:
        for evidence_id in memory.get("evidence_ids", []):
            by_evidence[str(evidence_id)].append(memory)
    return by_evidence


def _memory_query(memory: dict[str, Any], fallback: str) -> str:
    parts = [
        memory.get("entity"),
        memory.get("subject"),
        memory.get("event"),
        memory.get("content"),
        memory.get("summary"),
        memory.get("time"),
        memory.get("location"),
        memory.get("value"),
    ]
    text = " ".join(str(part) for part in parts if part).strip()
    return text or fallback


def oracle_planner_plan(row: dict[str, Any], executor: MemoryToolExecutor, max_calls_per_tool: int = 2) -> dict[str, Any]:
    evidence_index = _build_evidence_index(executor)
    question = str(row.get("question", "")).strip()
    calls_by_tool: dict[str, list[str]] = {"search_core_memory": [], "search_episodic_memory": []}
    core_fields: set[str] = set()
    for evidence_id in row.get("evidence", []) or []:
        for memory in evidence_index.get(str(evidence_id), []):
            store = memory.get("store")
            if store == "core":
                query = _memory_query(memory, question)
                if query and query not in calls_by_tool["search_core_memory"]:
                    calls_by_tool["search_core_memory"].append(query)
                field = memory.get("field")
                if isinstance(field, str) and field in CORE_FIELDS:
                    core_fields.add(field)
            elif store == "episodic":
                query = _memory_query(memory, question)
                if query and query not in calls_by_tool["search_episodic_memory"]:
                    calls_by_tool["search_episodic_memory"].append(query)

    tool_calls: list[dict[str, Any]] = []
    core_queries = calls_by_tool["search_core_memory"][:max_calls_per_tool]
    if core_queries:
        tool_calls.append(
            {
                "name": "search_core_memory",
                "arguments": {
                    "queries": core_queries,
                    "fields": sorted(core_fields) or sorted(CORE_FIELDS),
                },
            }
        )
    episodic_queries = calls_by_tool["search_episodic_memory"][:max_calls_per_tool]
    if episodic_queries:
        tool_calls.append(
            {
                "name": "search_episodic_memory",
                "arguments": {
                    "queries": episodic_queries,
                },
            }
        )
    return {"tool_calls": tool_calls[:4]}


def baseline_plan(name: str, row: dict[str, Any], executor: MemoryToolExecutor) -> dict[str, Any]:
    if name == "no_memory":
        return no_memory_plan(row)
    if name == "naive_rag":
        return naive_rag_plan(row)
    if name == "oracle_planner":
        return oracle_planner_plan(row, executor)
    raise ValueError(f"unknown planner baseline: {name}")


def run_planner_baselines(
    planner_data_path: str | Path,
    stores_path: str | Path,
    output_dir: str | Path,
    *,
    baselines: list[str] | None = None,
    summary_path: str | Path | None = None,
    limit: int | None = None,
    eval_mode: str = "lexical",
    token_budget: int = 0,
    skip_generation: bool = True,
) -> dict[str, Any]:
    os.environ.setdefault("MTP_SKIP_ANSWER_GENERATION", "1")
    os.environ.setdefault("MTP_REWARD_ANSWER_WEIGHT", "0.0")
    os.environ.setdefault("MTP_REWARD_EVIDENCE_WEIGHT", "1.0")
    os.environ.setdefault("MTP_REWARD_PARTIAL_ANSWER_WEIGHT", "0.0")
    os.environ.setdefault("MTP_REWARD_TOKEN_COST_WEIGHT", "0.0")
    os.environ.setdefault("MTP_REWARD_TOOL_COST_WEIGHT", "0.0")
    os.environ.setdefault("MTP_REWARD_UNDER_TOOL_WEIGHT", "0.0")
    names = baselines or ["no_memory", "naive_rag", "oracle_planner"]
    unknown = [name for name in names if name not in BASELINE_NAMES]
    if unknown:
        raise ValueError(f"unknown planner baselines: {', '.join(unknown)}")
    rows = read_jsonl(planner_data_path)
    if limit is not None:
        rows = rows[:limit]
    executor = MemoryToolExecutor.from_jsonl(stores_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "planner_data": str(planner_data_path),
        "stores": str(stores_path),
        "output_dir": str(out_dir),
        "baselines": {},
    }
    for name in names:
        traces = []
        for row in rows:
            plan = baseline_plan(name, row, executor)
            trace = evaluate_planner_output(
                row,
                plan,
                executor,
                eval_mode=eval_mode,
                token_budget=token_budget,
                skip_generation=skip_generation,
            )
            trace["baseline_name"] = name
            trace["planner_raw_output"] = json.dumps(plan, ensure_ascii=False)
            traces.append(trace)
        trace_path = out_dir / f"{name}.reward_traces.jsonl"
        write_jsonl(trace_path, traces)
        baseline_summary = summarize_reward_rows(traces, eval_mode)
        baseline_summary["trace"] = str(trace_path)
        summary["baselines"][name] = baseline_summary
    if summary_path:
        write_json(summary_path, summary)
    return summary
