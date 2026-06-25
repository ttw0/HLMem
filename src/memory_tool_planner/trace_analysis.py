from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .answer_eval import write_json, write_jsonl
from .locomo_planner_data import read_jsonl


REWARD_ABLATIONS = {
    "current": {"answer": 1.0, "evidence": 0.5, "token": 0.0, "tool": 0.0},
    "cost_off": {"answer": 1.0, "evidence": 0.5, "token": 0.0, "tool": 0.0},
    "low_cost": {"answer": 1.0, "evidence": 0.5, "token": 0.05, "tool": 0.0},
    "answer_heavy": {"answer": 1.5, "evidence": 0.5, "token": 0.05, "tool": 0.0},
    "evidence_heavy": {"answer": 1.0, "evidence": 1.0, "token": 0.05, "tool": 0.0},
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round(value: float) -> float:
    return round(value, 6)


def _question_id(row: dict[str, Any]) -> str:
    value = row.get("question_id")
    if value is None:
        value = f"{row.get('conversation_index', '')}:{row.get('qa_index', '')}:{row.get('question', '')}"
    return str(value)


def dedupe_trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last row for each question_id, matching current result docs."""
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        qid = _question_id(row)
        if qid not in by_id:
            order.append(qid)
        by_id[qid] = row
    return [by_id[qid] for qid in order]


def answer_score(row: dict[str, Any]) -> float:
    reward = row.get("reward", {})
    if isinstance(reward, dict) and "answer" in reward:
        try:
            return float(reward.get("answer", 0.0))
        except (TypeError, ValueError):
            return 0.0
    eval_result = row.get("eval", {})
    if "llm_correct" in eval_result:
        return 1.0 if eval_result.get("llm_correct") else 0.0
    if eval_result.get("exact_match") or eval_result.get("contains_match"):
        return 1.0
    return 0.0


def evidence_score(row: dict[str, Any]) -> float:
    reward = row.get("reward", {})
    if isinstance(reward, dict) and "evidence" in reward:
        try:
            return float(reward.get("evidence", 0.0))
        except (TypeError, ValueError):
            return 0.0
    gold = {str(item) for item in row.get("gold_evidence_ids", []) if item is not None}
    retrieved = {str(item) for item in row.get("retrieved_evidence_ids", []) if item is not None}
    return len(gold & retrieved) / len(gold) if gold else 0.0


def memory_tokens(row: dict[str, Any]) -> int:
    try:
        return int(row.get("memory_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def tool_count(row: dict[str, Any]) -> int:
    try:
        return int(row.get("tool_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    output = row.get("planner_output")
    if not isinstance(output, dict):
        return []
    calls = output.get("tool_calls", [])
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def tool_names(row: dict[str, Any]) -> list[str]:
    return [str(call.get("name", "")) for call in tool_calls(row) if call.get("name")]


def duplicate_query_count(row: dict[str, Any]) -> int:
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for call in tool_calls(row):
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        raw_queries = args.get("queries")
        if isinstance(raw_queries, list):
            queries = [query.strip() for query in raw_queries if isinstance(query, str) and query.strip()]
        else:
            query = args.get("query")
            queries = [query.strip()] if isinstance(query, str) and query.strip() else []
        for query in queries:
            key = (str(call.get("name", "")), query)
            if key in seen:
                duplicates += 1
            seen.add(key)
    return duplicates


def normalize_query(query: str) -> str:
    lowered = query.lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered)).strip()


def normalized_duplicate_query_count(row: dict[str, Any]) -> int:
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for call in tool_calls(row):
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        raw_queries = args.get("queries")
        if isinstance(raw_queries, list):
            queries = [query for query in raw_queries if isinstance(query, str) and query.strip()]
        else:
            query = args.get("query")
            queries = [query] if isinstance(query, str) and query.strip() else []
        for query in queries:
            normalized = normalize_query(query)
            if not normalized:
                continue
            key = (str(call.get("name", "")), normalized)
            if key in seen:
                duplicates += 1
            seen.add(key)
    return duplicates


def evidence_hit(row: dict[str, Any]) -> bool:
    return evidence_score(row) > 0.0


def full_evidence_hit(row: dict[str, Any]) -> bool:
    score = evidence_score(row)
    return score >= 1.0 if row.get("gold_evidence_ids") else False


def planner_failure_bucket(row: dict[str, Any]) -> str:
    if not row.get("planner_valid", True):
        return "invalid_plan"
    gold = {str(item) for item in row.get("gold_evidence_ids", []) if item is not None}
    retrieved = {str(item) for item in row.get("retrieved_evidence_ids", []) if item is not None}
    if not gold:
        return "no_gold_evidence"
    if not retrieved:
        return "no_retrieved_evidence"
    overlap = gold & retrieved
    if not overlap:
        names = set(tool_names(row))
        if "search_episodic_memory" not in names:
            return "no_overlap_no_episodic_call"
        if "search_core_memory" not in names:
            return "no_overlap_no_core_call"
        return "no_overlap"
    if len(overlap) < len(gold):
        return "partial_evidence"
    return "full_evidence"


def token_cost(row: dict[str, Any], memory_token_budget: int) -> float:
    return 0.0 if memory_token_budget <= 0 else memory_tokens(row) / memory_token_budget


def tool_cost(row: dict[str, Any], max_tool_calls: int) -> float:
    return tool_count(row) / max(1, max_tool_calls)


def ablated_reward(row: dict[str, Any], weights: dict[str, float], *, memory_token_budget: int, max_tool_calls: int) -> float:
    if not row.get("planner_valid", True):
        return -1.0
    return (
        weights["answer"] * answer_score(row)
        + weights["evidence"] * evidence_score(row)
        - weights["token"] * token_cost(row, memory_token_budget)
        - weights["tool"] * tool_cost(row, max_tool_calls)
    )


def summarize_rows(rows: list[dict[str, Any]], *, memory_token_budget: int, max_tool_calls: int) -> dict[str, Any]:
    rows = dedupe_trace_rows(rows)
    total = len(rows)
    valid_rows = [row for row in rows if row.get("planner_valid", True)]
    answer_values = [answer_score(row) for row in rows]
    evidence_values = [evidence_score(row) for row in rows]
    token_values = [float(memory_tokens(row)) for row in rows]
    tool_values = [float(tool_count(row)) for row in rows]
    reward_values = [float(row.get("reward", {}).get("total", 0.0)) for row in rows]
    tool_name_counts: Counter[str] = Counter()
    tool_count_distribution: Counter[str] = Counter()
    failure_buckets: Counter[str] = Counter()
    duplicate_query_rows = 0
    normalized_duplicate_query_rows = 0
    for row in rows:
        names = tool_names(row)
        tool_name_counts.update(names)
        tool_count_distribution[str(tool_count(row))] += 1
        failure_buckets[planner_failure_bucket(row)] += 1
        if duplicate_query_count(row) > 0:
            duplicate_query_rows += 1
        if normalized_duplicate_query_count(row) > 0:
            normalized_duplicate_query_rows += 1
    summary = {
        "samples": total,
        "planner_valid_rate": _round(len(valid_rows) / total) if total else 0.0,
        "answer_mean": _round(_mean(answer_values)),
        "evidence_mean": _round(_mean(evidence_values)),
        "evidence_hit_rate": _round(sum(1 for row in rows if evidence_hit(row)) / total) if total else 0.0,
        "full_evidence_rate": _round(sum(1 for row in rows if full_evidence_hit(row)) / total) if total else 0.0,
        "reward_mean": _round(_mean(reward_values)),
        "avg_memory_tokens": _round(_mean(token_values)),
        "avg_tool_count": _round(_mean(tool_values)),
        "tool_count_distribution": dict(sorted(tool_count_distribution.items(), key=lambda item: int(item[0]))),
        "tool_name_counts": dict(sorted(tool_name_counts.items())),
        "duplicate_query_rate": _round(duplicate_query_rows / total) if total else 0.0,
        "normalized_duplicate_query_rate": _round(normalized_duplicate_query_rows / total) if total else 0.0,
        "planner_failure_buckets": dict(sorted(failure_buckets.items())),
    }
    summary["reward_ablations"] = {
        name: _round(_mean([ablated_reward(row, weights, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls) for row in rows]))
        for name, weights in REWARD_ABLATIONS.items()
    }
    return summary


def planner_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = dedupe_trace_rows(rows)
    total = len(rows)
    valid_rows = [row for row in rows if row.get("planner_valid", True)]
    invalid_json = 0
    empty_calls = 0
    error_counts: Counter[str] = Counter()
    for row in rows:
        errors = row.get("planner_errors") or []
        if not row.get("planner_valid", True):
            if any((error.get("code") if isinstance(error, dict) else str(error)) == "INVALID_JSON" for error in errors):
                invalid_json += 1
        if tool_count(row) == 0:
            empty_calls += 1
        for error in errors:
            code = error.get("code") if isinstance(error, dict) else str(error)
            error_counts[str(code)] += 1
    return {
        "samples": total,
        "reward_mean": _round(_mean([float(row.get("reward", {}).get("total", 0.0)) for row in rows])),
        "evidence_mean": _round(_mean([evidence_score(row) for row in rows])),
        "hit_rate": _round(sum(1 for row in rows if evidence_hit(row)) / total) if total else 0.0,
        "full_hit_rate": _round(sum(1 for row in rows if full_evidence_hit(row)) / total) if total else 0.0,
        "valid_rate": _round(len(valid_rows) / total) if total else 0.0,
        "invalid_json": invalid_json,
        "empty_calls": empty_calls,
        "duplicate_query_rate": _round(sum(1 for row in rows if duplicate_query_count(row) > 0) / total) if total else 0.0,
        "normalized_duplicate_query_rate": _round(sum(1 for row in rows if normalized_duplicate_query_count(row) > 0) / total) if total else 0.0,
        "error_counts": dict(sorted(error_counts.items())),
    }


def assert_summary_matches_trace(summary: dict[str, Any], rows: list[dict[str, Any]], *, label: str = "trace") -> None:
    recomputed = planner_metric_summary(rows)
    metric_keys = [
        "samples",
        "reward_mean",
        "evidence_mean",
        "hit_rate",
        "full_hit_rate",
        "valid_rate",
        "invalid_json",
        "empty_calls",
        "duplicate_query_rate",
    ]
    mismatches = []
    for key in metric_keys:
        if key not in summary:
            continue
        expected = recomputed[key]
        actual = summary[key]
        if isinstance(expected, float):
            try:
                matches = abs(float(actual) - expected) <= 1e-6
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            mismatches.append({"metric": key, "summary": actual, "trace": expected})
    if mismatches:
        raise ValueError(f"{label} summary does not match trace: {json.dumps(mismatches, ensure_ascii=False)}")


def group_summary(rows: list[dict[str, Any]], *, memory_token_budget: int, max_tool_calls: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dedupe_trace_rows(rows):
        grouped[str(row.get("category", "unknown"))].append(row)
    return {
        group: summarize_rows(group_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls)
        for group, group_rows in sorted(grouped.items())
    }


def conversation_summary(rows: list[dict[str, Any]], *, memory_token_budget: int, max_tool_calls: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dedupe_trace_rows(rows):
        grouped[str(row.get("conversation_index", "unknown"))].append(row)
    return {
        group: summarize_rows(group_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls)
        for group, group_rows in sorted(grouped.items(), key=lambda item: item[0])
    }


def tool_set_summary(rows: list[dict[str, Any]], *, memory_token_budget: int, max_tool_calls: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dedupe_trace_rows(rows):
        names = sorted(set(tool_names(row)))
        key = "+".join(names) if names else "none"
        grouped[key].append(row)
    return {
        group: summarize_rows(group_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls)
        for group, group_rows in sorted(grouped.items())
    }


def evidence_sets(row: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    gold = {str(item) for item in row.get("gold_evidence_ids", []) if item is not None}
    retrieved = {str(item) for item in row.get("retrieved_evidence_ids", []) if item is not None}
    return gold, retrieved, gold & retrieved


def compact_tool_queries(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for call in tool_calls(row):
        args = call.get("arguments", {})
        queries = args.get("queries") if isinstance(args, dict) else []
        items.append(
            {
                "name": call.get("name"),
                "queries": [query for query in queries if isinstance(query, str)] if isinstance(queries, list) else [],
            }
        )
    return items


def paired_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    baseline_name: str,
    candidate_name: str,
    memory_token_budget: int,
    max_tool_calls: int,
) -> list[dict[str, Any]]:
    baseline_by_id = {_question_id(row): row for row in dedupe_trace_rows(baseline_rows)}
    candidate_by_id = {_question_id(row): row for row in dedupe_trace_rows(candidate_rows)}
    rows = []
    for qid in sorted(set(baseline_by_id) & set(candidate_by_id)):
        base = baseline_by_id[qid]
        cand = candidate_by_id[qid]
        base_answer = answer_score(base)
        cand_answer = answer_score(cand)
        base_evidence = evidence_score(base)
        cand_evidence = evidence_score(cand)
        item = {
            "question_id": qid,
            "category": cand.get("category", base.get("category")),
            "question": cand.get("question", base.get("question", "")),
            f"{baseline_name}_answer": base_answer,
            f"{candidate_name}_answer": cand_answer,
            "answer_delta": _round(cand_answer - base_answer),
            f"{baseline_name}_evidence": base_evidence,
            f"{candidate_name}_evidence": cand_evidence,
            "evidence_delta": _round(cand_evidence - base_evidence),
            f"{baseline_name}_tool_count": tool_count(base),
            f"{candidate_name}_tool_count": tool_count(cand),
            "tool_count_delta": tool_count(cand) - tool_count(base),
            f"{baseline_name}_memory_tokens": memory_tokens(base),
            f"{candidate_name}_memory_tokens": memory_tokens(cand),
            "memory_tokens_delta": memory_tokens(cand) - memory_tokens(base),
            "outcome": paired_outcome(base_answer, cand_answer),
            "planner_outcome": paired_planner_outcome(base_evidence, cand_evidence),
            "missing_gold_evidence": sorted(evidence_sets(cand)[0] - evidence_sets(cand)[2]),
            f"{baseline_name}_tool_queries": compact_tool_queries(base),
            f"{candidate_name}_tool_queries": compact_tool_queries(cand),
            "reward_ablations": {},
        }
        for name, weights in REWARD_ABLATIONS.items():
            base_reward = ablated_reward(base, weights, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls)
            cand_reward = ablated_reward(cand, weights, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls)
            item["reward_ablations"][name] = {
                f"{baseline_name}_reward": _round(base_reward),
                f"{candidate_name}_reward": _round(cand_reward),
                "delta": _round(cand_reward - base_reward),
            }
        rows.append(item)
    return rows


def paired_outcome(baseline_answer: float, candidate_answer: float) -> str:
    if candidate_answer > baseline_answer:
        return "candidate_wins"
    if candidate_answer < baseline_answer:
        return "baseline_wins"
    if candidate_answer > 0:
        return "both_correct"
    return "both_wrong"


def paired_planner_outcome(baseline_evidence: float, candidate_evidence: float) -> str:
    if candidate_evidence > baseline_evidence:
        return "candidate_wins"
    if candidate_evidence < baseline_evidence:
        return "baseline_wins"
    if candidate_evidence > 0:
        return "both_hit"
    return "both_miss"


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    outcomes: dict[str, int] = defaultdict(int)
    planner_outcomes: dict[str, int] = defaultdict(int)
    for row in rows:
        outcomes[row["outcome"]] += 1
        planner_outcomes[row["planner_outcome"]] += 1
    by_ablation = {}
    for name in REWARD_ABLATIONS:
        deltas = [float(row["reward_ablations"][name]["delta"]) for row in rows]
        by_ablation[name] = {
            "avg_delta": _round(_mean(deltas)),
            "candidate_higher": sum(1 for value in deltas if value > 0),
            "baseline_higher": sum(1 for value in deltas if value < 0),
            "ties": sum(1 for value in deltas if value == 0),
        }
    return {
        "paired_samples": total,
        "outcomes": dict(sorted(outcomes.items())),
        "planner_outcomes": dict(sorted(planner_outcomes.items())),
        "candidate_win_rate": _round(outcomes["candidate_wins"] / total) if total else 0.0,
        "baseline_win_rate": _round(outcomes["baseline_wins"] / total) if total else 0.0,
        "planner_candidate_win_rate": _round(planner_outcomes["candidate_wins"] / total) if total else 0.0,
        "planner_baseline_win_rate": _round(planner_outcomes["baseline_wins"] / total) if total else 0.0,
        "reward_ablation_deltas": by_ablation,
        "avg_answer_delta": _round(_mean([float(row["answer_delta"]) for row in rows])),
        "avg_evidence_delta": _round(_mean([float(row["evidence_delta"]) for row in rows])),
        "avg_tool_count_delta": _round(_mean([float(row["tool_count_delta"]) for row in rows])),
        "avg_memory_tokens_delta": _round(_mean([float(row["memory_tokens_delta"]) for row in rows])),
    }


def run_trace_analysis(
    baseline_trace_path: str | Path,
    candidate_trace_path: str | Path,
    output_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    baseline_name: str = "sft",
    candidate_name: str = "grpo",
    memory_token_budget: int = 0,
    max_tool_calls: int = 4,
) -> dict[str, Any]:
    baseline_rows = read_jsonl(baseline_trace_path)
    candidate_rows = read_jsonl(candidate_trace_path)
    pairs = paired_rows(
        baseline_rows,
        candidate_rows,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        memory_token_budget=memory_token_budget,
        max_tool_calls=max_tool_calls,
    )
    write_jsonl(output_path, pairs)
    summary = {
        "baseline": {
            "name": baseline_name,
            "trace": str(baseline_trace_path),
            **summarize_rows(baseline_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls),
            "by_category": group_summary(baseline_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls),
            "by_conversation": conversation_summary(baseline_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls),
            "by_tool_set": tool_set_summary(baseline_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls),
        },
        "candidate": {
            "name": candidate_name,
            "trace": str(candidate_trace_path),
            **summarize_rows(candidate_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls),
            "by_category": group_summary(candidate_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls),
            "by_conversation": conversation_summary(candidate_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls),
            "by_tool_set": tool_set_summary(candidate_rows, memory_token_budget=memory_token_budget, max_tool_calls=max_tool_calls),
        },
        "paired": summarize_pairs(pairs),
        "output": str(output_path),
    }
    if summary_path:
        write_json(summary_path, summary)
    return summary


def summary_to_text(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2)
