from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback for local development.
    fcntl = None

from .answer import answer_question
from .answer_eval import evaluate_answer_row, write_json, write_jsonl
from .config import OpenAISettings
from .context import build_memory_context
from .locomo_planner_data import read_jsonl, retrieved_evidence
from .openai_protocol import OpenAIProtocolClient
from .planner import validate_planner_output
from .retrieval import MemoryToolExecutor

_TOKENIZER: Any | None = None
_TOKENIZER_SOURCE: str | None = None
_TOKENIZER_LOAD_ERROR: str | None = None


DEFAULT_REWARD_WEIGHTS = {
    "answer": 1.0,
    "evidence": 0.5,
    "token": 0.0,
    "tool": 0.0,
    "under_tool": 0.0,
}


def reward_weights_from_env() -> dict[str, float]:
    def get_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        return float(raw)

    return {
        "answer": get_float("MTP_REWARD_ANSWER_WEIGHT", DEFAULT_REWARD_WEIGHTS["answer"]),
        "evidence": get_float("MTP_REWARD_EVIDENCE_WEIGHT", DEFAULT_REWARD_WEIGHTS["evidence"]),
        "token": get_float("MTP_REWARD_TOKEN_COST_WEIGHT", DEFAULT_REWARD_WEIGHTS["token"]),
        "tool": get_float("MTP_REWARD_TOOL_COST_WEIGHT", DEFAULT_REWARD_WEIGHTS["tool"]),
        "under_tool": get_float("MTP_REWARD_UNDER_TOOL_WEIGHT", DEFAULT_REWARD_WEIGHTS["under_tool"]),
    }


def min_tool_count_from_env() -> int:
    raw = os.environ.get("MTP_REWARD_MIN_TOOL_COUNT")
    if raw is None or raw == "":
        return 0
    return max(0, int(raw))


def partial_answer_weight_from_env() -> float:
    raw = os.environ.get("MTP_REWARD_PARTIAL_ANSWER_WEIGHT")
    if raw is None or raw == "":
        return 0.25
    return max(0.0, float(raw))


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def qwen3_tokenizer_name_from_env() -> str:
    return (
        os.environ.get("MTP_QWEN3_TOKENIZER_NAME_OR_PATH")
        or os.environ.get("MTP_TOKENIZER_NAME_OR_PATH")
        or os.environ.get("TOKENIZER_NAME_OR_PATH")
        or os.environ.get("ACTOR_MODEL_PATH")
        or "Qwen/Qwen3-8B"
    )


def _load_qwen3_tokenizer() -> Any | None:
    global _TOKENIZER, _TOKENIZER_SOURCE, _TOKENIZER_LOAD_ERROR

    source = qwen3_tokenizer_name_from_env()
    if _TOKENIZER is not None and _TOKENIZER_SOURCE == source:
        return _TOKENIZER
    if _TOKENIZER_LOAD_ERROR and _TOKENIZER_SOURCE == source:
        return None

    _TOKENIZER_SOURCE = source
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        _TOKENIZER_LOAD_ERROR = None
        return _TOKENIZER
    except Exception as exc:  # pragma: no cover - runtime dependency boundary.
        _TOKENIZER = None
        _TOKENIZER_LOAD_ERROR = str(exc)
        return None


def estimate_tokens_rough(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    rough_cjk_chars = sum(1 for char in stripped if "\u4e00" <= char <= "\u9fff")
    word_tokens = len(stripped.split())
    char_tokens = len(stripped) // 4
    return max(1, rough_cjk_chars + max(word_tokens, char_tokens))


def estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    tokenizer = _load_qwen3_tokenizer()
    if tokenizer is None:
        return estimate_tokens_rough(stripped)
    return len(tokenizer.encode(stripped, add_special_tokens=False))


def evidence_recall(gold_evidence: list[Any], tool_results: dict[str, Any] | None) -> float:
    gold = {str(item) for item in gold_evidence if item is not None}
    if not gold or not tool_results:
        return 0.0
    retrieved = retrieved_evidence(tool_results)
    return len(gold & retrieved) / len(gold)


def answer_reward(eval_result: dict[str, Any], eval_mode: str) -> float:
    if eval_mode in {"llm", "hybrid"} and "llm_correct" in eval_result:
        if eval_result.get("llm_correct"):
            return 1.0
        return partial_answer_weight_from_env() * float(eval_result.get("token_f1", 0.0) or 0.0)
    if eval_result.get("exact_match") or eval_result.get("contains_match"):
        return 1.0
    return partial_answer_weight_from_env() * float(eval_result.get("token_f1", 0.0) or 0.0)


def reward_breakdown(
    *,
    valid: bool,
    answer_score: float = 0.0,
    evidence_score: float = 0.0,
    memory_tokens: int = 0,
    tool_count: int = 0,
    memory_token_budget: int = 0,
    weights: dict[str, float] | None = None,
    min_tool_count: int | None = None,
) -> dict[str, Any]:
    reward_weights = weights or DEFAULT_REWARD_WEIGHTS
    required_tool_count = min_tool_count_from_env() if min_tool_count is None else max(0, min_tool_count)
    if not valid:
        return {
            "answer": 0.0,
            "evidence": 0.0,
            "token_cost": 0.0,
            "tool_cost": 0.0,
            "under_tool_penalty": 0.0,
            "invalid_penalty": 1.0,
            "weights": {key: round(value, 4) for key, value in reward_weights.items()},
            "min_tool_count": required_tool_count,
            "total": -1.0,
        }
    token_cost = 0.0 if memory_token_budget <= 0 else memory_tokens / memory_token_budget
    tool_cost = tool_count / 4
    under_tool_penalty = max(0, required_tool_count - tool_count)
    total = (
        reward_weights["answer"] * answer_score
        + reward_weights["evidence"] * evidence_score
        - reward_weights["token"] * token_cost
        - reward_weights["tool"] * tool_cost
        - reward_weights.get("under_tool", 0.0) * under_tool_penalty
    )
    return {
        "answer": round(answer_score, 4),
        "evidence": round(evidence_score, 4),
        "token_cost": round(token_cost, 4),
        "tool_cost": round(tool_cost, 4),
        "under_tool_penalty": round(under_tool_penalty, 4),
        "invalid_penalty": 0.0,
        "weights": {key: round(value, 4) for key, value in reward_weights.items()},
        "min_tool_count": required_tool_count,
        "total": round(total, 4),
    }


def judge_client_for_mode(eval_mode: str) -> OpenAIProtocolClient | None:
    if eval_mode not in {"llm", "hybrid"}:
        return None
    settings = OpenAISettings.from_env("JUDGE")
    if settings.offline:
        return None
    return OpenAIProtocolClient(settings)


def build_eval_source(row: dict[str, Any], ground_truth: dict[str, Any] | None = None) -> dict[str, Any]:
    source = ground_truth or {}
    conversation_index = source.get("conversation_index", row.get("conversation_index"))
    qa_index = source.get("qa_index", row.get("qa_index"))
    question_id = source.get("question_id", row.get("question_id"))
    if not question_id and conversation_index is not None and qa_index is not None:
        question_id = f"locomo_{conversation_index}_qa{qa_index}"
    return {
        "question": source.get("question", row.get("question", "")),
        "answer": source.get("gold_answer", row.get("answer")),
        "evidence": source.get("gold_evidence_ids", row.get("evidence", [])),
        "category": source.get("category", row.get("category")),
        "question_id": question_id,
        "conversation_index": conversation_index,
        "qa_index": qa_index,
    }


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def evaluate_planner_output(
    row: dict[str, Any],
    planner_output: str | dict[str, Any],
    executor: MemoryToolExecutor,
    *,
    eval_mode: str = "lexical",
    judge_client: OpenAIProtocolClient | None = None,
    num_judge_runs: int = 1,
    token_budget: int = 0,
    max_memory_chars: int | None = None,
    prompt_style: str = "concise",
    skip_generation: bool = False,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    reward_weights = reward_weights_from_env()
    validation = validate_planner_output(planner_output)
    raw_output = stringify(planner_output)
    source = build_eval_source(row, ground_truth=ground_truth)
    if not validation.valid:
        return {
            "question_id": source.get("question_id"),
            "conversation_index": source.get("conversation_index"),
            "qa_index": source.get("qa_index"),
            "question": source["question"],
            "planner_raw_output": raw_output,
            "planner_valid": False,
            "planner_errors": validation.errors,
            "tool_results": {"tool_results": []},
            "memory_context": "",
            "generated_answer": "",
            "golden_answer": source.get("answer"),
            "eval": {},
            "reward": reward_breakdown(valid=False, weights=reward_weights),
            "duration_ms": int((time.time() - started) * 1000),
        }

    parsed = validation.parsed or {"tool_calls": []}
    conversation_index = source.get("conversation_index")
    tool_results = executor.execute_plan(
        parsed,
        conversation_index=_optional_int(conversation_index),
    )
    context_obj = build_memory_context(tool_results, token_budget=token_budget, max_memory_chars=max_memory_chars)
    memory_context = context_obj.get("memory_context", "")
    generated = ""
    answer_error = ""
    if not skip_generation:
        try:
            generated = answer_question(source["question"], memory_context, prompt_style=prompt_style)
        except Exception as exc:  # pragma: no cover - runtime boundary
            answer_error = str(exc)
    eval_row = {
        "question": source["question"],
        "generated_answer": generated,
        "golden_answer": source.get("answer"),
        "category": source.get("category"),
    }
    eval_result = evaluate_answer_row(eval_row, eval_mode=eval_mode, judge_client=judge_client, num_judge_runs=num_judge_runs)
    memory_tokens = estimate_tokens(memory_context)
    ev_recall = evidence_recall(source.get("evidence", []), tool_results)
    reward = reward_breakdown(
        valid=True,
        answer_score=answer_reward(eval_result, eval_mode),
        evidence_score=ev_recall,
        memory_tokens=memory_tokens,
        tool_count=len(parsed.get("tool_calls", [])),
        memory_token_budget=token_budget,
        weights=reward_weights,
    )
    result = {
        "question_id": source.get("question_id"),
        "conversation_index": source.get("conversation_index"),
        "qa_index": source.get("qa_index"),
        "question": source["question"],
        "planner_raw_output": raw_output,
        "planner_valid": True,
        "planner_output": parsed,
        "planner_errors": [],
        "tool_results": tool_results,
        "memory_context": memory_context,
        "memories": context_obj.get("memories", []),
        "generated_answer": generated,
        "golden_answer": source.get("answer"),
        "gold_evidence_ids": source.get("evidence", []),
        "retrieved_evidence_ids": sorted(retrieved_evidence(tool_results)),
        "memory_tokens": memory_tokens,
        "tool_count": len(parsed.get("tool_calls", [])),
        "eval": eval_result,
        "reward": reward,
        "duration_ms": int((time.time() - started) * 1000),
    }
    if answer_error:
        result["answer_error"] = answer_error
    return result


def planner_output_for_row(row: dict[str, Any], candidate_field: str, use_gold_plan: bool) -> str | dict[str, Any]:
    if use_gold_plan:
        return row.get("planner_target", {"tool_calls": []})
    if candidate_field in row:
        return row[candidate_field]
    raise KeyError(f"row does not contain candidate planner output field: {candidate_field}")


def summarize_reward_rows(rows: list[dict[str, Any]], eval_mode: str) -> dict[str, Any]:
    total = len(rows)
    valid = sum(1 for row in rows if row.get("planner_valid"))
    reward_values = [float(row.get("reward", {}).get("total", 0.0)) for row in rows]
    llm_rows = [row for row in rows if "llm_correct" in row.get("eval", {})]
    return {
        "samples": total,
        "eval_mode": eval_mode,
        "planner_valid": valid,
        "planner_valid_rate": round(valid / total, 4) if total else 0.0,
        "avg_reward": round(sum(reward_values) / total, 4) if total else 0.0,
        "avg_tool_count": round(sum(row.get("tool_count", 0) for row in rows) / total, 4) if total else 0.0,
        "avg_memory_tokens": round(sum(row.get("memory_tokens", 0) for row in rows) / total, 4) if total else 0.0,
        "llm_correct": sum(1 for row in llm_rows if row.get("eval", {}).get("llm_correct")),
        "llm_accuracy": round(sum(1 for row in llm_rows if row.get("eval", {}).get("llm_correct")) / len(llm_rows), 4) if llm_rows else None,
    }


def run_reward_runner(
    planner_data_path: str | Path,
    stores_path: str | Path,
    output_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    candidate_field: str = "planner_output",
    use_gold_plan: bool = False,
    limit: int | None = None,
    eval_mode: str = "lexical",
    num_judge_runs: int = 1,
    token_budget: int = 0,
    max_memory_chars: int | None = None,
    prompt_style: str = "concise",
    skip_generation: bool = False,
) -> dict[str, Any]:
    rows = read_jsonl(planner_data_path)
    if limit is not None:
        rows = rows[:limit]
    executor = MemoryToolExecutor.from_jsonl(stores_path)
    judge_client = judge_client_for_mode(eval_mode)
    output_rows = []
    for row in rows:
        planner_output = planner_output_for_row(row, candidate_field, use_gold_plan)
        output_rows.append(
            evaluate_planner_output(
                row,
                planner_output,
                executor,
                eval_mode=eval_mode,
                judge_client=judge_client,
                num_judge_runs=num_judge_runs,
                token_budget=token_budget,
                max_memory_chars=max_memory_chars,
                prompt_style=prompt_style,
                skip_generation=skip_generation,
            )
        )
    write_jsonl(output_path, output_rows)
    summary = summarize_reward_rows(output_rows, eval_mode)
    summary["output"] = str(output_path)
    if summary_path:
        write_json(summary_path, summary)
    return summary


def parse_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str) and ground_truth.strip():
        try:
            parsed = json.loads(ground_truth)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {"gold_answer": ground_truth}
    return {}


def compute_verl_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    *,
    eval_mode: str | None = None,
) -> float:
    extra = extra_info or {}
    truth = parse_ground_truth(ground_truth)
    stores_path = extra.get("stores_path") or os.environ.get("MTP_MEMORY_STORES_PATH")
    if not stores_path:
        return -1.0
    row = {
        "question_id": truth.get("question_id") or extra.get("question_id"),
        "question": truth.get("question", ""),
        "answer": truth.get("gold_answer"),
        "evidence": truth.get("gold_evidence_ids", extra.get("evidence", [])),
        "category": truth.get("category", extra.get("category")),
        "conversation_index": truth.get("conversation_index", extra.get("conversation_index")),
    }
    mode = eval_mode or os.environ.get("MTP_REWARD_EVAL_MODE", "lexical")
    judge_client = judge_client_for_mode(mode)
    executor = MemoryToolExecutor.from_jsonl(stores_path)
    trace = evaluate_planner_output(
        row,
        solution_str,
        executor,
        eval_mode=mode,
        judge_client=judge_client,
        num_judge_runs=int(os.environ.get("MTP_NUM_JUDGE_RUNS", "1")),
        token_budget=int(os.environ.get("MTP_MEMORY_TOKEN_BUDGET", "0")),
        max_memory_chars=int(os.environ["MTP_MAX_MEMORY_CHARS"]) if os.environ.get("MTP_MAX_MEMORY_CHARS") else None,
        prompt_style=os.environ.get("MTP_ANSWER_PROMPT_STYLE", "concise"),
        skip_generation=os.environ.get("MTP_SKIP_ANSWER_GENERATION", "").lower() in {"1", "true", "yes"},
        ground_truth=truth,
    )
    trace_path = os.environ.get("MTP_REWARD_TRACE_PATH")
    if trace_path:
        append_reward_trace(trace, Path(trace_path), unique_key=os.environ.get("MTP_REWARD_TRACE_UNIQUE_KEY"))
    return float(trace.get("reward", {}).get("total", -1.0))


def append_reward_trace(trace: dict[str, Any], path: Path, *, unique_key: str | None = None) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_value = trace.get(unique_key) if unique_key else None
    with path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if unique_key and key_value is not None:
                handle.seek(0)
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if existing.get(unique_key) == key_value:
                        return False
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            return True
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
