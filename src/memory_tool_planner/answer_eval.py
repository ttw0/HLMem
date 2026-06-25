from __future__ import annotations

import json
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .answer import answer_question
from .config import OpenAISettings
from .context import build_memory_context
from .openai_protocol import OpenAIProtocolClient


LLM_JUDGE_SYSTEM_PROMPT = "You are an expert grader that determines if answers to questions match a gold standard answer."

LLM_JUDGE_USER_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
(1) a question,
(2) a gold answer,
(3) a generated answer.

The generated answer may be longer than the gold answer. Be generous: if it
touches on the same topic and preserves the key fact, count it as CORRECT.

For time related questions, accept equivalent date formats and relative time
references if they clearly refer to the same date or period.

Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

Return JSON only:
{{"label": "CORRECT" or "WRONG"}}
"""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stringify_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_answer(text: Any) -> str:
    value = stringify_answer(text).lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def token_f1(golden: Any, generated: Any) -> float:
    gold_tokens = normalize_answer(golden).split()
    gen_tokens = normalize_answer(generated).split()
    if not gold_tokens and not gen_tokens:
        return 1.0
    if not gold_tokens or not gen_tokens:
        return 0.0
    common = Counter(gold_tokens) & Counter(gen_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(gen_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(golden: Any, generated: Any) -> bool:
    return normalize_answer(golden) == normalize_answer(generated)


def contains_match(golden: Any, generated: Any) -> bool:
    gold = normalize_answer(golden)
    gen = normalize_answer(generated)
    if not gold or not gen:
        return False
    return gold in gen or gen in gold


def extract_judge_json(content: str) -> dict[str, Any] | None:
    cleaned = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*"label"\s*:\s*"[^"]+"[^{}]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def judge_with_llm(
    question: str,
    golden_answer: Any,
    generated_answer: str,
    client: OpenAIProtocolClient,
    num_runs: int = 1,
) -> dict[str, Any]:
    judgments = []
    raw_outputs = []
    user_prompt = LLM_JUDGE_USER_PROMPT.format(
        question=question,
        golden_answer=stringify_answer(golden_answer),
        generated_answer=generated_answer,
    )
    for _ in range(max(1, num_runs)):
        content = client.chat(
            [
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        raw_outputs.append(content)
        parsed = extract_judge_json(content)
        label = str(parsed.get("label", "") if parsed else "").strip().upper()
        judgments.append(label == "CORRECT")
    return {
        "llm_judgments": {f"judgment_{index + 1}": value for index, value in enumerate(judgments)},
        "llm_raw_outputs": raw_outputs,
        "llm_correct": sum(judgments) / len(judgments) >= 0.5,
    }


def memory_context_from_row(row: dict[str, Any], token_budget: int, max_memory_chars: int | None = None, force_rebuild: bool = False) -> dict[str, Any]:
    tool_results = row.get("tool_results")
    if force_rebuild and isinstance(tool_results, dict):
        return build_memory_context(tool_results, token_budget=token_budget, max_memory_chars=max_memory_chars)
    context = row.get("memory_context")
    if isinstance(context, dict) and "memory_context" in context:
        return context
    if isinstance(context, str):
        return {"memory_context": context, "memories": []}
    if isinstance(tool_results, dict):
        return build_memory_context(tool_results, token_budget=token_budget, max_memory_chars=max_memory_chars)
    return {"memory_context": "", "memories": []}


def evaluate_answer_row(
    row: dict[str, Any],
    *,
    eval_mode: str,
    judge_client: OpenAIProtocolClient | None,
    num_judge_runs: int,
) -> dict[str, Any]:
    generated = row.get("generated_answer", "")
    golden = row.get("golden_answer", row.get("answer"))
    result = {
        "exact_match": exact_match(golden, generated),
        "contains_match": contains_match(golden, generated),
        "token_f1": round(token_f1(golden, generated), 4),
    }
    if eval_mode == "llm" or eval_mode == "hybrid":
        if judge_client is None:
            raise RuntimeError("LLM judge requires online chat settings")
        result.update(judge_with_llm(row.get("question", ""), golden, generated, judge_client, num_runs=num_judge_runs))
    return result


def summarize_results(rows: list[dict[str, Any]], eval_mode: str) -> dict[str, Any]:
    total = len(rows)
    exact = sum(1 for row in rows if row.get("eval", {}).get("exact_match"))
    contains = sum(1 for row in rows if row.get("eval", {}).get("contains_match"))
    f1_values = [row.get("eval", {}).get("token_f1", 0.0) for row in rows]
    llm_rows = [row for row in rows if "llm_correct" in row.get("eval", {})]
    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "exact": 0, "contains": 0, "llm_correct": 0})
    for row in rows:
        category = str(row.get("category", "unknown"))
        stats = by_category[category]
        stats["total"] += 1
        stats["exact"] += int(bool(row.get("eval", {}).get("exact_match")))
        stats["contains"] += int(bool(row.get("eval", {}).get("contains_match")))
        stats["llm_correct"] += int(bool(row.get("eval", {}).get("llm_correct")))
    category_summary = {}
    for category, stats in sorted(by_category.items()):
        total_cat = stats["total"]
        category_summary[category] = {
            **stats,
            "exact_accuracy": round(stats["exact"] / total_cat, 4) if total_cat else 0.0,
            "contains_accuracy": round(stats["contains"] / total_cat, 4) if total_cat else 0.0,
            "llm_accuracy": round(stats["llm_correct"] / total_cat, 4) if total_cat and llm_rows else None,
        }
    return {
        "samples": total,
        "eval_mode": eval_mode,
        "exact_correct": exact,
        "exact_accuracy": round(exact / total, 4) if total else 0.0,
        "contains_correct": contains,
        "contains_accuracy": round(contains / total, 4) if total else 0.0,
        "avg_token_f1": round(sum(f1_values) / total, 4) if total else 0.0,
        "llm_correct": sum(1 for row in llm_rows if row.get("eval", {}).get("llm_correct")),
        "llm_accuracy": round(sum(1 for row in llm_rows if row.get("eval", {}).get("llm_correct")) / len(llm_rows), 4) if llm_rows else None,
        "by_category": category_summary,
    }


def run_answer_eval(
    planner_data_path: str | Path,
    output_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    limit: int | None = None,
    prompt_style: str = "concise",
    eval_mode: str = "lexical",
    num_judge_runs: int = 1,
    token_budget: int = 0,
    max_memory_chars: int | None = None,
    rebuild_context: bool = False,
    include_context: bool = True,
    skip_generation: bool = False,
) -> dict[str, Any]:
    settings = OpenAISettings.from_env("JUDGE")
    judge_client = OpenAIProtocolClient(settings) if (eval_mode in {"llm", "hybrid"} and not settings.offline) else None
    rows = read_jsonl(planner_data_path)
    if limit is not None:
        rows = rows[:limit]
    output_rows = []
    for index, row in enumerate(rows):
        started = time.time()
        context_obj = memory_context_from_row(row, token_budget=token_budget, max_memory_chars=max_memory_chars, force_rebuild=rebuild_context)
        memory_context = context_obj.get("memory_context", "")
        generated = row.get("generated_answer", "")
        error = ""
        if not skip_generation:
            try:
                generated = answer_question(row["question"], memory_context, prompt_style=prompt_style)
            except Exception as exc:  # pragma: no cover - runtime boundary
                error = str(exc)
                generated = ""
        result_row = {
            "question_id": row.get("question_id") or f"locomo_{row.get('conversation_index', 0)}_qa{row.get('qa_index', index)}",
            "sample_id": row.get("sample_id"),
            "conversation_index": row.get("conversation_index"),
            "qa_index": row.get("qa_index"),
            "question": row.get("question", ""),
            "generated_answer": generated,
            "golden_answer": row.get("answer"),
            "category": row.get("category"),
            "evidence": row.get("evidence", []),
            "retrieval_eval": row.get("retrieval_eval", {}),
            "answer_duration_ms": int((time.time() - started) * 1000),
        }
        if error:
            result_row["error"] = error
        if include_context:
            result_row["memory_context"] = memory_context
            result_row["memories"] = context_obj.get("memories", [])
        result_row["eval"] = evaluate_answer_row(result_row, eval_mode=eval_mode, judge_client=judge_client, num_judge_runs=num_judge_runs)
        output_rows.append(result_row)
    write_jsonl(output_path, output_rows)
    summary = summarize_results(output_rows, eval_mode=eval_mode)
    summary["output"] = str(output_path)
    if summary_path:
        write_json(summary_path, summary)
    return summary
