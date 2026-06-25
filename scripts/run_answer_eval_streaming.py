#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from memory_tool_planner.answer import answer_question
from memory_tool_planner.answer_eval import (
    evaluate_answer_row,
    memory_context_from_row,
    read_jsonl,
    summarize_results,
    write_json,
)
from memory_tool_planner.config import OpenAISettings
from memory_tool_planner.openai_protocol import OpenAIProtocolClient


def _row_key(row: dict[str, Any], index: int) -> str:
    return str(row.get("question_id") or f"locomo_{row.get('conversation_index', 0)}_qa{row.get('qa_index', index)}")


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl_row(handle: Any, row: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False)
    with lock:
        handle.write(line + "\n")
        handle.flush()


def _evaluate_one(
    index: int,
    row: dict[str, Any],
    *,
    prompt_style: str,
    eval_mode: str,
    num_judge_runs: int,
    token_budget: int,
    max_memory_chars: int | None,
    rebuild_context: bool,
    include_context: bool,
    skip_generation: bool,
) -> dict[str, Any]:
    started = time.time()
    key = _row_key(row, index)
    context_obj = memory_context_from_row(
        row,
        token_budget=token_budget,
        max_memory_chars=max_memory_chars,
        force_rebuild=rebuild_context,
    )
    memory_context = context_obj.get("memory_context", "")
    generated = row.get("generated_answer", "")
    errors: list[str] = []
    if not skip_generation:
        try:
            generated = answer_question(row["question"], memory_context, prompt_style=prompt_style)
        except Exception as exc:
            errors.append(f"answer: {exc}")
            generated = ""

    result_row: dict[str, Any] = {
        "question_id": key,
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
    if include_context:
        result_row["memory_context"] = memory_context
        result_row["memories"] = context_obj.get("memories", [])

    judge_client = None
    if eval_mode in {"llm", "hybrid"}:
        judge_client = OpenAIProtocolClient(OpenAISettings.from_env("JUDGE"))
    try:
        result_row["eval"] = evaluate_answer_row(
            result_row,
            eval_mode=eval_mode,
            judge_client=judge_client,
            num_judge_runs=num_judge_runs,
        )
    except Exception as exc:
        errors.append(f"judge: {exc}")
        result_row["eval"] = {
            "exact_match": False,
            "contains_match": False,
            "token_f1": 0.0,
        }
    if errors:
        result_row["error"] = "; ".join(errors)
    return result_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming, resumable answer-generation and QA judge eval.")
    parser.add_argument("--planner-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-style", choices=["concise", "cot"], default="concise")
    parser.add_argument("--eval-mode", choices=["lexical", "llm", "hybrid"], default="llm")
    parser.add_argument("--num-judge-runs", type=int, default=1)
    parser.add_argument("--token-budget", type=int, default=8000)
    parser.add_argument("--max-memory-chars", type=int, default=2500)
    parser.add_argument("--rebuild-context", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "0"
    rows = read_jsonl(args.planner_data)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(output_path) if args.resume else []
    done_keys = {str(row.get("question_id")) for row in existing if row.get("question_id")}
    pending = [(index, row) for index, row in enumerate(rows) if _row_key(row, index) not in done_keys]
    print(
        json.dumps(
            {
                "planner_data": args.planner_data,
                "output": str(output_path),
                "summary": str(summary_path),
                "total": len(rows),
                "existing": len(existing),
                "pending": len(pending),
                "workers": args.workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    lock = threading.Lock()
    mode = "a" if args.resume and existing else "w"
    completed = len(existing)
    with output_path.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    _evaluate_one,
                    index,
                    row,
                    prompt_style=args.prompt_style,
                    eval_mode=args.eval_mode,
                    num_judge_runs=args.num_judge_runs,
                    token_budget=args.token_budget,
                    max_memory_chars=args.max_memory_chars,
                    rebuild_context=args.rebuild_context,
                    include_context=not args.no_context,
                    skip_generation=args.skip_generation,
                )
                for index, row in pending
            ]
            for future in as_completed(futures):
                result = future.result()
                _write_jsonl_row(handle, result, lock)
                completed += 1
                if completed % max(1, args.progress_every) == 0 or completed == len(rows):
                    print(json.dumps({"completed": completed, "total": len(rows)}, ensure_ascii=False), flush=True)

    all_rows = _load_existing(output_path)
    summary = summarize_results(all_rows, args.eval_mode)
    summary["output"] = str(output_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
