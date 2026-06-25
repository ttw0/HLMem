#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from memory_tool_planner.config import load_env
from memory_tool_planner.context import build_memory_context
from memory_tool_planner.locomo_planner_data import prepare_planner_data, read_jsonl, retrieved_evidence, write_jsonl
from memory_tool_planner.openai_protocol import OpenAIProtocolClient
from memory_tool_planner.retrieval import MemoryToolExecutor
from memory_tool_planner.teacher_sft import build_teacher_prompt, evidence_score, score_teacher_candidate
from memory_tool_planner.training_data import DEFAULT_CURRENT_DATE, DEFAULT_TIMEZONE, export_sft_data


VARIANTS = {
    "teacher_direct": 1,
    "teacher_direct_scoped": 1,
    "teacher_best_of_4": 4,
    "teacher_best_of_8_scoped": 8,
}


def _load_completed(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    completed: dict[int, dict[str, Any]] = {}
    for fallback_index, row in enumerate(rows):
        index = row.get("_source_index", fallback_index)
        if isinstance(index, int):
            completed[index] = row
    return completed


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _label_one(
    index: int,
    row: dict[str, Any],
    *,
    client: OpenAIProtocolClient,
    executor: MemoryToolExecutor,
    candidates_per_sample: int,
    temperature: float,
    current_date: str,
    timezone: str,
    token_budget: int,
    max_memory_chars: int | None,
    variant: str,
) -> dict[str, Any]:
    prompt = build_teacher_prompt(row, current_date=current_date, timezone=timezone)
    candidates = []
    for _ in range(candidates_per_sample):
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
        raise RuntimeError(f"no valid teacher candidate for index={index} question={row.get('question', '')!r}")

    selected = valid_candidates[0] if candidates_per_sample == 1 else max(valid_candidates, key=lambda item: float(item.get("score", -1.0)))
    output_row = dict(row)
    output_row["_source_index"] = index
    output_row["planner_target"] = selected.get("parsed", {"tool_calls": []})
    output_row["tool_results"] = selected.get("tool_results", {"tool_results": []})
    output_row["memory_context"] = selected.get("memory_context", build_memory_context(output_row["tool_results"], token_budget=token_budget, max_memory_chars=max_memory_chars))
    output_row["retrieval_eval"] = selected.get(
        "retrieval_eval",
        {
            "gold_evidence": [str(item) for item in row.get("evidence", []) if item is not None],
            "retrieved_evidence": sorted(retrieved_evidence(output_row["tool_results"])),
            "evidence_score": evidence_score(row.get("evidence", []), output_row["tool_results"]),
        },
    )
    output_row["label_source"] = "teacher_distilled"
    output_row["teacher_model"] = client.settings.chat_model
    output_row["teacher_label_strategy"] = variant
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
    return output_row


def _write_summary(path: Path, rows: list[dict[str, Any]], *, variant: str, candidates_per_sample: int, teacher_model: str) -> None:
    scores = [float(row.get("retrieval_eval", {}).get("evidence_score", 0.0)) for row in rows]
    selected_nonzero = sum(1 for score in scores if score > 0)
    summary = {
        "variant": variant,
        "teacher_model": teacher_model,
        "candidates_per_sample": candidates_per_sample,
        "samples": len(rows),
        "avg_evidence_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "hit_rate": round(selected_nonzero / len(scores), 4) if scores else 0.0,
        "label_source": "teacher_distilled",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def label_split(
    *,
    base_path: Path,
    labeled_path: Path,
    summary_path: Path,
    stores_path: Path,
    variant: str,
    candidates_per_sample: int,
    workers: int,
    limit: int | None,
    temperature: float,
    current_date: str,
    timezone: str,
    token_budget: int,
    max_memory_chars: int | None,
    rebuild_labels: bool,
) -> list[dict[str, Any]]:
    rows = read_jsonl(base_path)
    if limit is not None:
        rows = rows[:limit]
    if rebuild_labels and labeled_path.exists():
        labeled_path.unlink()
    completed = {} if rebuild_labels else _load_completed(labeled_path)
    client = OpenAIProtocolClient.from_env("TEACHER")
    executor = MemoryToolExecutor.from_jsonl(stores_path)
    lock = threading.Lock()
    pending = [(index, row) for index, row in enumerate(rows) if index not in completed]
    print(f"{variant} {base_path.stem}: {len(completed)}/{len(rows)} already complete, {len(pending)} pending")

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _label_one,
                index,
                row,
                client=client,
                executor=executor,
                candidates_per_sample=candidates_per_sample,
                temperature=temperature,
                current_date=current_date,
                timezone=timezone,
                token_budget=token_budget,
                max_memory_chars=max_memory_chars,
                variant=variant,
            ): index
            for index, row in pending
        }
        for done_count, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                labeled = future.result()
            except Exception as exc:
                failures.append(f"index={index}: {exc}")
                print(f"FAILED {variant} index={index}: {exc}")
                continue
            completed[index] = labeled
            _append_jsonl(labeled_path, labeled, lock)
            if done_count % 10 == 0 or done_count == len(pending):
                print(f"{variant}: completed {len(completed)}/{len(rows)}")

    if failures:
        raise RuntimeError(f"{variant} failed for {len(failures)} samples; first failure: {failures[0]}")
    ordered = [completed[index] for index in range(len(rows))]
    write_jsonl(labeled_path, ordered)
    _write_summary(summary_path, ordered, variant=variant, candidates_per_sample=candidates_per_sample, teacher_model=client.settings.chat_model)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SFT ablation data for direct teacher labels vs best-of-N teacher distillation.")
    parser.add_argument("--locomo", default="data/locomo/locomo10_precess.json")
    parser.add_argument("--stores", default="memory_store/locomo_evermemos_memory_stores.embedded.jsonl")
    parser.add_argument("--planner-data-dir", default="data/planner_data")
    parser.add_argument("--sft-dir", default="data/sft")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--splits", nargs="+", choices=["train", "valid"], default=["train", "valid"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--current-date", default=DEFAULT_CURRENT_DATE)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument("--max-memory-chars", type=int)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--rebuild-labels", action="store_true")
    args = parser.parse_args()

    load_env()
    os.environ["MEMORY_TOOL_PLANNER_OFFLINE"] = "0"

    planner_dir = Path(args.planner_data_dir)
    sft_dir = Path(args.sft_dir)
    stores_path = Path(args.stores)
    for split in args.splits:
        base_path = planner_dir / f"locomo_planner_data.{split}.base.jsonl"
        if not args.skip_prepare:
            prepare_planner_data(args.locomo, base_path, split=split)
        for variant in args.variants:
            candidates = VARIANTS[variant]
            labeled_path = planner_dir / f"locomo_planner_data.{split}.{variant}.jsonl"
            summary_path = planner_dir / f"locomo_planner_data.{split}.{variant}.summary.json"
            label_split(
                base_path=base_path,
                labeled_path=labeled_path,
                summary_path=summary_path,
                stores_path=stores_path,
                variant=variant,
                candidates_per_sample=candidates,
                workers=args.workers,
                limit=args.limit,
                temperature=args.temperature,
                current_date=args.current_date,
                timezone=args.timezone,
                token_budget=args.token_budget,
                max_memory_chars=args.max_memory_chars,
                rebuild_labels=args.rebuild_labels,
            )
            suffix = f"{variant}_{split}"
            export_sft_data(labeled_path, sft_dir / f"{suffix}.jsonl", output_format="jsonl", current_date=args.current_date, timezone=args.timezone)
            export_sft_data(labeled_path, sft_dir / f"{suffix}.parquet", output_format="parquet", current_date=args.current_date, timezone=args.timezone)
            print(f"exported {sft_dir / f'{suffix}.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
