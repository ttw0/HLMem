#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_tool_planner.locomo_planner_data import write_jsonl
from memory_tool_planner.reward import evaluate_planner_output, parse_ground_truth
from memory_tool_planner.retrieval import MemoryToolExecutor
from memory_tool_planner.trace_analysis import planner_metric_summary


def _records(path: Path) -> list[dict[str, Any]]:
    return pd.read_parquet(path).to_dict("records")


def _prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("prompt")
    if isinstance(messages, list):
        return [{"role": item.get("role", "user"), "content": item.get("content", "")} for item in messages if isinstance(item, dict)]
    return [{"role": "user", "content": str(messages or "")}]


def _source_row(row: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    return {
        "question_id": ground_truth.get("question_id", extra.get("question_id")),
        "conversation_index": ground_truth.get("conversation_index", extra.get("conversation_index")),
        "qa_index": ground_truth.get("qa_index", extra.get("qa_index")),
        "question": ground_truth.get("question", ""),
        "answer": ground_truth.get("gold_answer"),
        "evidence": ground_truth.get("gold_evidence_ids", extra.get("evidence", [])),
        "category": ground_truth.get("category", extra.get("category")),
    }


def _tokenize_batch(tokenizer: Any, prompts: list[list[dict[str, str]]]) -> Any:
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in prompts
    ]
    return tokenizer(texts, return_tensors="pt", padding=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _records(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    executor = MemoryToolExecutor.from_jsonl(args.stores)
    output_rows: list[dict[str, Any]] = []

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        encoded = _tokenize_batch(tokenizer, [_prompt_messages(row) for row in batch])
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completions = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        for row, completion in zip(batch, completions, strict=True):
            reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
            ground_truth = parse_ground_truth(reward_model.get("ground_truth"))
            output_rows.append(
                evaluate_planner_output(
                    _source_row(row, ground_truth),
                    completion.strip(),
                    executor,
                    eval_mode="lexical",
                    skip_generation=True,
                    ground_truth=ground_truth,
                )
            )
        print(json.dumps({"done": len(output_rows), "total": len(rows)}, ensure_ascii=False), flush=True)

    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(trace_path, output_rows)

    summary = {
        args.name: {
            **planner_metric_summary(output_rows),
            "trace": str(trace_path),
            "raw_rows": len(output_rows),
            "unique_question_id": len({row.get("question_id") for row in output_rows}),
            "model": args.model,
            "backend": "hf_transformers_bf16",
        }
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run base planner eval with one HF Transformers model copy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--stores", default=os.environ.get("MTP_MEMORY_STORES_PATH", "memory_store/locomo_evermemos_memory_stores.embedded.jsonl"))
    parser.add_argument("--trace", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
