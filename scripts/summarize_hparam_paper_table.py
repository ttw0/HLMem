#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {path}")
    return [row for row in data if isinstance(row, dict)]


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _format_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.6f}"


def _seeds_text(row: dict[str, Any]) -> str:
    seeds = row.get("seeds")
    if isinstance(seeds, list):
        return ",".join(str(seed) for seed in seeds)
    seed = row.get("seed")
    return "" if seed is None else str(seed)


def _norm(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _norm(row.get("model")),
        _norm(row.get("stage")),
        _norm(row.get("case")),
        _norm(row.get("actor_lr")),
        _norm(row.get("critic_lr")),
        _norm(row.get("kl_coef")),
        _norm(row.get("temperature")),
        _norm(row.get("group_size")),
        _norm(row.get("rollout_max_num_seqs")),
        _norm(row.get("token_cost"), "0.0"),
        _norm(row.get("tool_cost"), "0.0"),
        _norm(row.get("under_tool"), "0.0"),
        _norm(row.get("min_tool"), "0"),
        tuple(row.get("seeds") or ([row.get("seed")] if row.get("seed") is not None else [])),
    )


def _training_index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {_key(row): row for row in rows}


def _merge_rows(eval_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]], baseline: float | None) -> list[dict[str, Any]]:
    training_by_key = _training_index(training_rows)
    out = []
    for row in eval_rows:
        merged = dict(row)
        train = training_by_key.get(_key(row), {})
        for key in (
            "initial_val_acc_mean",
            "final_train_step_mean",
            "final_train_score_mean",
            "final_train_score_mean_mean",
            "final_train_reward_mean",
            "final_train_reward_mean_mean",
            "final_actor_entropy_mean",
            "final_actor_kl_loss_mean",
            "final_response_length_mean",
            "final_response_length_mean_mean",
            "final_throughput_mean",
            "initial_val_acc",
            "final_train_step",
            "final_train_score_mean",
            "final_train_reward_mean",
            "final_actor_entropy",
            "final_actor_kl_loss",
            "final_response_length_mean",
            "final_throughput",
        ):
            if key in train:
                merged[key] = train.get(key)
        for source, target in (
            ("final_train_score_mean_mean", "final_train_score_mean"),
            ("final_train_reward_mean_mean", "final_train_reward_mean"),
            ("final_response_length_mean_mean", "final_response_length_mean"),
        ):
            if source in train:
                merged[target] = train.get(source)
        evidence = _as_float(merged.get("test_evidence_mean") or merged.get("test_evidence"))
        merged["delta_test_evidence"] = round(evidence - baseline, 6) if evidence is not None and baseline is not None else None
        out.append(merged)
    return sorted(
        out,
        key=lambda row: (
            _as_float(row.get("test_evidence_mean") or row.get("test_evidence")) is not None,
            _as_float(row.get("test_evidence_mean") or row.get("test_evidence")) or -1.0,
            _as_float(row.get("test_full_hit_mean") or row.get("test_full_hit")) or -1.0,
        ),
        reverse=True,
    )


def _row_metric(row: dict[str, Any], aggregate_key: str, raw_key: str | None = None) -> Any:
    if aggregate_key in row:
        return row.get(aggregate_key)
    if raw_key is not None:
        return row.get(raw_key)
    return None


def _markdown(rows: list[dict[str, Any]], top_k: int | None) -> str:
    if top_k is not None:
        rows = rows[:top_k]
    lines = [
        "| model | stage | case | seeds | lr | kl | temp | group | max seqs | tool cost | under-tool | min tool | valid evidence | test evidence | delta test | hit | full-hit | valid rate | empty | invalid-json | dup-query | invalid-args | train reward | entropy | KL loss | response len |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {stage} | {case} | {seeds} | {lr} | {kl} | {temp} | {group} | {max_seqs} | {tool_cost} | {under_tool} | {min_tool} | {valid_evidence} | {test_evidence} | {delta_test} | {hit} | {full_hit} | {valid_rate} | {empty} | {invalid_json} | {dup_query} | {invalid_args} | {train_reward} | {entropy} | {kl_loss} | {response_len} |".format(
                model=_cell(row.get("model")),
                stage=_cell(row.get("stage")),
                case=_cell(row.get("case")),
                seeds=_seeds_text(row),
                lr=_cell(row.get("actor_lr")),
                kl=_cell(row.get("kl_coef")),
                temp=_cell(row.get("temperature")),
                group=_cell(row.get("group_size")),
                max_seqs=_cell(row.get("rollout_max_num_seqs")),
                tool_cost=_cell(row.get("tool_cost")),
                under_tool=_cell(row.get("under_tool")),
                min_tool=_cell(row.get("min_tool")),
                valid_evidence=_cell(_row_metric(row, "valid_evidence_mean", "valid_evidence")),
                test_evidence=_cell(_row_metric(row, "test_evidence_mean", "test_evidence")),
                delta_test=_format_delta(_as_float(row.get("delta_test_evidence"))),
                hit=_cell(_row_metric(row, "test_hit_mean", "test_hit")),
                full_hit=_cell(_row_metric(row, "test_full_hit_mean", "test_full_hit")),
                valid_rate=_cell(_row_metric(row, "test_valid_rate_mean", "test_valid_rate")),
                empty=_cell(_row_metric(row, "test_empty_calls_mean", "test_empty_calls")),
                invalid_json=_cell(_row_metric(row, "test_invalid_json_mean", "test_invalid_json")),
                dup_query=_cell(_row_metric(row, "test_duplicate_query_rate_mean", "test_duplicate_query_rate")),
                invalid_args=_cell(_row_metric(row, "test_invalid_arguments_mean", "test_invalid_arguments")),
                train_reward=_cell(_row_metric(row, "final_train_reward_mean", "final_train_reward_mean")),
                entropy=_cell(_row_metric(row, "final_actor_entropy_mean", "final_actor_entropy")),
                kl_loss=_cell(_row_metric(row, "final_actor_kl_loss_mean", "final_actor_kl_loss")),
                response_len=_cell(_row_metric(row, "final_response_length_mean", "final_response_length_mean")),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a compact paper-ready hyperparameter progress table.")
    parser.add_argument("--eval-json", required=True, help="Aggregate or per-run JSON from summarize_hparam_sweep.py.")
    parser.add_argument("--training-json", help="Aggregate or per-run JSON from summarize_training_log_metrics.py.")
    parser.add_argument("--baseline-test-evidence", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--markdown-output", required=True, help="Output Markdown path.")
    args = parser.parse_args()

    eval_rows = _load_rows(Path(args.eval_json))
    training_rows = _load_rows(Path(args.training_json) if args.training_json else None)
    rows = _merge_rows(eval_rows, training_rows, args.baseline_test_evidence)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown = _markdown(rows, args.top_k)
    markdown_output = Path(args.markdown_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
