#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path
from typing import Any


CASE_NAMES = (
    "temp_high_tool_0p02",
    "temp_low_tool_0p02",
    "tool_0p05_under_0p05",
    "tool_0p02_under_0p02",
    "lr_2p5e6_temp_0p7",
    "kl_high_temp_high",
    "group6_temp_0p9",
    "group3_temp_0p7",
    "kl_low_temp_low",
    "lr_high_kl_high",
    "kl_low_tool_0p02",
    "lr_low_kl_low",
    "under_tool_min2",
    "under_tool_0p15",
    "under_tool_0p10",
    "under_tool_0p08",
    "under_tool_0p02",
    "under_tool_0p01",
    "tool_cost_0p10",
    "tool_cost_0p08",
    "tool_cost_0p05",
    "tool_cost_0p02",
    "tool_cost_0p01",
    "tool_cost_0p005",
    "temp_high",
    "temp_low",
    "temp_1p2",
    "temp_1p1",
    "temp_0p9",
    "temp_0p7",
    "temp_0p5",
    "kl_high",
    "kl_low",
    "kl_8e3",
    "kl_5e3",
    "kl_2e3",
    "kl_1p5e3",
    "kl_7e4",
    "kl_5e4",
    "kl_1e4",
    "lr_high",
    "lr_low",
    "lr_6e6",
    "lr_4e6",
    "lr_2p5e6",
    "lr_1p5e6",
    "lr_7p5e7",
    "lr_5e7",
    "group8",
    "group6",
    "group5",
    "group3",
    "group2",
    "tool_cost",
    "under_tool",
    "current",
)

ERROR_COUNT_KEYS = (
    "DUPLICATE_CALL",
    "INVALID_ARGUMENTS",
    "TOO_MANY_CALLS",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _env(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return {}
    data = _load_json(config_path)
    env = data.get("env", {})
    return env if isinstance(env, dict) else {}


def _metric(summary: dict[str, Any], suffix: str) -> dict[str, Any]:
    for key, value in summary.items():
        if key.endswith(suffix) and isinstance(value, dict):
            return value
    return {}


def _error_count(metrics: dict[str, Any], name: str) -> int | None:
    error_counts = metrics.get("error_counts")
    if not isinstance(error_counts, dict):
        return None
    value = error_counts.get(name)
    return int(value) if isinstance(value, (int, float)) else None


def _case_and_seed(run_name: str) -> tuple[str, int | None]:
    seed = None
    match = re.search(r"_seed(\d+)$", run_name)
    stem = run_name
    if match:
        seed = int(match.group(1))
        stem = run_name[: match.start()]
    for case_name in CASE_NAMES:
        if stem.endswith(f"_{case_name}"):
            return case_name, seed
    return stem.rsplit("_", 1)[-1], seed


def _model_label(run_name: str) -> str:
    if "qwen3_8b" in run_name:
        return "Qwen3-8B"
    if "qwen3_4b" in run_name:
        return "Qwen3-4B"
    if "qwen3_1_7b" in run_name:
        return "Qwen3-1.7B"
    if "llama3_2_3b" in run_name:
        return "Llama-3.2-3B"
    return ""


def _stage_label(run_name: str) -> str:
    if "_hparam_phase2_" in run_name:
        return "phase2"
    if "_hparam_validate_" in run_name:
        return "validate"
    if "_hparam_" in run_name:
        return "hparam"
    if "_ppo_" in run_name:
        return "ppo"
    if "_grpo_" in run_name:
        return "grpo"
    return ""


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return round(statistics.stdev(values), 6)


def _numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("model"),
            row.get("stage"),
            row.get("case"),
            row.get("actor_lr"),
            row.get("critic_lr"),
            row.get("kl_coef"),
            row.get("temperature"),
            row.get("group_size"),
            row.get("rollout_max_num_seqs"),
            row.get("token_cost"),
            row.get("tool_cost"),
            row.get("under_tool"),
            row.get("min_tool"),
        )
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, group_rows in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0]))):
        seeds = sorted(row["seed"] for row in group_rows if row.get("seed") is not None)
        (
            model,
            stage,
            case,
            actor_lr,
            critic_lr,
            kl_coef,
            temperature,
            group_size,
            rollout_max_num_seqs,
            token_cost,
            tool_cost,
            under_tool,
            min_tool,
        ) = key
        aggregate_rows.append(
            {
                "model": model,
                "stage": stage,
                "case": case,
                "seeds": seeds,
                "n": len(group_rows),
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "kl_coef": kl_coef,
                "temperature": temperature,
                "group_size": group_size,
                "rollout_max_num_seqs": rollout_max_num_seqs,
                "token_cost": token_cost,
                "tool_cost": tool_cost,
                "under_tool": under_tool,
                "min_tool": min_tool,
                "valid_evidence_mean": _mean(_numeric(group_rows, "valid_evidence")),
                "valid_evidence_std": _std(_numeric(group_rows, "valid_evidence")),
                "test_evidence_mean": _mean(_numeric(group_rows, "test_evidence")),
                "test_evidence_std": _std(_numeric(group_rows, "test_evidence")),
                "test_hit_mean": _mean(_numeric(group_rows, "test_hit")),
                "test_hit_std": _std(_numeric(group_rows, "test_hit")),
                "test_full_hit_mean": _mean(_numeric(group_rows, "test_full_hit")),
                "test_full_hit_std": _std(_numeric(group_rows, "test_full_hit")),
                "test_valid_rate_mean": _mean(_numeric(group_rows, "test_valid_rate")),
                "test_valid_rate_std": _std(_numeric(group_rows, "test_valid_rate")),
                "test_invalid_json_mean": _mean(_numeric(group_rows, "test_invalid_json")),
                "test_invalid_json_std": _std(_numeric(group_rows, "test_invalid_json")),
                "test_empty_calls_mean": _mean(_numeric(group_rows, "test_empty_calls")),
                "test_empty_calls_std": _std(_numeric(group_rows, "test_empty_calls")),
                "test_duplicate_query_rate_mean": _mean(_numeric(group_rows, "test_duplicate_query_rate")),
                "test_duplicate_query_rate_std": _std(_numeric(group_rows, "test_duplicate_query_rate")),
                "test_normalized_duplicate_query_rate_mean": _mean(
                    _numeric(group_rows, "test_normalized_duplicate_query_rate")
                ),
                "test_normalized_duplicate_query_rate_std": _std(
                    _numeric(group_rows, "test_normalized_duplicate_query_rate")
                ),
                "test_duplicate_calls_mean": _mean(_numeric(group_rows, "test_duplicate_calls")),
                "test_duplicate_calls_std": _std(_numeric(group_rows, "test_duplicate_calls")),
                "test_invalid_arguments_mean": _mean(_numeric(group_rows, "test_invalid_arguments")),
                "test_invalid_arguments_std": _std(_numeric(group_rows, "test_invalid_arguments")),
                "test_too_many_calls_mean": _mean(_numeric(group_rows, "test_too_many_calls")),
                "test_too_many_calls_std": _std(_numeric(group_rows, "test_too_many_calls")),
            }
        )
    return aggregate_rows


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _mean_std(row: dict[str, Any], mean_key: str, std_key: str) -> str:
    mean = row.get(mean_key)
    std = row.get(std_key)
    if mean is None:
        return ""
    return f"{_cell(mean)}±{_cell(std)}"


def _markdown(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model | stage | run | case | seed | lr | kl | temp | group | max seqs | tool_cost | under_tool | min_tool | valid evidence | test evidence | test hit | full-hit | valid_rate | invalid_json | empty | dup query | dup calls | invalid args | too many |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {stage} | {run_name} | {case} | {seed} | {actor_lr} | {kl_coef} | {temperature} | {group_size} | {rollout_max_num_seqs} | {tool_cost} | {under_tool} | {min_tool} | {valid_evidence} | {test_evidence} | {test_hit} | {test_full_hit} | {test_valid_rate} | {test_invalid_json} | {test_empty_calls} | {test_duplicate_query_rate} | {test_duplicate_calls} | {test_invalid_arguments} | {test_too_many_calls} |".format(
                **{key: _cell(value) for key, value in row.items()}
            )
        )

    if aggregate_rows:
        lines.extend(
            [
                "",
                "| model | stage | case | seeds | n | lr | kl | temp | group | max seqs | tool_cost | under_tool | min_tool | valid evidence mean±std | test evidence mean±std | hit mean±std | full-hit mean±std | valid-rate mean±std | invalid-json mean±std | empty mean±std | dup-query mean±std | dup-call mean±std | invalid-args mean±std | too-many mean±std |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in aggregate_rows:
            values = {key: _cell(value) for key, value in row.items()}
            values["seeds"] = ",".join(str(seed) for seed in row["seeds"])
            values["valid_evidence"] = _mean_std(row, "valid_evidence_mean", "valid_evidence_std")
            values["test_evidence"] = _mean_std(row, "test_evidence_mean", "test_evidence_std")
            values["test_hit"] = _mean_std(row, "test_hit_mean", "test_hit_std")
            values["test_full_hit"] = _mean_std(row, "test_full_hit_mean", "test_full_hit_std")
            values["test_valid_rate"] = _mean_std(row, "test_valid_rate_mean", "test_valid_rate_std")
            values["test_invalid_json"] = _mean_std(row, "test_invalid_json_mean", "test_invalid_json_std")
            values["test_empty_calls"] = _mean_std(row, "test_empty_calls_mean", "test_empty_calls_std")
            values["test_duplicate_query_rate"] = _mean_std(
                row, "test_duplicate_query_rate_mean", "test_duplicate_query_rate_std"
            )
            values["test_duplicate_calls"] = _mean_std(row, "test_duplicate_calls_mean", "test_duplicate_calls_std")
            values["test_invalid_arguments"] = _mean_std(
                row, "test_invalid_arguments_mean", "test_invalid_arguments_std"
            )
            values["test_too_many_calls"] = _mean_std(row, "test_too_many_calls_mean", "test_too_many_calls_std")
            lines.append(
                "| {model} | {stage} | {case} | {seeds} | {n} | {actor_lr} | {kl_coef} | {temperature} | {group_size} | {rollout_max_num_seqs} | {tool_cost} | {under_tool} | {min_tool} | {valid_evidence} | {test_evidence} | {test_hit} | {test_full_hit} | {test_valid_rate} | {test_invalid_json} | {test_empty_calls} | {test_duplicate_query_rate} | {test_duplicate_calls} | {test_invalid_arguments} | {test_too_many_calls} |".format(
                    **values
                )
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize GRPO/PPO hyperparameter sweep eval summaries.")
    parser.add_argument("--summary-glob", required=True, help="Glob for *_eval_summary.json files.")
    parser.add_argument("--run-config-dir", default="outputs/v0/run_configs")
    parser.add_argument("--model-label", help="Optional model label to force in output tables.")
    parser.add_argument("--stage-label", help="Optional stage label to force in output tables.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregate-output", help="Optional JSON output grouped by case and hyperparameter tuple.")
    parser.add_argument("--markdown-output", help="Optional paper-ready Markdown table output.")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(match) for match in glob.glob(args.summary_glob)):
        summary = _load_json(path)
        run_name = path.name.removesuffix("_eval_summary.json")
        case, seed = _case_and_seed(run_name)
        config = _env(Path(args.run_config_dir) / f"{run_name}.json")
        valid = _metric(summary, "_valid")
        test = _metric(summary, "_test")
        rows.append(
            {
                "model": args.model_label or _model_label(run_name),
                "stage": args.stage_label or _stage_label(run_name),
                "run_name": run_name,
                "case": case,
                "seed": seed,
                "actor_lr": config.get("GRPO_ACTOR_LR") or config.get("PPO_ACTOR_LR"),
                "critic_lr": config.get("PPO_CRITIC_LR"),
                "kl_coef": config.get("GRPO_KL_COEF") or config.get("PPO_KL_COEF"),
                "temperature": config.get("GRPO_ROLLOUT_TEMPERATURE") or config.get("PPO_ROLLOUT_TEMPERATURE"),
                "group_size": config.get("GRPO_GROUP_SIZE") or config.get("PPO_ROLLOUT_N"),
                "rollout_max_num_seqs": config.get("GRPO_ROLLOUT_MAX_NUM_SEQS") or config.get("PPO_ROLLOUT_MAX_NUM_SEQS"),
                "token_cost": config.get("MTP_REWARD_TOKEN_COST_WEIGHT", "0.0"),
                "tool_cost": config.get("MTP_REWARD_TOOL_COST_WEIGHT", "0.0"),
                "under_tool": config.get("MTP_REWARD_UNDER_TOOL_WEIGHT", "0.0"),
                "min_tool": config.get("MTP_REWARD_MIN_TOOL_COUNT", "0"),
                "valid_reward": valid.get("reward_mean"),
                "valid_evidence": valid.get("evidence_mean"),
                "valid_full_hit": valid.get("full_hit_rate"),
                "valid_rate": valid.get("valid_rate"),
                "valid_duplicate_query_rate": valid.get("duplicate_query_rate"),
                "valid_normalized_duplicate_query_rate": valid.get("normalized_duplicate_query_rate"),
                "test_reward": test.get("reward_mean"),
                "test_evidence": test.get("evidence_mean"),
                "test_hit": test.get("hit_rate"),
                "test_full_hit": test.get("full_hit_rate"),
                "test_valid_rate": test.get("valid_rate"),
                "test_invalid_json": test.get("invalid_json"),
                "test_empty_calls": test.get("empty_calls"),
                "test_duplicate_query_rate": test.get("duplicate_query_rate"),
                "test_normalized_duplicate_query_rate": test.get("normalized_duplicate_query_rate"),
                "test_duplicate_calls": _error_count(test, "DUPLICATE_CALL"),
                "test_invalid_arguments": _error_count(test, "INVALID_ARGUMENTS"),
                "test_too_many_calls": _error_count(test, "TOO_MANY_CALLS"),
                "summary": str(path),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    aggregate_rows = _aggregate(rows)
    if args.aggregate_output:
        aggregate_output = Path(args.aggregate_output)
        aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        aggregate_output.write_text(json.dumps(aggregate_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown = _markdown(rows, aggregate_rows)
    if args.markdown_output:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(markdown, encoding="utf-8")

    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
