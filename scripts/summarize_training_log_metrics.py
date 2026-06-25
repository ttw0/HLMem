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

SNAPSHOT_METRICS = (
    "critic/score/mean",
    "critic/rewards/mean",
    "actor/entropy",
    "actor/kl_loss",
    "actor/pg_loss",
    "actor/grad_norm",
    "response_length/mean",
    "response/aborted_ratio",
    "num_turns/mean",
    "perf/throughput",
    "timing_s/step",
)

FINAL_METRICS = (
    "initial_val_acc",
    "final_train_step",
    "final_train_score_mean",
    "final_train_reward_mean",
    "final_actor_entropy",
    "final_actor_kl_loss",
    "final_actor_pg_loss",
    "final_response_length_mean",
    "final_aborted_ratio",
    "final_throughput",
    "final_step_seconds",
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")
WRAPPER_RUN_RE = re.compile(r"_(dense|repeat|expanded)_seed\d+$")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _env(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return {}
    data = _load_json(config_path)
    env = data.get("env", {})
    return env if isinstance(env, dict) else {}


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


def _to_number(value: str) -> int | float | None:
    value = value.strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value):
        return None
    number = float(value)
    if number.is_integer() and "." not in value and "e" not in value.lower():
        return int(number)
    return number


def _strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line.replace("\r", "\n"))


def _parse_metrics(payload: str) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for part in payload.split(" - "):
        if ":" not in part:
            continue
        key, raw_value = part.split(":", 1)
        value = _to_number(raw_value)
        if value is not None:
            metrics[key.strip()] = value
    return metrics


def _parse_log(path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        for line in _strip_ansi(raw_line).splitlines():
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            metrics = _parse_metrics(match.group(2))
            if not metrics:
                continue
            has_train_metric = any(
                key.startswith(("actor/", "critic/", "response", "perf/", "timing_s/", "training/"))
                for key in metrics
            )
            phase = "train" if has_train_metric else "val"
            steps.append({"step": step, "phase": phase, "metrics": metrics})
    return steps


def _metric(step: dict[str, Any] | None, key: str) -> int | float | None:
    if not step:
        return None
    value = step.get("metrics", {}).get(key)
    return value if isinstance(value, (int, float)) else None


def _last_step(steps: list[dict[str, Any]], phase: str) -> dict[str, Any] | None:
    filtered = [step for step in steps if step.get("phase") == phase]
    if not filtered:
        return None
    return max(filtered, key=lambda step: step["step"])


def _selected_train_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_steps_by_step = {
        step["step"]: step
        for step in steps
        if step.get("phase") == "train" and any(key in step.get("metrics", {}) for key in SNAPSHOT_METRICS)
    }
    if not train_steps_by_step:
        return []
    ordered_steps = sorted(train_steps_by_step)
    wanted = {ordered_steps[0], ordered_steps[len(ordered_steps) // 2], ordered_steps[-1]}
    return [train_steps_by_step[step] for step in ordered_steps if step in wanted]


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


def _summarize_run(path: Path, run_config_dir: Path, model_label: str | None, stage_label: str | None) -> dict[str, Any] | None:
    run_name = path.name.removesuffix(".log")
    if WRAPPER_RUN_RE.search(run_name):
        return None
    steps = _parse_log(path)
    if not steps:
        return None

    case, seed = _case_and_seed(run_name)
    config = _env(run_config_dir / f"{run_name}.json")
    initial_val = next((step for step in steps if step.get("phase") == "val"), None)
    final_train = _last_step(steps, "train")
    selected_steps = []
    for step in _selected_train_steps(steps):
        selected_steps.append(
            {
                "step": step["step"],
                "score_mean": _metric(step, "critic/score/mean"),
                "reward_mean": _metric(step, "critic/rewards/mean"),
                "entropy": _metric(step, "actor/entropy"),
                "kl_loss": _metric(step, "actor/kl_loss"),
                "pg_loss": _metric(step, "actor/pg_loss"),
                "grad_norm": _metric(step, "actor/grad_norm"),
                "response_len": _metric(step, "response_length/mean"),
                "aborted_ratio": _metric(step, "response/aborted_ratio"),
                "turns": _metric(step, "num_turns/mean"),
                "throughput": _metric(step, "perf/throughput"),
                "step_seconds": _metric(step, "timing_s/step"),
            }
        )

    return {
        "model": model_label or _model_label(run_name),
        "stage": stage_label or _stage_label(run_name),
        "run_name": run_name,
        "case": case,
        "seed": seed,
        "actor_lr": config.get("GRPO_ACTOR_LR") or config.get("PPO_ACTOR_LR"),
        "critic_lr": config.get("PPO_CRITIC_LR"),
        "kl_coef": config.get("GRPO_KL_COEF") or config.get("PPO_KL_COEF"),
        "temperature": config.get("GRPO_ROLLOUT_TEMPERATURE") or config.get("PPO_ROLLOUT_TEMPERATURE"),
        "group_size": config.get("GRPO_GROUP_SIZE") or config.get("PPO_ROLLOUT_N"),
        "rollout_max_num_seqs": config.get("GRPO_ROLLOUT_MAX_NUM_SEQS") or config.get("PPO_ROLLOUT_MAX_NUM_SEQS"),
        "tool_cost": config.get("MTP_REWARD_TOOL_COST_WEIGHT", "0.0"),
        "under_tool": config.get("MTP_REWARD_UNDER_TOOL_WEIGHT", "0.0"),
        "min_tool": config.get("MTP_REWARD_MIN_TOOL_COUNT", "0"),
        "initial_val_acc": _metric(initial_val, "val-core/locomo_memory_tool_planner/acc/mean@1"),
        "initial_val_reward": _metric(initial_val, "val-aux/locomo_memory_tool_planner/reward/mean@1"),
        "final_train_step": final_train["step"] if final_train else None,
        "final_train_score_mean": _metric(final_train, "critic/score/mean"),
        "final_train_reward_mean": _metric(final_train, "critic/rewards/mean"),
        "final_actor_entropy": _metric(final_train, "actor/entropy"),
        "final_actor_kl_loss": _metric(final_train, "actor/kl_loss"),
        "final_actor_pg_loss": _metric(final_train, "actor/pg_loss"),
        "final_actor_grad_norm": _metric(final_train, "actor/grad_norm"),
        "final_response_length_mean": _metric(final_train, "response_length/mean"),
        "final_aborted_ratio": _metric(final_train, "response/aborted_ratio"),
        "final_turns_mean": _metric(final_train, "num_turns/mean"),
        "final_throughput": _metric(final_train, "perf/throughput"),
        "final_step_seconds": _metric(final_train, "timing_s/step"),
        "selected_steps": selected_steps,
        "log": str(path),
    }


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
            row.get("tool_cost"),
            row.get("under_tool"),
            row.get("min_tool"),
        )
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, group_rows in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0]))):
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
            tool_cost,
            under_tool,
            min_tool,
        ) = key
        out: dict[str, Any] = {
            "model": model,
            "stage": stage,
            "case": case,
            "seeds": sorted(row["seed"] for row in group_rows if row.get("seed") is not None),
            "n": len(group_rows),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "kl_coef": kl_coef,
            "temperature": temperature,
            "group_size": group_size,
            "rollout_max_num_seqs": rollout_max_num_seqs,
            "tool_cost": tool_cost,
            "under_tool": under_tool,
            "min_tool": min_tool,
        }
        for metric in FINAL_METRICS:
            out[f"{metric}_mean"] = _mean(_numeric(group_rows, metric))
            out[f"{metric}_std"] = _std(_numeric(group_rows, metric))
        aggregate_rows.append(out)
    return aggregate_rows


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _mean_std(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None:
        return ""
    return f"{_cell(mean)}+/-{_cell(std)}"


def _markdown(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model | stage | run | case | seed | lr | kl | temp | group | max seqs | tool_cost | under_tool | init val acc | last step | train score | train reward | entropy | kl loss | response len | throughput | step seconds |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {stage} | {run_name} | {case} | {seed} | {actor_lr} | {kl_coef} | {temperature} | {group_size} | {rollout_max_num_seqs} | {tool_cost} | {under_tool} | {initial_val_acc} | {final_train_step} | {final_train_score_mean} | {final_train_reward_mean} | {final_actor_entropy} | {final_actor_kl_loss} | {final_response_length_mean} | {final_throughput} | {final_step_seconds} |".format(
                **{key: _cell(value) for key, value in row.items()}
            )
        )

    lines.extend(
        [
            "",
            "| model | stage | run | case | seed | step | score | reward | entropy | kl loss | pg loss | response len | throughput | step seconds |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        for step in row.get("selected_steps", []):
            values = {key: _cell(value) for key, value in step.items()}
            lines.append(
                f"| {_cell(row['model'])} | {_cell(row['stage'])} | {_cell(row['run_name'])} | {_cell(row['case'])} | {_cell(row['seed'])} | {values['step']} | {values['score_mean']} | {values['reward_mean']} | {values['entropy']} | {values['kl_loss']} | {values['pg_loss']} | {values['response_len']} | {values['throughput']} | {values['step_seconds']} |"
            )

    if aggregate_rows:
        lines.extend(
            [
                "",
                "| model | stage | case | seeds | n | lr | kl | temp | group | max seqs | init val acc mean+/-std | last step mean+/-std | train score mean+/-std | train reward mean+/-std | entropy mean+/-std | kl loss mean+/-std | response len mean+/-std | throughput mean+/-std |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in aggregate_rows:
            seeds = ",".join(str(seed) for seed in row["seeds"])
            lines.append(
                "| {model} | {stage} | {case} | {seeds} | {n} | {actor_lr} | {kl_coef} | {temperature} | {group_size} | {rollout_max_num_seqs} | {init_acc} | {last_step} | {score} | {reward} | {entropy} | {kl_loss} | {response_len} | {throughput} |".format(
                    model=_cell(row["model"]),
                    stage=_cell(row["stage"]),
                    case=_cell(row["case"]),
                    seeds=seeds,
                    n=_cell(row["n"]),
                    actor_lr=_cell(row["actor_lr"]),
                    kl_coef=_cell(row["kl_coef"]),
                    temperature=_cell(row["temperature"]),
                    group_size=_cell(row["group_size"]),
                    rollout_max_num_seqs=_cell(row["rollout_max_num_seqs"]),
                    init_acc=_mean_std(row, "initial_val_acc"),
                    last_step=_mean_std(row, "final_train_step"),
                    score=_mean_std(row, "final_train_score_mean"),
                    reward=_mean_std(row, "final_train_reward_mean"),
                    entropy=_mean_std(row, "final_actor_entropy"),
                    kl_loss=_mean_std(row, "final_actor_kl_loss"),
                    response_len=_mean_std(row, "final_response_length_mean"),
                    throughput=_mean_std(row, "final_throughput"),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize VERL training-process metrics from GRPO/PPO log files.")
    parser.add_argument("--log-glob", required=True, help="Glob for per-run *.log files.")
    parser.add_argument("--run-config-dir", default="outputs/v0/run_configs")
    parser.add_argument("--model-label", help="Optional model label to force in output tables.")
    parser.add_argument("--stage-label", help="Optional stage label to force in output tables.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregate-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(match) for match in glob.glob(args.log_glob)):
        row = _summarize_run(path, Path(args.run_config_dir), args.model_label, args.stage_label)
        if row is not None:
            rows.append(row)

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
