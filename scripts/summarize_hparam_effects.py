#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "test_evidence_mean",
    "test_full_hit_mean",
    "test_valid_rate_mean",
    "test_hit_mean",
)


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {path}")
    return [row for row in data if isinstance(row, dict)]


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _mean_std(row: dict[str, Any], mean_key: str, std_key: str) -> str:
    mean = _as_float(row.get(mean_key))
    std = _as_float(row.get(std_key))
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.6f}"
    return f"{mean:.6f}+/-{std:.6f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.6f}"


def _baseline_from_args(args: argparse.Namespace) -> dict[str, float] | None:
    values = {
        "test_evidence_mean": args.baseline_test_evidence,
        "test_full_hit_mean": args.baseline_full_hit,
        "test_valid_rate_mean": args.baseline_valid_rate,
        "test_hit_mean": args.baseline_hit,
    }
    parsed = {key: _as_float(value) for key, value in values.items() if value is not None}
    return parsed or None


def _find_baseline(rows: list[dict[str, Any]], baseline_case: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("case") == baseline_case]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (int(row.get("n") or 0), _as_float(row.get("test_evidence_mean")) or -1.0))


def _risk_label(row: dict[str, Any], baseline: dict[str, float] | None, baseline_case: str) -> str:
    if row.get("case") == baseline_case:
        return "baseline"
    if not baseline:
        return "rank-only"
    delta_evidence = _as_float(row.get("delta_test_evidence"))
    delta_full_hit = _as_float(row.get("delta_full_hit"))
    delta_valid_rate = _as_float(row.get("delta_valid_rate"))
    evidence_std = _as_float(row.get("test_evidence_std"))
    if delta_evidence is None:
        return "missing"
    if delta_evidence < -0.002:
        return "regression"
    if delta_evidence > 0 and evidence_std is not None and evidence_std > abs(delta_evidence):
        return "unstable"
    if delta_evidence > 0 and (delta_full_hit is None or delta_full_hit >= -0.001) and (
        delta_valid_rate is None or delta_valid_rate >= -0.001
    ):
        return "candidate"
    if delta_evidence > 0:
        return "tradeoff"
    return "neutral"


def _enrich(rows: list[dict[str, Any]], baseline: dict[str, float] | None, baseline_case: str) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        out = dict(row)
        if baseline:
            out["delta_test_evidence"] = _delta(row, baseline, "test_evidence_mean")
            out["delta_full_hit"] = _delta(row, baseline, "test_full_hit_mean")
            out["delta_valid_rate"] = _delta(row, baseline, "test_valid_rate_mean")
            out["delta_hit"] = _delta(row, baseline, "test_hit_mean")
        out["effect_label"] = _risk_label(out, baseline, baseline_case)
        enriched.append(out)
    return sorted(
        enriched,
        key=lambda row: (
            _as_float(row.get("test_evidence_mean")) is not None,
            _as_float(row.get("test_evidence_mean")) or -1.0,
            _as_float(row.get("test_full_hit_mean")) or -1.0,
        ),
        reverse=True,
    )


def _delta(row: dict[str, Any], baseline: dict[str, float], key: str) -> float | None:
    value = _as_float(row.get(key))
    base = baseline.get(key)
    if value is None or base is None:
        return None
    return round(value - base, 6)


def _markdown(rows: list[dict[str, Any]], baseline: dict[str, float] | None, top_k: int) -> str:
    lines = []
    if baseline:
        lines.extend(
            [
                f"Baseline test evidence: {_cell(baseline.get('test_evidence_mean'))}",
                "",
            ]
        )
    lines.extend(
        [
            "| rank | model | stage | case | label | seeds | n | lr | kl | temp | group | max seqs | tool_cost | under_tool | min_tool | test evidence | delta evidence | full-hit | delta full-hit | valid-rate | delta valid-rate | empty | invalid-json | dup-query | dup-calls | invalid-args |",
            "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(rows[:top_k], start=1):
        seeds = row.get("seeds") or []
        if isinstance(seeds, list):
            seeds_text = ",".join(str(seed) for seed in seeds)
        else:
            seeds_text = str(seeds)
        lines.append(
            "| {rank} | {model} | {stage} | {case} | {label} | {seeds} | {n} | {lr} | {kl} | {temp} | {group} | {max_seqs} | {tool_cost} | {under_tool} | {min_tool} | {test_evidence} | {delta_evidence} | {full_hit} | {delta_full_hit} | {valid_rate} | {delta_valid_rate} | {empty} | {invalid_json} | {dup_query} | {dup_calls} | {invalid_args} |".format(
                rank=rank,
                model=_cell(row.get("model")),
                stage=_cell(row.get("stage")),
                case=_cell(row.get("case")),
                label=_cell(row.get("effect_label")),
                seeds=seeds_text,
                n=_cell(row.get("n")),
                lr=_cell(row.get("actor_lr")),
                kl=_cell(row.get("kl_coef")),
                temp=_cell(row.get("temperature")),
                group=_cell(row.get("group_size")),
                max_seqs=_cell(row.get("rollout_max_num_seqs")),
                tool_cost=_cell(row.get("tool_cost")),
                under_tool=_cell(row.get("under_tool")),
                min_tool=_cell(row.get("min_tool")),
                test_evidence=_mean_std(row, "test_evidence_mean", "test_evidence_std"),
                delta_evidence=_format_delta(_as_float(row.get("delta_test_evidence"))),
                full_hit=_mean_std(row, "test_full_hit_mean", "test_full_hit_std"),
                delta_full_hit=_format_delta(_as_float(row.get("delta_full_hit"))),
                valid_rate=_mean_std(row, "test_valid_rate_mean", "test_valid_rate_std"),
                delta_valid_rate=_format_delta(_as_float(row.get("delta_valid_rate"))),
                empty=_mean_std(row, "test_empty_calls_mean", "test_empty_calls_std"),
                invalid_json=_mean_std(row, "test_invalid_json_mean", "test_invalid_json_std"),
                dup_query=_mean_std(row, "test_duplicate_query_rate_mean", "test_duplicate_query_rate_std"),
                dup_calls=_mean_std(row, "test_duplicate_calls_mean", "test_duplicate_calls_std"),
                invalid_args=_mean_std(row, "test_invalid_arguments_mean", "test_invalid_arguments_std"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank hparam aggregate rows and compute deltas vs current.")
    parser.add_argument("--aggregate-json", required=True, help="Input aggregate JSON from summarize_hparam_sweep.py.")
    parser.add_argument("--baseline-case", default="current")
    parser.add_argument("--baseline-test-evidence", type=float)
    parser.add_argument("--baseline-full-hit", type=float)
    parser.add_argument("--baseline-valid-rate", type=float)
    parser.add_argument("--baseline-hit", type=float)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--output", required=True, help="Output JSON with enriched rows.")
    parser.add_argument("--markdown-output", help="Output paper-ready Markdown ranking table.")
    args = parser.parse_args()

    rows = _load_json(Path(args.aggregate_json))
    baseline = _baseline_from_args(args)
    if baseline is None:
        baseline_row = _find_baseline(rows, args.baseline_case)
        if baseline_row is not None:
            baseline = {
                key: value
                for key in METRIC_KEYS
                if (value := _as_float(baseline_row.get(key))) is not None
            }

    enriched = _enrich(rows, baseline, args.baseline_case)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown = _markdown(enriched, baseline, args.top_k)
    if args.markdown_output:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
