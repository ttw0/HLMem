#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_parquet_rows(path: str) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _write_parquet_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _question_id_from_rl(row: dict[str, Any]) -> str:
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    question_id = extra.get("question_id")
    if question_id:
        return str(question_id)
    ground_truth = row.get("reward_model", {}).get("ground_truth", "{}")
    try:
        parsed = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except json.JSONDecodeError:
        parsed = {}
    return str(parsed.get("question_id", ""))


def _question_id_from_trace(row: dict[str, Any]) -> str:
    return str(row.get("question_id", ""))


def _dedupe_trace_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep the last row for each question_id, matching eval summary behavior."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = _question_id_from_trace(row)
        if qid:
            by_id[qid] = row
    return by_id


def _evidence_score(row: dict[str, Any]) -> float:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    try:
        return float(reward.get("evidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _gold_count_from_rl(row: dict[str, Any]) -> int:
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    evidence = extra.get("evidence")
    if evidence is not None:
        return len(evidence)
    ground_truth = row.get("reward_model", {}).get("ground_truth", "{}")
    try:
        parsed = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except json.JSONDecodeError:
        parsed = {}
    gold = parsed.get("gold_evidence_ids", [])
    return len(gold) if isinstance(gold, list) else 0


def _trace_scores_by_id(trace_paths: list[str]) -> tuple[dict[str, list[float]], dict[str, Any]]:
    scores_by_id: dict[str, list[float]] = {}
    summary = {"traces": []}
    for path_value in trace_paths:
        path = Path(path_value)
        rows = _read_jsonl(path)
        deduped = _dedupe_trace_rows(rows)
        for qid, row in deduped.items():
            scores_by_id.setdefault(qid, []).append(_evidence_score(row))
        summary["traces"].append(
            {
                "path": str(path),
                "rows": len(rows),
                "unique_question_id": len(deduped),
            }
        )
    return scores_by_id, summary


def _single_trace_score_by_id(trace_path: str | None) -> dict[str, float]:
    if not trace_path:
        return {}
    return {qid: _evidence_score(row) for qid, row in _dedupe_trace_rows(_read_jsonl(Path(trace_path))).items()}


def _score_stats(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
    }


def build_hard_rows(base_rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    trace_paths = list(args.trace or [])
    trace_scores_by_id, trace_summary = _trace_scores_by_id(trace_paths)
    reference_scores_by_id = _single_trace_score_by_id(args.reference_trace)
    output_rows: list[dict[str, Any]] = []
    hard_question_rows: list[dict[str, Any]] = []
    repeat_counts: Counter[int] = Counter()
    reasons: Counter[str] = Counter()
    score_buckets: Counter[str] = Counter()

    for row in base_rows:
        qid = _question_id_from_rl(row)
        scores = trace_scores_by_id.get(qid, [])
        stats = _score_stats(scores)
        min_evidence = stats["min"]
        max_evidence = stats["max"]
        reference_evidence = reference_scores_by_id.get(qid)
        repeats = 1
        row_reasons = []

        if not scores:
            repeats += max(0, args.missing_trace_repeat)
            if args.missing_trace_repeat > 0:
                row_reasons.append("missing_trace")
        elif min_evidence is not None and min_evidence < args.hard_threshold:
            repeats += max(0, args.hard_repeat)
            row_reasons.append("evidence_lt_1" if args.hard_threshold == 1.0 else "evidence_lt_threshold")

        if scores and min_evidence is not None and min_evidence <= args.zero_threshold:
            repeats += max(0, args.no_overlap_repeat)
            row_reasons.append("evidence_eq_0")

        if (
            scores
            and min_evidence is not None
            and max_evidence is not None
            and max_evidence - min_evidence >= args.disagreement_threshold
        ):
            repeats += max(0, args.disagreement_repeat)
            row_reasons.append("trace_disagreement")

        if reference_evidence is not None:
            candidate_best = max_evidence if max_evidence is not None else 0.0
            gap = reference_evidence - candidate_best
            if gap >= args.reference_gap_threshold:
                repeats += max(0, args.reference_gap_repeat)
                row_reasons.append("reference_gap")
        else:
            gap = None

        if _gold_count_from_rl(row) > 1:
            repeats += max(0, args.multi_evidence_repeat)
            row_reasons.append("multi_evidence")

        if scores:
            if min_evidence == 0.0:
                score_buckets["min_eq_0"] += 1
            elif min_evidence is not None and min_evidence < 1.0:
                score_buckets["min_between_0_and_1"] += 1
            elif min_evidence == 1.0:
                score_buckets["min_eq_1"] += 1
        else:
            score_buckets["missing_trace"] += 1

        for reason in row_reasons or ["base"]:
            reasons[reason] += 1
        repeat_counts[repeats] += 1

        if row_reasons:
            hard_question_rows.append(
                {
                    "question_id": qid,
                    "repeat_count": repeats,
                    "hard_reasons": row_reasons,
                    "trace_scores": scores,
                    "trace_score_min": min_evidence,
                    "trace_score_max": max_evidence,
                    "trace_score_mean": stats["mean"],
                    "reference_evidence": reference_evidence,
                    "reference_gap": gap,
                    "gold_evidence_count": _gold_count_from_rl(row),
                }
            )

        for repeat_index in range(repeats):
            copied = dict(row)
            extra = dict(copied.get("extra_info") or {})
            extra["hard_repeat_index"] = repeat_index
            extra["hard_repeat_count"] = repeats
            extra["hard_reasons"] = row_reasons
            extra["hard_trace_scores"] = scores
            extra["hard_trace_score_min"] = min_evidence
            extra["hard_trace_score_max"] = max_evidence
            extra["hard_trace_score_mean"] = stats["mean"]
            if reference_evidence is not None:
                extra["hard_reference_evidence"] = reference_evidence
                extra["hard_reference_gap"] = gap
            copied["extra_info"] = extra
            output_rows.append(copied)

    summary = {
        "input": args.input,
        "traces": trace_summary["traces"],
        "reference_trace": args.reference_trace,
        "output": args.output,
        "input_rows": len(base_rows),
        "output_rows": len(output_rows),
        "hard_question_rows": len(hard_question_rows),
        "repeat_counts": dict(sorted(repeat_counts.items())),
        "reasons": dict(sorted(reasons.items())),
        "score_buckets": dict(sorted(score_buckets.items())),
        "config": {
            "hard_threshold": args.hard_threshold,
            "hard_repeat": args.hard_repeat,
            "multi_evidence_repeat": args.multi_evidence_repeat,
            "no_overlap_repeat": args.no_overlap_repeat,
            "zero_threshold": args.zero_threshold,
            "disagreement_threshold": args.disagreement_threshold,
            "disagreement_repeat": args.disagreement_repeat,
            "reference_gap_threshold": args.reference_gap_threshold,
            "reference_gap_repeat": args.reference_gap_repeat,
            "missing_trace_repeat": args.missing_trace_repeat,
        },
    }
    return output_rows, summary, hard_question_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a hard-sampling RL parquet by repeating low-reward and multi-evidence samples.")
    parser.add_argument("--input", required=True, help="Base RL parquet.")
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        help="Reward trace used to identify hard samples. Can be passed multiple times; duplicate question_id rows keep the last trace row.",
    )
    parser.add_argument("--reference-trace", help="Optional stronger reference trace, e.g. oracle, used to upsample samples with remaining evidence gap.")
    parser.add_argument("--output", required=True, help="Output hard RL parquet.")
    parser.add_argument("--hard-threshold", type=float, default=1.0, help="Repeat samples whose minimum trace evidence is below this threshold.")
    parser.add_argument("--hard-repeat", type=int, default=3, help="Extra repeats for evidence below hard-threshold samples.")
    parser.add_argument("--multi-evidence-repeat", type=int, default=3, help="Extra repeats for samples with multiple gold evidence ids.")
    parser.add_argument("--no-overlap-repeat", type=int, default=3, help="Extra repeats for evidence <= zero-threshold samples.")
    parser.add_argument("--zero-threshold", type=float, default=0.0, help="Threshold for no-overlap repeats.")
    parser.add_argument("--disagreement-threshold", type=float, default=0.5, help="Minimum max-min evidence spread across traces for disagreement repeats.")
    parser.add_argument("--disagreement-repeat", type=int, default=0, help="Extra repeats for samples with disagreement across traces.")
    parser.add_argument("--reference-gap-threshold", type=float, default=0.25, help="Minimum reference minus candidate-best evidence gap for reference-gap repeats.")
    parser.add_argument("--reference-gap-repeat", type=int, default=0, help="Extra repeats for samples with a strong reference gap.")
    parser.add_argument("--missing-trace-repeat", type=int, default=0, help="Extra repeats for base samples missing from all traces.")
    args = parser.parse_args()

    base_rows = _read_parquet_rows(args.input)
    output_rows, summary, hard_question_rows = build_hard_rows(base_rows, args)

    output = Path(args.output)
    _write_parquet_rows(output, output_rows)
    summary["output"] = str(output)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hard_questions_path = output.with_suffix(".hard_questions.jsonl")
    with hard_questions_path.open("w", encoding="utf-8") as handle:
        for row in hard_question_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
