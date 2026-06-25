#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memory_tool_planner.locomo_planner_data import read_jsonl
from memory_tool_planner.trace_analysis import assert_summary_matches_trace, dedupe_trace_rows, planner_metric_summary


def summarize_trace(path: Path) -> dict[str, Any]:
    raw_rows = read_jsonl(path)
    rows = dedupe_trace_rows(raw_rows)
    summary = planner_metric_summary(rows)
    summary["trace"] = str(path)
    summary["raw_rows"] = len(raw_rows)
    summary["unique_question_id"] = len(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize deterministic planner reward traces and optionally verify an existing summary.")
    parser.add_argument("--trace", action="append", required=True, help="Reward trace JSONL. Can be passed multiple times.")
    parser.add_argument("--name", action="append", help="Display name for each trace. Defaults to path stem.")
    parser.add_argument("--output", required=True, help="Output summary JSON path.")
    parser.add_argument("--verify-existing", help="Existing summary JSON to verify against the provided trace names.")
    args = parser.parse_args()

    trace_paths = [Path(item) for item in args.trace]
    names = args.name or [path.stem for path in trace_paths]
    if len(names) != len(trace_paths):
        raise SystemExit("--name count must match --trace count")

    result: dict[str, Any] = {}
    for name, path in zip(names, trace_paths, strict=True):
        result[name] = summarize_trace(path)

    if args.verify_existing:
        existing = json.loads(Path(args.verify_existing).read_text(encoding="utf-8"))
        for name, summary in result.items():
            if name not in existing:
                raise SystemExit(f"missing summary entry in {args.verify_existing}: {name}")
            rows = dedupe_trace_rows(read_jsonl(Path(summary["trace"])))
            assert_summary_matches_trace(existing[name], rows, label=name)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
