# Data Notes

This repository includes a small LoCoMo-derived example for reproducing the
planner data path.

Included files:

- `data/locomo/locomo10.json`: source conversation subset.
- `data/locomo/locomo10_precess.json`: derived file with `qa[].person`
  annotations.
- `data/planner_data/*.base.jsonl`: base planner-data splits.
- `data/sft/teacher_best_of_4_*.jsonl`: lightweight SFT examples.

Not included:

- prebuilt memory stores
- embedding files
- Parquet exports
- checkpoints
- training logs
- reward traces
- generated diagnostics

Regenerate these artifacts locally with the CLI commands in `README.md`.
