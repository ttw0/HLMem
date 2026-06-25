# HLMem

HLMem is a lightweight research scaffold for training a memory tool planner. The
planner receives a user question and a small memory-tool schema, then emits
structured `tool_calls` for core and episodic memory search. The main training
signal is retrieval evidence quality: the policy is optimized to choose useful
memory tools and write better queries, not to train the answer model itself.

## What Is Included

- `src/memory_tool_planner/`: schemas, planner-output validation, retrieval,
  reward computation, data export, trace analysis, and CLI commands.
- `scripts/`: reproducible data preparation, SFT, planner-only GRPO, evaluation,
  and result-summary helpers.
- `data/locomo/`: the small LoCoMo subset used by the examples, plus a derived
  person-annotated file.
- `data/planner_data/`: base planner-data JSONL splits.
- `data/sft/`: lightweight SFT JSONL examples. Parquet files are generated
  locally and are not committed.
- `tests/`: smoke and behavior tests for the planner stack.

Generated outputs, reward traces, checkpoints, logs, prebuilt memory stores, and
API credentials are intentionally excluded.

## Install

```bash
uv sync --extra dev
```

For data export:

```bash
uv sync --extra data
```

For GPU training with VERL, create a Python 3.12 environment on the training
machine, install this project, then run:

```bash
bash scripts/gpu_install_verl.sh
```

The training scripts are environment-variable driven. Copy `.env.example` to
`.env` and fill OpenAI-compatible endpoints only when online model calls are
needed.

## Local Smoke Test

Run the unit tests:

```bash
uv run python -m unittest discover -s tests
```

Run an offline end-to-end dry run:

```bash
uv run mtp dry-run --output outputs/v0/dry_run
```

Prepare a tiny planner-data sample:

```bash
uv run mtp prepare-planner-data \
  --locomo data/locomo/locomo10_precess.json \
  --output outputs/v0/planner_data/sample.jsonl \
  --limit-qa 5
```

## Data Preparation

The included `locomo10_precess.json` already contains the `person` annotation
needed by planner training. To regenerate it with an OpenAI-compatible chat
model:

```bash
MEMORY_TOOL_PLANNER_OFFLINE=0 \
uv run python scripts/annotate_locomo_person.py \
  --input data/locomo/locomo10.json \
  --output data/locomo/locomo10_precess.json \
  --parallel-workers 4
```

Build planner-data splits:

```bash
for split in train valid test; do
  uv run mtp prepare-planner-data \
    --locomo data/locomo/locomo10_precess.json \
    --output outputs/v0/planner_data/locomo_planner_data.${split}.base.jsonl \
    --split "$split"
done
```

Teacher-distilled SFT labels can be regenerated with:

```bash
MEMORY_TOOL_PLANNER_OFFLINE=0 \
uv run mtp generate-teacher-sft-data \
  --planner-data outputs/v0/planner_data/locomo_planner_data.train.base.jsonl \
  --stores memory_store/locomo_memory_stores.embedded.jsonl \
  --output outputs/v0/planner_data/locomo_planner_data.train.teacher_best_of_4.jsonl \
  --summary outputs/v0/planner_data/locomo_planner_data.train.teacher_best_of_4.summary.json \
  --candidates-per-sample 4 \
  --temperature 0.4 \
  --token-budget 0
```

Export SFT and RL training files:

```bash
uv run mtp export-sft-data \
  --planner-data outputs/v0/planner_data/locomo_planner_data.train.teacher_best_of_4.jsonl \
  --output data/sft/teacher_best_of_4_train.parquet

uv run mtp export-rl-data \
  --planner-data outputs/v0/planner_data/locomo_planner_data.train.base.jsonl \
  --stores memory_store/locomo_memory_stores.embedded.jsonl \
  --output data/rl/teacher_best_of_4_train.parquet \
  --drop-empty-evidence
```

## Training

SFT initializes a base chat model so it can reliably emit legal planner JSON:

```bash
PROJECT_DIR="$(pwd)" \
SFT_VARIANT=teacher_best_of_4 \
BASE_MODEL=Qwen/Qwen3-8B \
bash scripts/gpu_run_sft_verl.sh
```

Planner-only GRPO optimizes evidence recall and query construction. It disables
answer generation and answer-judge reward by default:

```bash
PROJECT_DIR="$(pwd)" \
GRPO_DATA_VARIANT=teacher_best_of_4 \
ACTOR_MODEL_PATH=outputs/v0/sft/qwen3_8b_teacher_best_of_4/global_step_72/huggingface \
bash scripts/gpu_run_planner_only_grpo.sh
```

Important environment variables:

- `MTP_MEMORY_STORES_PATH`: embedded JSONL memory store used by reward execution.
- `MTP_REWARD_TRACE_PATH`: JSONL reward trace output.
- `MTP_RETRIEVAL_TOP_K`: executor top-k, default `5`.
- `MTP_REWARD_EVIDENCE_WEIGHT`: evidence reward weight, default `1.0`.
- `MTP_REWARD_ANSWER_WEIGHT`: answer reward weight, default `0.0`.

## Evaluation

Run deterministic planner reward evaluation with generated planner outputs:

```bash
uv run mtp reward-runner \
  --planner-data data/planner_data/locomo_planner_data.valid.base.jsonl \
  --stores memory_store/locomo_memory_stores.embedded.jsonl \
  --output outputs/v0/reward_debug/valid.rewards.jsonl \
  --summary outputs/v0/reward_debug/valid.summary.json \
  --offline \
  --skip-generation
```

Summarize paired traces:

```bash
uv run mtp analyze-traces \
  --baseline-trace outputs/v0/reward_debug/baseline.jsonl \
  --candidate-trace outputs/v0/reward_debug/grpo.jsonl \
  --output outputs/v0/diagnostics/paired_errors.jsonl \
  --summary outputs/v0/diagnostics/paired_summary.json
```

## License

Apache-2.0. See `LICENSE`.
