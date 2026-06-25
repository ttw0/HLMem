#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.conda-py312}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
PLANNER_DATA_DIR="${PLANNER_DATA_DIR:-$DATA_DIR/planner_data}"
SFT_DATA_DIR="${SFT_DATA_DIR:-$DATA_DIR/sft}"
RL_DATA_DIR="${RL_DATA_DIR:-$DATA_DIR/rl}"
MEMORY_STORE_DIR="${MEMORY_STORE_DIR:-$PROJECT_DIR/memory_store}"
LOCOMO_PATH="${LOCOMO_PATH:-$DATA_DIR/locomo/locomo10_precess.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/v0}"
CORE_MODE="${MTP_CORE_MODE:-llm}"
BOUNDARY_MODE="${MTP_BOUNDARY_MODE:-heuristic}"
EPISODE_MODE="${MTP_EPISODE_MODE:-heuristic}"
REBUILD_STORES="${MTP_REBUILD_STORES:-0}"
EMBED_MEMORY_STORES="${MTP_EMBED_MEMORY_STORES:-1}"
REBUILD_PLANNER_DATA="${MTP_REBUILD_PLANNER_DATA:-0}"
REBUILD_TEACHER_DATA="${MTP_REBUILD_TEACHER_DATA:-0}"
SFT_VARIANT="${SFT_VARIANT:-teacher_best_of_4}"
TEACHER_CANDIDATES_PER_SAMPLE="${MTP_TEACHER_CANDIDATES_PER_SAMPLE:-4}"
TEACHER_TEMPERATURE="${MTP_TEACHER_TEMPERATURE:-0.4}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$PLANNER_DATA_DIR" "$SFT_DATA_DIR" "$RL_DATA_DIR" "$MEMORY_STORE_DIR"

RAW_MEMORY_STORES="$MEMORY_STORE_DIR/locomo_evermemos_memory_stores.jsonl"
MEMORY_STORES="$MEMORY_STORE_DIR/locomo_evermemos_memory_stores.embedded.jsonl"

if [ "$REBUILD_STORES" = "0" ] && [ -s "$RAW_MEMORY_STORES" ]; then
  echo "Reusing existing memory store: $RAW_MEMORY_STORES"
else
  "$PYTHON_BIN" -m memory_tool_planner.cli build-locomo-memcell \
    --input "$LOCOMO_PATH" \
    --output "$RAW_MEMORY_STORES" \
    --boundary-mode "$BOUNDARY_MODE" \
    --episode-mode "$EPISODE_MODE" \
    --core-mode "$CORE_MODE"
fi

if [ "$EMBED_MEMORY_STORES" = "1" ]; then
  if [ "$REBUILD_STORES" = "0" ] && [ -s "$MEMORY_STORES" ]; then
    echo "Reusing embedded memory store: $MEMORY_STORES"
  else
    "$PYTHON_BIN" -m memory_tool_planner.cli index \
      --online \
      --input "$RAW_MEMORY_STORES" \
      --output "$MEMORY_STORES"
  fi
else
  MEMORY_STORES="$RAW_MEMORY_STORES"
  export MTP_ALLOW_KEYWORD_ONLY_RETRIEVAL=1
fi

for split in train valid test; do
  base_path="$PLANNER_DATA_DIR/locomo_planner_data.$split.base.jsonl"
  if [ "$REBUILD_PLANNER_DATA" = "0" ] && [ -s "$base_path" ]; then
    echo "Reusing planner data: $base_path"
  else
    "$PYTHON_BIN" -m memory_tool_planner.cli prepare-planner-data \
      --locomo "$LOCOMO_PATH" \
      --output "$base_path" \
      --split "$split"
  fi
done

for split in train valid; do
  teacher_path="$PLANNER_DATA_DIR/locomo_planner_data.$split.$SFT_VARIANT.jsonl"
  if [ "$REBUILD_TEACHER_DATA" = "0" ] && [ -s "$teacher_path" ]; then
    echo "Reusing teacher data: $teacher_path"
  else
    "$PYTHON_BIN" -m memory_tool_planner.cli generate-teacher-sft-data \
      --online \
      --planner-data "$PLANNER_DATA_DIR/locomo_planner_data.$split.base.jsonl" \
      --stores "$MEMORY_STORES" \
      --output "$teacher_path" \
      --summary "$PLANNER_DATA_DIR/locomo_planner_data.$split.$SFT_VARIANT.summary.json" \
      --candidates-per-sample "$TEACHER_CANDIDATES_PER_SAMPLE" \
      --temperature "$TEACHER_TEMPERATURE" \
      --token-budget 0
  fi
done

"$PYTHON_BIN" -m memory_tool_planner.cli export-sft-data \
  --planner-data "$PLANNER_DATA_DIR/locomo_planner_data.train.$SFT_VARIANT.jsonl" \
  --output "$SFT_DATA_DIR/${SFT_VARIANT}_train.parquet"

"$PYTHON_BIN" -m memory_tool_planner.cli export-sft-data \
  --planner-data "$PLANNER_DATA_DIR/locomo_planner_data.valid.$SFT_VARIANT.jsonl" \
  --output "$SFT_DATA_DIR/${SFT_VARIANT}_valid.parquet"

"$PYTHON_BIN" -m memory_tool_planner.cli export-rl-data \
  --planner-data "$PLANNER_DATA_DIR/locomo_planner_data.train.base.jsonl" \
  --stores "$MEMORY_STORES" \
  --output "$RL_DATA_DIR/${SFT_VARIANT}_train.parquet" \
  --drop-empty-evidence

"$PYTHON_BIN" -m memory_tool_planner.cli export-rl-data \
  --planner-data "$PLANNER_DATA_DIR/locomo_planner_data.valid.base.jsonl" \
  --stores "$MEMORY_STORES" \
  --output "$RL_DATA_DIR/${SFT_VARIANT}_valid.parquet" \
  --drop-empty-evidence

"$PYTHON_BIN" -m memory_tool_planner.cli export-rl-data \
  --planner-data "$PLANNER_DATA_DIR/locomo_planner_data.test.base.jsonl" \
  --stores "$MEMORY_STORES" \
  --output "$RL_DATA_DIR/${SFT_VARIANT}_test.parquet" \
  --drop-empty-evidence
