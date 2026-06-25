#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.conda-py312}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/v0}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
SFT_DATA_DIR="${SFT_DATA_DIR:-$DATA_DIR/sft}"
SFT_VARIANT="${SFT_VARIANT:-teacher_best_of_4}"
RUN_NAME="${SFT_RUN_NAME:-qwen3_8b_${SFT_VARIANT}}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${SFT_MAX_LENGTH:-4096}"
MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-8192}"
LR="${SFT_LR:-1e-5}"
SAVE_DIR="${SFT_SAVE_DIR:-$OUTPUT_ROOT/sft/$RUN_NAME}"
EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-sft_${RUN_NAME}_locomo}"
NUM_WORKERS="${SFT_NUM_WORKERS:-4}"
TRAIN_FILES="${SFT_TRAIN_FILES:-$SFT_DATA_DIR/${SFT_VARIANT}_train.parquet}"
VAL_FILES="${SFT_VAL_FILES:-$SFT_DATA_DIR/${SFT_VARIANT}_valid.parquet}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.messages_key=messages \
  data.enable_thinking_key=enable_thinking \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_length="$MAX_LENGTH" \
  data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
  data.truncation=left \
  data.ignore_input_ids_mismatch=True \
  data.num_workers="$NUM_WORKERS" \
  model.path="$BASE_MODEL" \
  model.trust_remote_code=True \
  +model.override_config.attn_implementation=sdpa \
  model.enable_gradient_checkpointing=True \
  model.use_remove_padding=True \
  engine.model_dtype=bf16 \
  engine.dtype=bfloat16 \
  engine.param_offload=False \
  engine.optimizer_offload=False \
  optim.lr="$LR" \
  trainer.project_name=memory_tool_planner \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.default_hdfs_dir=null \
  trainer.logger='["console"]' \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.nnodes=1 \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=after_each_epoch \
  checkpoint.save_contents='["model","hf_model","optimizer","extra"]'
