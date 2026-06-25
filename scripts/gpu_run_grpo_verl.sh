#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.conda-py312}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/v0}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
RL_DATA_DIR="${RL_DATA_DIR:-$DATA_DIR/rl}"
GRPO_DATA_VARIANT="${GRPO_DATA_VARIANT:-teacher_best_of_4}"
RUN_NAME="${GRPO_RUN_NAME:-qwen3_8b_${GRPO_DATA_VARIANT}}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$OUTPUT_ROOT/sft/qwen3_8b/global_step_72/huggingface}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${GRPO_PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
GROUP_SIZE="${GRPO_GROUP_SIZE:-4}"
ROLLOUT_TEMPERATURE="${GRPO_ROLLOUT_TEMPERATURE:-0.7}"
ACTOR_LR="${GRPO_ACTOR_LR:-1e-6}"
KL_COEF="${GRPO_KL_COEF:-0.001}"
SEED="${GRPO_SEED:-42}"
MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-3584}"
MAX_RESPONSE_LENGTH="${GRPO_MAX_RESPONSE_LENGTH:-512}"
ROLLOUT_MAX_MODEL_LEN="${GRPO_ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_MAX_NUM_SEQS="${GRPO_ROLLOUT_MAX_NUM_SEQS:-16}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${GRPO_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$ROLLOUT_MAX_MODEL_LEN}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${GRPO_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}"
ROLLOUT_ENFORCE_EAGER="${GRPO_ROLLOUT_ENFORCE_EAGER:-True}"
PPO_MAX_TOKEN_LEN_PER_GPU="${GRPO_PPO_MAX_TOKEN_LEN_PER_GPU:-$ROLLOUT_MAX_MODEL_LEN}"
TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS:-1}"
SAVE_FREQ="${GRPO_SAVE_FREQ:-24}"
TEST_FREQ="${GRPO_TEST_FREQ:-24}"
SAVE_DIR="${GRPO_SAVE_DIR:-$OUTPUT_ROOT/grpo/$RUN_NAME}"
EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME:-grpo_${RUN_NAME}_locomo}"
TRAIN_FILES="${GRPO_TRAIN_FILES:-$RL_DATA_DIR/${GRPO_DATA_VARIANT}_train.parquet}"
VAL_FILES="${GRPO_VAL_FILES:-$RL_DATA_DIR/${GRPO_DATA_VARIANT}_valid.parquet}"
VAL_BATCH_SIZE="${GRPO_VAL_BATCH_SIZE:-null}"
VAL_ONLY="${GRPO_VAL_ONLY:-False}"
VAL_BEFORE_TRAIN="${GRPO_VAL_BEFORE_TRAIN:-True}"
MEMORY_STORES="${MTP_MEMORY_STORES_PATH:-$PROJECT_DIR/memory_store/locomo_evermemos_memory_stores.embedded.jsonl}"
RUN_CONFIG_PATH="${GRPO_RUN_CONFIG_PATH:-$OUTPUT_ROOT/run_configs/${RUN_NAME}.json}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
if [[ "${GRPO_ENABLE_EXPANDABLE_SEGMENTS:-0}" == "1" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
else
  unset PYTORCH_CUDA_ALLOC_CONF
fi
export MTP_MEMORY_STORES_PATH="$MEMORY_STORES"
export MTP_REWARD_EVAL_MODE="${MTP_REWARD_EVAL_MODE:-lexical}"
export MTP_NUM_JUDGE_RUNS="${MTP_NUM_JUDGE_RUNS:-1}"
export MTP_MEMORY_TOKEN_BUDGET="${MTP_MEMORY_TOKEN_BUDGET:-0}"
export MTP_RETRIEVAL_TOP_K="${MTP_RETRIEVAL_TOP_K:-5}"
export MTP_REWARD_ANSWER_WEIGHT="${MTP_REWARD_ANSWER_WEIGHT:-0.0}"
export MTP_REWARD_EVIDENCE_WEIGHT="${MTP_REWARD_EVIDENCE_WEIGHT:-1.0}"
export MTP_REWARD_PARTIAL_ANSWER_WEIGHT="${MTP_REWARD_PARTIAL_ANSWER_WEIGHT:-0.0}"
export MTP_REWARD_TOKEN_COST_WEIGHT="${MTP_REWARD_TOKEN_COST_WEIGHT:-0.0}"
export MTP_REWARD_TOOL_COST_WEIGHT="${MTP_REWARD_TOOL_COST_WEIGHT:-0.0}"
export MTP_REWARD_UNDER_TOOL_WEIGHT="${MTP_REWARD_UNDER_TOOL_WEIGHT:-0.0}"
export MTP_REWARD_MIN_TOOL_COUNT="${MTP_REWARD_MIN_TOOL_COUNT:-0}"
export MTP_ANSWER_PROMPT_STYLE="${MTP_ANSWER_PROMPT_STYLE:-concise}"
export MTP_REWARD_TRACE_PATH="${MTP_REWARD_TRACE_PATH:-$OUTPUT_ROOT/reward_debug/${RUN_NAME}_reward_traces.jsonl}"
export OPENAI_JUDGE_MODEL="${OPENAI_JUDGE_MODEL:-}"
export MEMORY_TOOL_PLANNER_OFFLINE="${MTP_MEMORY_TOOL_PLANNER_OFFLINE:-0}"
export GRPO_RUN_NAME="$RUN_NAME"
export GRPO_DATA_VARIANT="$GRPO_DATA_VARIANT"
export ACTOR_MODEL_PATH="$ACTOR_MODEL_PATH"
export GRPO_TRAIN_FILES="$TRAIN_FILES"
export GRPO_VAL_FILES="$VAL_FILES"
export GRPO_TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
export GRPO_PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE"
export GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU="$PPO_MICRO_BATCH_SIZE_PER_GPU"
export GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
export GRPO_GROUP_SIZE="$GROUP_SIZE"
export GRPO_ROLLOUT_TEMPERATURE="$ROLLOUT_TEMPERATURE"
export GRPO_MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH"
export GRPO_MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH"
export GRPO_ROLLOUT_MAX_MODEL_LEN="$ROLLOUT_MAX_MODEL_LEN"
export GRPO_ROLLOUT_MAX_NUM_SEQS="$ROLLOUT_MAX_NUM_SEQS"
export GRPO_ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
export GRPO_ROLLOUT_GPU_MEMORY_UTILIZATION="$ROLLOUT_GPU_MEMORY_UTILIZATION"
export GRPO_ROLLOUT_ENFORCE_EAGER="$ROLLOUT_ENFORCE_EAGER"
export GRPO_PPO_MAX_TOKEN_LEN_PER_GPU="$PPO_MAX_TOKEN_LEN_PER_GPU"
export GRPO_ACTOR_LR="$ACTOR_LR"
export GRPO_KL_COEF="$KL_COEF"
export GRPO_TOTAL_EPOCHS="$TOTAL_EPOCHS"
export GRPO_SAVE_FREQ="$SAVE_FREQ"
export GRPO_TEST_FREQ="$TEST_FREQ"
export GRPO_SEED="$SEED"
export VERL_GLOBAL_SEED="${VERL_GLOBAL_SEED:-$SEED}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-$SEED}"

if [[ "$VAL_ONLY" == "True" || "$VAL_ONLY" == "true" || "$VAL_ONLY" == "1" ]]; then
  export MTP_REWARD_TRACE_UNIQUE_KEY="${MTP_REWARD_TRACE_UNIQUE_KEY:-question_id}"
  if [[ -n "${MTP_REWARD_TRACE_PATH:-}" ]]; then
    mkdir -p "$(dirname "$MTP_REWARD_TRACE_PATH")"
    : > "$MTP_REWARD_TRACE_PATH"
  fi
fi

"$PYTHON_BIN" scripts/write_run_config_snapshot.py "$RUN_CONFIG_PATH"

"$PYTHON_BIN" - "$MEMORY_STORES" <<'PY'
import json
import os
import sys
from pathlib import Path

stores_path = Path(sys.argv[1])
if not stores_path.exists():
    raise SystemExit(f"memory store not found: {stores_path}")

allow_keyword_only = os.environ.get("MTP_ALLOW_KEYWORD_ONLY_RETRIEVAL", "0") == "1"
checked = 0
embedded = 0
with stores_path.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        checked += 1
        row = json.loads(line)
        embedding = row.get("embedding")
        if isinstance(embedding, list) and embedding:
            embedded += 1
        if checked >= 100:
            break

if checked == 0:
    raise SystemExit(f"memory store is empty: {stores_path}")
if embedded == 0 and not allow_keyword_only:
    raise SystemExit(
        f"memory store has no embeddings in first {checked} rows: {stores_path}; "
        "build an embedded store or set MTP_ALLOW_KEYWORD_ONLY_RETRIEVAL=1"
    )
print(json.dumps({"memory_store": str(stores_path), "sampled": checked, "embedded": embedded}, ensure_ascii=False))
PY

"$PYTHON_BIN" - "$ACTOR_MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
tokenizer_config = model_dir / "tokenizer_config.json"
if tokenizer_config.exists() and (model_dir / "tokenizer.json").exists():
    data = json.loads(tokenizer_config.read_text())
    if data.get("tokenizer_class") == "Qwen2Tokenizer":
        data["tokenizer_class"] = "Qwen2TokenizerFast"
        tokenizer_config.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
PY

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.seed="$SEED" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.prompt_key=prompt \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.data_loader_seed="$SEED" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","hf_model","optimizer","extra"]' \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.seed="$SEED" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.seed="$SEED" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.n="$GROUP_SIZE" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.seed="$SEED" \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$N_GPUS_PER_NODE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  reward_model.enable=False \
  custom_reward_function.path="$PROJECT_DIR/scripts/verl_memory_tool_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.project_name=memory_tool_planner \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.default_hdfs_dir=null \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.val_only="$VAL_ONLY"
