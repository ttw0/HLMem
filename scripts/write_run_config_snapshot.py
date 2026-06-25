#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ENV_KEYS = [
    "PROJECT_DIR",
    "OUTPUT_ROOT",
    "DATA_DIR",
    "RL_DATA_DIR",
    "GRPO_DATA_VARIANT",
    "GRPO_RUN_NAME",
    "ACTOR_MODEL_PATH",
    "GRPO_TRAIN_FILES",
    "GRPO_VAL_FILES",
    "GRPO_TRAIN_BATCH_SIZE",
    "GRPO_PPO_MINI_BATCH_SIZE",
    "GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU",
    "GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU",
    "GRPO_GROUP_SIZE",
    "GRPO_ROLLOUT_TEMPERATURE",
    "GRPO_MAX_PROMPT_LENGTH",
    "GRPO_MAX_RESPONSE_LENGTH",
    "GRPO_ROLLOUT_MAX_MODEL_LEN",
    "GRPO_ROLLOUT_MAX_NUM_SEQS",
    "GRPO_ROLLOUT_MAX_NUM_BATCHED_TOKENS",
    "GRPO_ROLLOUT_GPU_MEMORY_UTILIZATION",
    "GRPO_ROLLOUT_ENFORCE_EAGER",
    "GRPO_PPO_MAX_TOKEN_LEN_PER_GPU",
    "GRPO_ACTOR_LR",
    "GRPO_KL_COEF",
    "GRPO_TOTAL_EPOCHS",
    "GRPO_SAVE_FREQ",
    "GRPO_TEST_FREQ",
    "GRPO_SEED",
    "VERL_GLOBAL_SEED",
    "PYTHONHASHSEED",
    "GRPO_VAL_ONLY",
    "GRPO_VAL_BEFORE_TRAIN",
    "GRPO_VAL_BATCH_SIZE",
    "PPO_DATA_VARIANT",
    "PPO_RUN_NAME",
    "PPO_TRAIN_FILES",
    "PPO_VAL_FILES",
    "PPO_ROLLOUT_N",
    "PPO_ROLLOUT_TEMPERATURE",
    "PPO_ACTOR_LR",
    "PPO_CRITIC_LR",
    "PPO_KL_COEF",
    "PPO_GAMMA",
    "PPO_LAMBDA",
    "PPO_ACTOR_GRAD_CLIP",
    "PPO_CRITIC_GRAD_CLIP",
    "PPO_ACTOR_PPO_EPOCHS",
    "PPO_CRITIC_PPO_EPOCHS",
    "PPO_CRITIC_CLIPRANGE_VALUE",
    "PPO_DATA_LOADER_NUM_WORKERS",
    "PPO_ACTOR_MODEL_DTYPE",
    "PPO_REF_MODEL_DTYPE",
    "PPO_CRITIC_MODEL_DTYPE",
    "PPO_CRITIC_MAX_TOKEN_LEN_PER_GPU",
    "PPO_CRITIC_FORWARD_MAX_TOKEN_LEN_PER_GPU",
    "PPO_ENABLE_EXPANDABLE_SEGMENTS",
    "PPO_TOTAL_EPOCHS",
    "PPO_SEED",
    "PPO_VAL_ONLY",
    "PPO_VAL_BEFORE_TRAIN",
    "PPO_VAL_BATCH_SIZE",
    "CRITIC_MODEL_PATH",
    "MTP_MEMORY_STORES_PATH",
    "MTP_SKIP_ANSWER_GENERATION",
    "MTP_REWARD_EVAL_MODE",
    "MTP_REWARD_ANSWER_WEIGHT",
    "MTP_REWARD_EVIDENCE_WEIGHT",
    "MTP_REWARD_PARTIAL_ANSWER_WEIGHT",
    "MTP_REWARD_TOKEN_COST_WEIGHT",
    "MTP_REWARD_TOOL_COST_WEIGHT",
    "MTP_REWARD_UNDER_TOOL_WEIGHT",
    "MTP_REWARD_MIN_TOOL_COUNT",
    "MTP_MEMORY_TOKEN_BUDGET",
    "MTP_RETRIEVAL_TOP_K",
    "MTP_REWARD_TRACE_UNIQUE_KEY",
    "MTP_REWARD_TRACE_PATH",
    "MTP_ANSWER_PROMPT_STYLE",
    "OPENAI_JUDGE_MODEL",
    "HF_ENDPOINT",
]


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_run_config_snapshot.py OUTPUT_JSON")
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "git": {
            "commit": run_git(["rev-parse", "HEAD"]),
            "branch": run_git(["branch", "--show-current"]),
            "status_short": run_git(["status", "--short"]),
        },
        "env": {key: os.environ.get(key, "") for key in ENV_KEYS},
    }
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_config": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
