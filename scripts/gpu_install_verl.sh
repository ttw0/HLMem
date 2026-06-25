#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.conda-py312}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
PIP_BIN="${PIP_BIN:-$ENV_DIR/bin/pip}"
VERL_DIR="${VERL_DIR:-$PROJECT_DIR/third_party/verl}"
USE_SGLANG="${USE_SGLANG:-0}"
USE_VERL_OFFICIAL_INSTALL="${USE_VERL_OFFICIAL_INSTALL:-0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
VLLM_WHEEL="${VLLM_WHEEL:-$PROJECT_DIR/third_party/wheelhouse/vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl}"
PIP_OPTS=(--retries 10 --timeout 120)
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-}"
VERL_PIP_INDEX_URL="${VERL_PIP_INDEX_URL:-https://pypi.org/simple}"
export PIP_INDEX_URL PIP_TRUSTED_HOST
export PATH="$ENV_DIR/bin:$PATH"

cd "$PROJECT_DIR"

echo "Using pip index: $PIP_INDEX_URL"
echo "Using python: $("$PYTHON_BIN" --version)"
echo "Using pip: $(command -v pip)"

"$PIP_BIN" install "${PIP_OPTS[@]}" --upgrade pip wheel "setuptools<80,>=77.0.3"
"$PIP_BIN" install "${PIP_OPTS[@]}" -e "$PROJECT_DIR"
"$PIP_BIN" install "${PIP_OPTS[@]}" pyarrow pandas datasets

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  "$PIP_BIN" install "${PIP_OPTS[@]}" --index-url "$PYTORCH_INDEX_URL" \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
fi

if [ -f "$VLLM_WHEEL" ] && ! "$PYTHON_BIN" -c "import vllm" >/dev/null 2>&1; then
  "$PIP_BIN" install "${PIP_OPTS[@]}" --no-cache-dir "$VLLM_WHEEL"
fi

if [ ! -d "$VERL_DIR/.git" ]; then
  mkdir -p "$(dirname "$VERL_DIR")"
  git clone https://github.com/verl-project/verl.git "$VERL_DIR"
fi

cd "$VERL_DIR"
export PIP_INDEX_URL="$VERL_PIP_INDEX_URL"
echo "Using verl pip index: $PIP_INDEX_URL"
if [ "$USE_VERL_OFFICIAL_INSTALL" = "1" ]; then
  USE_MEGATRON=0 USE_SGLANG="$USE_SGLANG" bash scripts/install_vllm_sglang_mcore.sh
else
  "$PIP_BIN" install "${PIP_OPTS[@]}" \
    "transformers[hf_xet]>=4.51.0" \
    accelerate \
    codetiming \
    datasets \
    dill \
    "fastapi[standard]>=0.115.0" \
    "grpcio>=1.62.1" \
    hf-transfer \
    hydra-core \
    liger-kernel \
    mathruler \
    "numpy<2.0.0" \
    "nvidia-ml-py>=12.560.30" \
    "optree>=0.13.0" \
    pandas \
    peft \
    pre-commit \
    py-spy \
    pyarrow \
    pybind11 \
    "pydantic>=2.9" \
    pylatexenc \
    pytest \
    qwen-vl-utils \
    "ray[default]" \
    ruff \
    tensorboard \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    torchdata \
    wandb
fi
"$PIP_BIN" install "${PIP_OPTS[@]}" --no-deps -e "$VERL_DIR"

"$PYTHON_BIN" - <<'PY'
import pyarrow
import torch
import verl

print("pyarrow", pyarrow.__version__)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("verl", verl.__file__)
PY
