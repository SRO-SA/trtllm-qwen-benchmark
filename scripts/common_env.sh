#!/bin/bash
set -euo pipefail

# Shared TensorRT-LLM benchmark environment.
# Keep dependency versions compatible with the NVIDIA TensorRT-LLM container.
# IMPORTANT: avoid "pip install -U" in this environment.

export PYTHON_BIN="${PYTHON_BIN:-python3}"

# Hugging Face cache/model locations
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

export MODEL_ROOT="${MODEL_ROOT:-/workspace/models}"

# Tiny smoke-test model
export TINY_MODEL="${TINY_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"

# Small Qwen DraftTarget compatibility models
export QWEN_TARGET_SMALL_ID="${QWEN_TARGET_SMALL_ID:-Qwen/Qwen3-1.7B}"
export QWEN_DRAFT_SMALL_ID="${QWEN_DRAFT_SMALL_ID:-Qwen/Qwen3-0.6B}"
export QWEN_TARGET_SMALL_PATH="${QWEN_TARGET_SMALL_PATH:-$MODEL_ROOT/Qwen3-1.7B}"
export QWEN_DRAFT_SMALL_PATH="${QWEN_DRAFT_SMALL_PATH:-$MODEL_ROOT/Qwen3-0.6B}"

# Final model default
export FINAL_MODEL_ID="${FINAL_MODEL_ID:-nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4}"
export FINAL_MODEL_PATH="${FINAL_MODEL_PATH:-$MODEL_ROOT/Qwen3-Coder-480B-A35B-Instruct-NVFP4}"

# Server defaults
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

# TensorRT-LLM 1.1.0 requires numpy < 2.
# transformers 4.56.0 requires huggingface_hub >= 0.34, < 1.0.
export PIN_NUMPY="numpy>=1.26,<2"
export PIN_PANDAS="pandas>=2.3,<3"
export PIN_REQUESTS="requests>=2.32,<3"
export PIN_TQDM="tqdm>=4.66,<5"
export PIN_HF_HUB="huggingface_hub[cli]>=0.34,<1.0"

safe_install_benchmark_deps() {
  echo "Installing/repairing safe benchmark dependencies..."
  "$PYTHON_BIN" -m pip install \
    "$PIN_NUMPY" \
    "$PIN_PANDAS" \
    "$PIN_REQUESTS" \
    "$PIN_TQDM" \
    "$PIN_HF_HUB"
}

verify_core_versions() {
  "$PYTHON_BIN" - <<'PY'
import numpy
print("numpy:", numpy.__version__)

try:
    import huggingface_hub
    print("huggingface_hub:", huggingface_hub.__version__)
except Exception as e:
    print("huggingface_hub import failed:", repr(e))

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    print("transformers import failed:", repr(e))

try:
    import tensorrt_llm
    print("TensorRT-LLM import: OK")
except Exception as e:
    print("TensorRT-LLM import failed:", repr(e))
    raise
PY
}
