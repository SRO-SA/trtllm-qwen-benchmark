#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
CONFIG_PATH="${CONFIG_PATH:-configs/spec_draft_target_qwen_small.yaml}"

echo "Starting DraftTarget speculative TensorRT-LLM server"
echo "Target model: ${MODEL_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Port: ${PORT}"
echo "Max seq len: ${MAX_SEQ_LEN}"

trtllm-serve serve \
  --backend pytorch \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --extra_llm_api_options "${CONFIG_PATH}" \
  "${MODEL_NAME}"