#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_NAME="${MODEL_NAME:-$QWEN_TARGET_SMALL_PATH}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen_small_baseline.yaml}"

echo "Starting baseline TensorRT-LLM server"
echo "Target model: $MODEL_NAME"
echo "Config: $CONFIG_PATH"
echo "Port: $PORT"
echo "Max seq len: $MAX_SEQ_LEN"

trtllm-serve serve \
  --backend pytorch \
  --host "$HOST" \
  --port "$PORT" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --extra_llm_api_options "$CONFIG_PATH" \
  "$MODEL_NAME"
