#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_NAME="${MODEL_NAME:-$QWEN_TARGET_SMALL_PATH}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

echo "Starting baseline TensorRT-LLM server"
echo "Target model: $MODEL_NAME"
echo "Port: $PORT"
echo "Max seq len: $MAX_SEQ_LEN"

trtllm-serve serve \
  --backend pytorch \
  --host "$HOST" \
  --port "$PORT" \
  --max_seq_len "$MAX_SEQ_LEN" \
  "$MODEL_NAME"
