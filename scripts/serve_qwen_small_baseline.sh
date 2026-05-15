#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

echo "Starting baseline TensorRT-LLM server"
echo "Target model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "Max seq len: ${MAX_SEQ_LEN}"

trtllm-serve serve \
  --backend pytorch \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  "${MODEL_NAME}"