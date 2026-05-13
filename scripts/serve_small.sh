#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PORT="${PORT:-8000}"

echo "Serving model: ${MODEL_NAME}"
echo "Port: ${PORT}"

trtllm-serve "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}"