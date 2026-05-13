#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PORT="${PORT:-8000}"

cat > extra-llm-api-config.yml <<'YAML'
enable_iter_perf_stats: true
YAML

echo "Serving model: ${MODEL_NAME}"
echo "Port: ${PORT}"

trtllm-serve serve \
  --backend pytorch \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --extra_llm_api_options ./extra-llm-api-config.yml \
  "${MODEL_NAME}"
