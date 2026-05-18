#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_NAME="${MODEL_NAME:-$TINY_MODEL}"

cat > extra-llm-api-config.yml <<'YAML'
enable_iter_perf_stats: true
YAML

echo "Serving model: $MODEL_NAME"
echo "Port: $PORT"

trtllm-serve serve \
  --backend pytorch \
  --host "$HOST" \
  --port "$PORT" \
  --extra_llm_api_options ./extra-llm-api-config.yml \
  "$MODEL_NAME"
