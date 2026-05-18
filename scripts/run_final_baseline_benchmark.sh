#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-NVFP4}"
OUT="${OUT:-results/final_baseline.csv}"

mkdir -p results

# Start conservative. Add 64k/128k only after 1k/8k/32k are stable.
CONTEXTS="${CONTEXTS:-1024 8192 32768}"
CONCURRENCIES="${CONCURRENCIES:-1 2}"

for CONTEXT in $CONTEXTS; do
  for CONCURRENCY in $CONCURRENCIES; do
    echo "Running final baseline: context=$CONTEXT concurrency=$CONCURRENCY"

    "$PYTHON_BIN" benchmark/benchmark_openai_stream.py \
      --host localhost \
      --port "$PORT" \
      --model "$MODEL_NAME" \
      --framework tensorrt-llm \
      --quantization nvfp4 \
      --context-len "$CONTEXT" \
      --concurrency "$CONCURRENCY" \
      --num-requests "${NUM_REQUESTS:-8}" \
      --max-tokens "${MAX_TOKENS:-256}" \
      --decode-mode baseline \
      --output "$OUT"
  done
done

echo "Final baseline benchmark finished."
cat "$OUT"
