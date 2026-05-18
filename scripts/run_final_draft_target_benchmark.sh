#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-NVFP4}"
OUT="${OUT:-results/final_draft_target.csv}"

mkdir -p results

# Speculative decoding should be tested after baseline is stable.
# Keep this smaller initially.
CONTEXTS="${CONTEXTS:-1024 8192}"
CONCURRENCIES="${CONCURRENCIES:-1}"

for CONTEXT in $CONTEXTS; do
  for CONCURRENCY in $CONCURRENCIES; do
    echo "Running final DraftTarget: context=$CONTEXT concurrency=$CONCURRENCY"

    "$PYTHON_BIN" benchmark/benchmark_openai_stream.py \
      --host localhost \
      --port "$PORT" \
      --model "$MODEL_NAME" \
      --framework tensorrt-llm \
      --quantization nvfp4-draft-target \
      --context-len "$CONTEXT" \
      --concurrency "$CONCURRENCY" \
      --num-requests "${NUM_REQUESTS:-8}" \
      --max-tokens "${MAX_TOKENS:-256}" \
      --decode-mode draft_target \
      --output "$OUT"
  done
done

echo "Final DraftTarget benchmark finished."
cat "$OUT"
