#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
DECODE_MODE="${DECODE_MODE:-baseline}"
OUT="${OUT:-results/spec_smoke_${DECODE_MODE}.csv}"

mkdir -p results

# DraftTarget server often runs with max_seq_len=1024 in cheap tests,
# so default contexts avoid 1024 + generated tokens exceeding the limit.
CONTEXTS="${CONTEXTS:-256 512}"
CONCURRENCIES="${CONCURRENCIES:-1 2}"

for CONTEXT in $CONTEXTS; do
  for CONCURRENCY in $CONCURRENCIES; do
    echo "Running $DECODE_MODE benchmark: context=$CONTEXT, concurrency=$CONCURRENCY"

    "$PYTHON_BIN" benchmark/benchmark_openai_stream.py \
      --host localhost \
      --port "$PORT" \
      --model "$MODEL_NAME" \
      --framework tensorrt-llm \
      --quantization "bf16-small-${DECODE_MODE}" \
      --context-len "$CONTEXT" \
      --concurrency "$CONCURRENCY" \
      --num-requests "${NUM_REQUESTS:-8}" \
      --max-tokens "${MAX_TOKENS:-128}" \
      --decode-mode "$DECODE_MODE" \
      --output "$OUT"
  done
done

echo "Finished $DECODE_MODE benchmark."
cat "$OUT"
