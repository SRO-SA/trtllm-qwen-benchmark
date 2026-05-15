#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"
DECODE_MODE="${DECODE_MODE:-baseline}"

mkdir -p results

for CONTEXT in 256 512 1024; do
  for CONCURRENCY in 1 2; do
    echo "Running ${DECODE_MODE} benchmark: context=${CONTEXT}, concurrency=${CONCURRENCY}"

    python3 benchmark/benchmark_openai_stream.py \
      --host localhost \
      --port "${PORT}" \
      --model "${MODEL_NAME}" \
      --framework tensorrt-llm \
      --quantization "bf16-small-test" \
      --context-len "${CONTEXT}" \
      --concurrency "${CONCURRENCY}" \
      --num-requests 8 \
      --max-tokens 128 \
      --output "results/spec_smoke_${DECODE_MODE}.csv"
  done
done

echo "Finished ${DECODE_MODE} benchmark."
cat "results/spec_smoke_${DECODE_MODE}.csv"