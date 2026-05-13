#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PORT="${PORT:-8000}"

mkdir -p results

for CONTEXT in 1024 2048; do
  for CONCURRENCY in 1 2 4; do
    echo "Running smoke benchmark: context=${CONTEXT}, concurrency=${CONCURRENCY}"

    python3 benchmark/benchmark_openai_stream.py \
      --host localhost \
      --port "${PORT}" \
      --model "${MODEL_NAME}" \
      --framework tensorrt-llm \
      --quantization smoke-test \
      --context-len "${CONTEXT}" \
      --concurrency "${CONCURRENCY}" \
      --num-requests 8 \
      --max-tokens 64 \
      --output results/smoke_results.csv
  done
done

echo "Smoke benchmark finished."
echo "Results:"
cat results/smoke_results.csv