#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME must be set to the model id returned by /v1/models or accepted by the server}"
PLAN_MODEL="${PLAN_MODEL:-$MODEL_NAME}"
DECODE_MODE="${DECODE_MODE:-baseline}"
QUANTIZATION="${QUANTIZATION:-unknown}"
OUT="${OUT:-results/benchmark_${DECODE_MODE}.csv}"

CONTEXTS="${CONTEXTS:-128 256}"
CONCURRENCIES="${CONCURRENCIES:-1}"
SERVER_MAX_SEQ_LEN="${SERVER_MAX_SEQ_LEN:-$MAX_SEQ_LEN}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
NUM_REQUESTS="${NUM_REQUESTS:-8}"
SAFETY_TOKENS="${SAFETY_TOKENS:-64}"
TP_SIZE="${TP_SIZE:-1}"
KV_DTYPE="${KV_DTYPE:-bf16}"
KV_MEMORY_FRACTION="${KV_MEMORY_FRACTION:-0.20}"
TIMEOUT_S="${TIMEOUT_S:-300}"
PLAN_OUT="${PLAN_OUT:-results/plan_${DECODE_MODE}.json}"

mkdir -p results

echo "======================================"
echo "TensorRT-LLM benchmark grid"
echo "======================================"
echo "MODEL_NAME=${MODEL_NAME}"
echo "PLAN_MODEL=${PLAN_MODEL}"
echo "DECODE_MODE=${DECODE_MODE}"
echo "QUANTIZATION=${QUANTIZATION}"
echo "CONTEXTS=${CONTEXTS}"
echo "CONCURRENCIES=${CONCURRENCIES}"
echo "SERVER_MAX_SEQ_LEN=${SERVER_MAX_SEQ_LEN}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "NUM_REQUESTS=${NUM_REQUESTS}"
echo "SAFETY_TOKENS=${SAFETY_TOKENS}"
echo "TP_SIZE=${TP_SIZE}"
echo "KV_DTYPE=${KV_DTYPE}"
echo "KV_MEMORY_FRACTION=${KV_MEMORY_FRACTION}"
echo "TIMEOUT_S=${TIMEOUT_S}"
echo "OUT=${OUT}"
echo "PLAN_OUT=${PLAN_OUT}"
echo "======================================"

echo "Planning safe cases..."
"$PYTHON_BIN" benchmark/plan_safe_tests.py \
  --model "$PLAN_MODEL" \
  --tp-size "$TP_SIZE" \
  --server-max-seq-len "$SERVER_MAX_SEQ_LEN" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --kv-dtype "$KV_DTYPE" \
  --safety-tokens "$SAFETY_TOKENS" \
  --kv-memory-fraction "$KV_MEMORY_FRACTION" \
  --contexts "$CONTEXTS" \
  --concurrency "$CONCURRENCIES" \
  --output "$PLAN_OUT" \
  --format summary

echo "Running planned safe cases..."

RUN_COUNT=0
while IFS=$'\t' read -r CONTEXT CONCURRENCY EST_TOTAL EST_KV; do
  [[ -z "${CONTEXT:-}" ]] && continue
  RUN_COUNT=$((RUN_COUNT + 1))
  echo "Running case #${RUN_COUNT}: context=${CONTEXT}, concurrency=${CONCURRENCY}, estimated_total_tokens=${EST_TOTAL}, estimated_kv_gb=${EST_KV}"

  "$PYTHON_BIN" benchmark/benchmark_openai_stream.py \
    --host localhost \
    --port "$PORT" \
    --model "$MODEL_NAME" \
    --framework tensorrt-llm \
    --quantization "$QUANTIZATION" \
    --decode-mode "$DECODE_MODE" \
    --context-len "$CONTEXT" \
    --concurrency "$CONCURRENCY" \
    --num-requests "$NUM_REQUESTS" \
    --max-tokens "$MAX_NEW_TOKENS" \
    --timeout-s "$TIMEOUT_S" \
    --output "$OUT"
done < <("$PYTHON_BIN" benchmark/plan_safe_tests.py \
  --model "$PLAN_MODEL" \
  --tp-size "$TP_SIZE" \
  --server-max-seq-len "$SERVER_MAX_SEQ_LEN" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --kv-dtype "$KV_DTYPE" \
  --safety-tokens "$SAFETY_TOKENS" \
  --kv-memory-fraction "$KV_MEMORY_FRACTION" \
  --contexts "$CONTEXTS" \
  --concurrency "$CONCURRENCIES" \
  --format tsv)

if (( RUN_COUNT == 0 )); then
  echo "No safe benchmark cases were selected. Increase SERVER_MAX_SEQ_LEN or reduce MAX_NEW_TOKENS/CONTEXTS."
  exit 1
fi

echo "Finished ${DECODE_MODE} benchmark. Results:"
cat "$OUT"
