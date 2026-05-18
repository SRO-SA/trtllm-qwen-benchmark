#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

# Assignment-specific runner for TensorRT-LLM + Qwen3-Coder-480B NVFP4.
# It explicitly covers the required assignment scenarios:
#   - Short prompts up to 1k
#   - Medium prompts up to 8k
#   - Long-context evaluation at 32k, 64k, 128k
#   - Multi-user / concurrent workloads
#   - Sustained throughput style runs via configurable NUM_REQUESTS
#
# This runner is intentionally watchdog-oriented:
#   - starts a fresh server for each context group
#   - waits for /health with timeout
#   - wraps each benchmark case with a wall-clock timeout
#   - appends failure rows to the CSV when a server/case fails
#   - saves logs, metrics, and diagnostics for failure analysis

MODEL_PATH="${MODEL_PATH:-$FINAL_MODEL_PATH}"
MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-NVFP4}"
CONFIG_PATH="${CONFIG_PATH:-configs/final_assignment_baseline.yaml}"
TP_SIZE="${TP_SIZE:-4}"
QUANTIZATION="${QUANTIZATION:-nvfp4}"
DECODE_MODE="${DECODE_MODE:-baseline}"
OUT="${OUT:-results/assignment_tensorrt_llm_qwen480b_baseline.csv}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SAFETY_TOKENS="${SAFETY_TOKENS:-512}"
NUM_REQUESTS="${NUM_REQUESTS:-8}"
KV_DTYPE="${KV_DTYPE:-bf16}"
KV_MEMORY_FRACTION="${KV_MEMORY_FRACTION:-0.70}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
CASE_TIMEOUT_S="${CASE_TIMEOUT_S:-$((TIMEOUT_S + 300))}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-3600}"
LOG_DIR="${LOG_DIR:-results/server_logs}"
METRICS_DIR="${METRICS_DIR:-results/metrics}"

mkdir -p results "$LOG_DIR" "$METRICS_DIR"

if [[ "${RESET_RESULTS:-0}" == "1" ]]; then
  rm -f "$OUT"
fi

append_stage_failure_rows() {
  local CONTEXTS="$1"
  local CONCURRENCIES="$2"
  local REASON="$3"

  for CONTEXT in $CONTEXTS; do
    for CONCURRENCY in $CONCURRENCIES; do
      "$PYTHON_BIN" scripts/append_failure_row.py \
        --output "$OUT" \
        --model "$MODEL_NAME" \
        --quantization "$QUANTIZATION" \
        --decode-mode "$DECODE_MODE" \
        --context-len "$CONTEXT" \
        --concurrency "$CONCURRENCY" \
        --max-tokens "$MAX_NEW_TOKENS" \
        --num-requests "$NUM_REQUESTS" \
        --error-message "$REASON"
    done
  done
}

run_stage() {
  local STAGE_NAME="$1"
  local SERVER_SEQ_LEN="$2"
  local CONTEXTS="$3"
  local CONCURRENCIES="$4"
  local STAGE_OUT_PLAN="results/plan_assignment_${STAGE_NAME}.json"
  local STAGE_LOG="${LOG_DIR}/server_${STAGE_NAME}.log"
  local STAGE_METRICS="${METRICS_DIR}/metrics_${STAGE_NAME}.json"
  local STAGE_STATUS=0

  echo "============================================================"
  echo "Assignment stage: ${STAGE_NAME}"
  echo "Model path: ${MODEL_PATH}"
  echo "Server max seq len: ${SERVER_SEQ_LEN}"
  echo "Contexts: ${CONTEXTS}"
  echo "Concurrencies: ${CONCURRENCIES}"
  echo "Output: ${OUT}"
  echo "Server log: ${STAGE_LOG}"
  echo "============================================================"

  bash scripts/stop_trtllm_server.sh || true

  MAX_SEQ_LEN="$SERVER_SEQ_LEN" \
  TP_SIZE="$TP_SIZE" \
  MODEL_PATH="$MODEL_PATH" \
  CONFIG_PATH="$CONFIG_PATH" \
  nohup bash scripts/serve_final_baseline.sh > "$STAGE_LOG" 2>&1 &

  echo $! > server.pid
  echo "Started server PID $(cat server.pid). Log: ${STAGE_LOG}"

  if ! TIMEOUT_S="$WAIT_TIMEOUT_S" SERVER_LOG="$STAGE_LOG" bash scripts/wait_for_server.sh; then
    echo "WARNING: Stage ${STAGE_NAME} server did not become healthy. Recording failure rows and continuing."
    SERVER_LOG="$STAGE_LOG" bash scripts/diagnose_server.sh || true
    append_stage_failure_rows "$CONTEXTS" "$CONCURRENCIES" "server_start_failed_or_timeout_${WAIT_TIMEOUT_S}s"
    bash scripts/stop_trtllm_server.sh || true
    echo "Completed failed stage ${STAGE_NAME}."
    return 0
  fi

  curl --max-time 10 --connect-timeout 2 -sS "http://localhost:${PORT}/v1/models" \
    > "${METRICS_DIR}/models_${STAGE_NAME}.json" || true

  MODEL_NAME="$MODEL_NAME" \
  PLAN_MODEL="$MODEL_PATH" \
  DECODE_MODE="$DECODE_MODE" \
  QUANTIZATION="$QUANTIZATION" \
  OUT="$OUT" \
  CONTEXTS="$CONTEXTS" \
  CONCURRENCIES="$CONCURRENCIES" \
  SERVER_MAX_SEQ_LEN="$SERVER_SEQ_LEN" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  NUM_REQUESTS="$NUM_REQUESTS" \
  SAFETY_TOKENS="$SAFETY_TOKENS" \
  TP_SIZE="$TP_SIZE" \
  KV_DTYPE="$KV_DTYPE" \
  KV_MEMORY_FRACTION="$KV_MEMORY_FRACTION" \
  TIMEOUT_S="$TIMEOUT_S" \
  CASE_TIMEOUT_S="$CASE_TIMEOUT_S" \
  SERVER_LOG="$STAGE_LOG" \
  PLAN_OUT="$STAGE_OUT_PLAN" \
  STOP_ON_CASE_FAILURE=1 \
  bash scripts/run_benchmark_grid.sh || STAGE_STATUS=$?

  if [[ "$STAGE_STATUS" != "0" ]]; then
    echo "WARNING: Stage ${STAGE_NAME} had benchmark failure/timeout status ${STAGE_STATUS}. Diagnostics captured; continuing to next assignment stage."
    SERVER_LOG="$STAGE_LOG" bash scripts/diagnose_server.sh || true
  fi

  curl -m 20 --connect-timeout 3 -sS "http://localhost:${PORT}/metrics" \
    -o "$STAGE_METRICS" || true

  bash scripts/stop_trtllm_server.sh || true

  echo "Completed stage ${STAGE_NAME}."
}

# Required assignment contexts and scenarios.
# Server seq length is intentionally larger than context + MAX_NEW_TOKENS + SAFETY_TOKENS.

run_stage "short_1k_multiuser" 2048 "1024" "1 2 4 8"
run_stage "medium_8k_multiuser" 16384 "8192" "1 2 4"
run_stage "long_32k" 65536 "32768" "1 2"
run_stage "long_64k" 131072 "65536" "1"
run_stage "long_128k" 262144 "131072" "1"

echo "============================================================"
echo "Assignment baseline benchmark complete."
echo "Results: ${OUT}"
echo "============================================================"
if [[ -f "$OUT" ]]; then
  cat "$OUT"
else
  echo "No result file generated. Check logs under ${LOG_DIR} and diagnostics under results/diagnostics."
fi
