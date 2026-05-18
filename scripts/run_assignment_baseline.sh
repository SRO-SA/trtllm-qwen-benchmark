#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

# Assignment-specific runner for TensorRT-LLM + Qwen3-Coder-480B NVFP4.
# It explicitly covers the required context windows and scenarios from the assignment:
#   - Short prompts up to 1k
#   - Medium prompts up to 8k
#   - Long-context evaluation at 32k, 64k, 128k
#   - Multi-user / concurrent workloads
#   - Sustained throughput style runs via configurable NUM_REQUESTS
#
# This script starts a fresh server for each context group with a large enough
# MAX_SEQ_LEN, then runs only assignment-relevant tests. It still uses the safe
# planner to prevent impossible cases, but the stage MAX_SEQ_LEN values are
# chosen so the required contexts should not be skipped because of sequence length.

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
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-3600}"
LOG_DIR="${LOG_DIR:-results/server_logs}"
METRICS_DIR="${METRICS_DIR:-results/metrics}"

mkdir -p results "$LOG_DIR" "$METRICS_DIR"

# Set RESET_RESULTS=1 if you want a clean CSV.
if [[ "${RESET_RESULTS:-0}" == "1" ]]; then
  rm -f "$OUT"
fi

run_stage() {
  local STAGE_NAME="$1"
  local SERVER_SEQ_LEN="$2"
  local CONTEXTS="$3"
  local CONCURRENCIES="$4"
  local STAGE_OUT_PLAN="results/plan_assignment_${STAGE_NAME}.json"
  local STAGE_LOG="${LOG_DIR}/server_${STAGE_NAME}.log"
  local STAGE_METRICS="${METRICS_DIR}/metrics_${STAGE_NAME}.json"

  echo "============================================================"
  echo "Assignment stage: ${STAGE_NAME}"
  echo "Model path: ${MODEL_PATH}"
  echo "Server max seq len: ${SERVER_SEQ_LEN}"
  echo "Contexts: ${CONTEXTS}"
  echo "Concurrencies: ${CONCURRENCIES}"
  echo "Output: ${OUT}"
  echo "============================================================"

  bash scripts/stop_trtllm_server.sh || true

  MAX_SEQ_LEN="$SERVER_SEQ_LEN" \
  TP_SIZE="$TP_SIZE" \
  MODEL_PATH="$MODEL_PATH" \
  CONFIG_PATH="$CONFIG_PATH" \
  nohup bash scripts/serve_final_baseline.sh > "$STAGE_LOG" 2>&1 &

  echo $! > server.pid
  echo "Started server PID $(cat server.pid). Log: ${STAGE_LOG}"

  TIMEOUT_S="$WAIT_TIMEOUT_S" bash scripts/wait_for_server.sh

  # Record the model id returned by /v1/models for debugging.
  curl -s "http://localhost:${PORT}/v1/models" > "${METRICS_DIR}/models_${STAGE_NAME}.json" || true

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
  PLAN_OUT="$STAGE_OUT_PLAN" \
  bash scripts/run_benchmark_grid.sh

  curl -m 20 --connect-timeout 3 -sS "http://localhost:${PORT}/metrics" \
    -o "$STAGE_METRICS" || true

  bash scripts/stop_trtllm_server.sh || true

  echo "Completed stage ${STAGE_NAME}."
}

# Required assignment contexts and scenarios.
# Server seq length = context + MAX_NEW_TOKENS + SAFETY_TOKENS, rounded upward.
# We use larger round values to keep TensorRT-LLM away from boundary conditions.

# Single-user short prompt up to 1k + multi-user concurrency for short prompts.
run_stage "short_1k_multiuser" 2048 "1024" "1 2 4 8"

# Medium prompts up to 8k + concurrent chat/code generation scenario.
run_stage "medium_8k_multiuser" 16384 "8192" "1 2 4"

# Long-context evaluation. Concurrency kept conservative because 480B + long context is expensive.
run_stage "long_32k" 65536 "32768" "1 2"
run_stage "long_64k" 131072 "65536" "1"

# Native 128k context evaluation. This requires very large KV cache capacity.
# If this fails due to memory, report it as the scalability/stability limit.
run_stage "long_128k" 262144 "131072" "1"

echo "============================================================"
echo "Assignment baseline benchmark complete."
echo "Results: ${OUT}"
echo "============================================================"
cat "$OUT"
