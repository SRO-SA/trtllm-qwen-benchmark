#!/bin/bash
set -euo pipefail

# Additional workload scenarios required by the assignment table:
#   - concurrent chat sessions
#   - parallel code generation
#   - sustained throughput testing
# These append to the same OUT CSV used by the baseline run.

source "$(dirname "$0")/common_env.sh"

MODEL_PATH="${MODEL_PATH:-$FINAL_MODEL_PATH}"
MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-NVFP4}"
OUT="${OUT:-results/assignment_tensorrt_llm_qwen480b_full.csv}"
PROMPT_FILE="${PROMPT_FILE:-data/assignment_prompts.jsonl}"
TP_SIZE="${TP_SIZE:-4}"
QUANTIZATION="${QUANTIZATION:-nvfp4}"

run_one_extra() {
  local NAME="$1"
  local WORKLOAD="$2"
  local PROFILE="$3"
  local STAGE="$4"
  local CONCS="$5"
  local REQS="$6"
  local MAXTOK="$7"
  local DURATION="${8:-0}"

  echo "============================================================"
  echo "Extra assignment scenario: $NAME"
  echo "Workload: $WORKLOAD"
  echo "Prompt profile: $PROFILE"
  echo "Stage: $STAGE"
  echo "Concurrencies: $CONCS"
  echo "Requests: $REQS"
  echo "Max tokens: $MAXTOK"
  echo "Duration target: $DURATION"
  echo "============================================================"

  local SHORT_CONCS="1 2 4 8"
  local MED_CONCS="1 2 4"
  if [[ "$STAGE" == "short_1k_multiuser" ]]; then
    SHORT_CONCS="$CONCS"
  elif [[ "$STAGE" == "medium_8k_multiuser" ]]; then
    MED_CONCS="$CONCS"
  fi

  SCENARIO_NAME="$NAME" \
  WORKLOAD_TYPE="$WORKLOAD" \
  PROMPT_PROFILE="$PROFILE" \
  PROMPT_FILE="$PROMPT_FILE" \
  DURATION_S="$DURATION" \
  OUT="$OUT" \
  RESET_RESULTS=0 \
  RUN_ONLY_STAGE="$STAGE" \
  SHORT_CONCURRENCIES="$SHORT_CONCS" \
  MED_CONCURRENCIES="$MED_CONCS" \
  SHORT_NUM_REQUESTS="$REQS" \
  MED_NUM_REQUESTS="$REQS" \
  SHORT_MAX_NEW_TOKENS="$MAXTOK" \
  MED_MAX_NEW_TOKENS="$MAXTOK" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  PLAN_MODEL="$MODEL_PATH" \
  TOKENIZER_PATH="$MODEL_PATH" \
  TP_SIZE="$TP_SIZE" \
  QUANTIZATION="$QUANTIZATION" \
  PYTHONPATH="$PWD:${PYTHONPATH:-}" \
  bash scripts/run_assignment_baseline.sh
}

# Concurrent chat sessions: tests interactive/chat-like prompts under increasing concurrency.
run_one_extra "concurrent_chat_sessions_1k" "chat" "chat_session" "short_1k_multiuser" "1 2 4 8 16" "16" "128" "0"
run_one_extra "concurrent_chat_sessions_8k" "chat" "chat_session" "medium_8k_multiuser" "1 2 4 8" "16" "128" "0"

# Parallel code generation: code-generation prompt profile under concurrent requests.
run_one_extra "parallel_code_generation_1k" "code_generation" "parallel_code_generation" "short_1k_multiuser" "1 2 4 8 16" "16" "256" "0"
run_one_extra "parallel_code_generation_8k" "code_generation" "parallel_code_generation" "medium_8k_multiuser" "1 2 4 8" "16" "256" "0"

# Sustained throughput: many requests at fixed high concurrency.
# We use request-count based sustained testing for reproducibility; total_time_s and
# aggregate_tps in the CSV capture sustained throughput over the full run.
run_one_extra "sustained_throughput_1k_c16" "sustained" "sustained_throughput" "short_1k_multiuser" "16" "128" "128" "0"
run_one_extra "sustained_throughput_8k_c8" "sustained" "sustained_throughput" "medium_8k_multiuser" "8" "64" "128" "0"

echo "Extra assignment scenarios complete. Results: $OUT"
