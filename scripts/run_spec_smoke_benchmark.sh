#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

# Small Qwen baseline/DraftTarget smoke benchmark. Uses local model path for
# planning when available, but uses MODEL_NAME for requests because the server
# may expose a shorter ID such as "Qwen3-1.7B".
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}" \
PLAN_MODEL="${PLAN_MODEL:-$QWEN_TARGET_SMALL_PATH}" \
DECODE_MODE="${DECODE_MODE:-baseline}" \
QUANTIZATION="${QUANTIZATION:-bf16-small-${DECODE_MODE}}" \
OUT="${OUT:-results/spec_smoke_${DECODE_MODE}.csv}" \
CONTEXTS="${CONTEXTS:-128 256 512 1024}" \
CONCURRENCIES="${CONCURRENCIES:-1 2}" \
SERVER_MAX_SEQ_LEN="${SERVER_MAX_SEQ_LEN:-$MAX_SEQ_LEN}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}" \
NUM_REQUESTS="${NUM_REQUESTS:-8}" \
SAFETY_TOKENS="${SAFETY_TOKENS:-64}" \
TP_SIZE="${TP_SIZE:-1}" \
KV_DTYPE="${KV_DTYPE:-bf16}" \
KV_MEMORY_FRACTION="${KV_MEMORY_FRACTION:-0.20}" \
TIMEOUT_S="${TIMEOUT_S:-300}" \
PLAN_OUT="${PLAN_OUT:-results/plan_spec_smoke_${DECODE_MODE}.json}" \
bash scripts/run_benchmark_grid.sh
