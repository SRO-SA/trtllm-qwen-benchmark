#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

# TinyLlama smoke benchmark. Uses the shared safe planner to avoid
# context_len + max_new_tokens exceeding the server's max_seq_len.
MODEL_NAME="${MODEL_NAME:-$TINY_MODEL}" \
PLAN_MODEL="${PLAN_MODEL:-$MODEL_NAME}" \
DECODE_MODE="${DECODE_MODE:-baseline}" \
QUANTIZATION="${QUANTIZATION:-smoke-test}" \
OUT="${OUT:-results/smoke_results.csv}" \
CONTEXTS="${CONTEXTS:-128 256 512 1024}" \
CONCURRENCIES="${CONCURRENCIES:-1 2 4}" \
SERVER_MAX_SEQ_LEN="${SERVER_MAX_SEQ_LEN:-$MAX_SEQ_LEN}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}" \
NUM_REQUESTS="${NUM_REQUESTS:-8}" \
SAFETY_TOKENS="${SAFETY_TOKENS:-64}" \
TP_SIZE="${TP_SIZE:-1}" \
KV_DTYPE="${KV_DTYPE:-bf16}" \
KV_MEMORY_FRACTION="${KV_MEMORY_FRACTION:-0.20}" \
TIMEOUT_S="${TIMEOUT_S:-300}" \
PLAN_OUT="${PLAN_OUT:-results/plan_smoke.json}" \
bash scripts/run_benchmark_grid.sh
