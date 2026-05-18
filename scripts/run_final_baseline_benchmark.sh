#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

# Final 480B baseline benchmark. This still uses the safe planner, so if
# CONTEXTS includes a value too close to SERVER_MAX_SEQ_LEN, that case is
# skipped rather than hanging the server.
MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-NVFP4}" \
PLAN_MODEL="${PLAN_MODEL:-$FINAL_MODEL_PATH}" \
DECODE_MODE="${DECODE_MODE:-baseline}" \
QUANTIZATION="${QUANTIZATION:-nvfp4}" \
OUT="${OUT:-results/final_baseline.csv}" \
CONTEXTS="${CONTEXTS:-1024 8192 32768}" \
CONCURRENCIES="${CONCURRENCIES:-1 2}" \
SERVER_MAX_SEQ_LEN="${SERVER_MAX_SEQ_LEN:-$MAX_SEQ_LEN}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}" \
NUM_REQUESTS="${NUM_REQUESTS:-8}" \
SAFETY_TOKENS="${SAFETY_TOKENS:-256}" \
TP_SIZE="${TP_SIZE:-4}" \
KV_DTYPE="${KV_DTYPE:-bf16}" \
KV_MEMORY_FRACTION="${KV_MEMORY_FRACTION:-0.70}" \
TIMEOUT_S="${TIMEOUT_S:-900}" \
PLAN_OUT="${PLAN_OUT:-results/plan_final_baseline.json}" \
bash scripts/run_benchmark_grid.sh
