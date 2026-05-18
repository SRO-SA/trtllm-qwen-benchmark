#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

# Final 480B DraftTarget benchmark. Keep conservative defaults; run after the
# baseline is stable.
MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-NVFP4}" \
PLAN_MODEL="${PLAN_MODEL:-$FINAL_MODEL_PATH}" \
DECODE_MODE="${DECODE_MODE:-draft_target}" \
QUANTIZATION="${QUANTIZATION:-nvfp4-draft-target}" \
OUT="${OUT:-results/final_draft_target.csv}" \
CONTEXTS="${CONTEXTS:-1024 8192}" \
CONCURRENCIES="${CONCURRENCIES:-1}" \
SERVER_MAX_SEQ_LEN="${SERVER_MAX_SEQ_LEN:-$MAX_SEQ_LEN}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}" \
NUM_REQUESTS="${NUM_REQUESTS:-8}" \
SAFETY_TOKENS="${SAFETY_TOKENS:-256}" \
TP_SIZE="${TP_SIZE:-4}" \
KV_DTYPE="${KV_DTYPE:-bf16}" \
KV_MEMORY_FRACTION="${KV_MEMORY_FRACTION:-0.50}" \
TIMEOUT_S="${TIMEOUT_S:-900}" \
PLAN_OUT="${PLAN_OUT:-results/plan_final_draft_target.json}" \
bash scripts/run_benchmark_grid.sh
