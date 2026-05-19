#!/bin/bash
set -euo pipefail

# Full assignment suite for TensorRT-LLM + Qwen3-Coder-480B NVFP4.
# Produces one CSV containing baseline context scaling, long-context tests,
# concurrent chat, parallel code generation, and sustained throughput scenarios.

source "$(dirname "$0")/common_env.sh"

OUT="${OUT:-results/assignment_tensorrt_llm_qwen480b_full.csv}"
MODEL_PATH="${MODEL_PATH:-$FINAL_MODEL_PATH}"
MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-NVFP4}"
PROMPT_FILE="${PROMPT_FILE:-data/assignment_prompts.jsonl}"

mkdir -p results results_backup

echo "Backing up existing results, if any..."
if [[ -d results ]]; then
  tar -czf "results_backup/results_before_full_suite_$(date +%Y%m%d_%H%M%S).tar.gz" results/ || true
fi

bash scripts/stop_trtllm_server.sh || true
bash scripts/clean_assignment_outputs.sh || true
rm -f "$OUT" results/assignment_summary.csv results/assignment_coverage_summary.csv results/max_concurrency_summary.csv

echo "============================================================"
echo "Running baseline/context-scaling assignment scenarios"
echo "============================================================"
SCENARIO_NAME="" \
WORKLOAD_TYPE="baseline" \
PROMPT_PROFILE="synthetic_code_context" \
PROMPT_FILE="$PROMPT_FILE" \
OUT="$OUT" \
RESET_RESULTS=1 \
MODEL_PATH="$MODEL_PATH" \
MODEL_NAME="$MODEL_NAME" \
PLAN_MODEL="$MODEL_PATH" \
TOKENIZER_PATH="$MODEL_PATH" \
PYTHONPATH="$PWD:${PYTHONPATH:-}" \
bash scripts/run_assignment_baseline.sh

echo "============================================================"
echo "Running extra workload-specific assignment scenarios"
echo "============================================================"
OUT="$OUT" \
MODEL_PATH="$MODEL_PATH" \
MODEL_NAME="$MODEL_NAME" \
PLAN_MODEL="$MODEL_PATH" \
TOKENIZER_PATH="$MODEL_PATH" \
PROMPT_FILE="$PROMPT_FILE" \
PYTHONPATH="$PWD:${PYTHONPATH:-}" \
bash scripts/run_assignment_extra_scenarios.sh

echo "============================================================"
echo "Summarizing assignment coverage and metrics"
echo "============================================================"
INPUT="$OUT" OUTPUT="results/assignment_summary.csv" bash scripts/run_assignment_summary.sh

echo "============================================================"
echo "Full assignment suite complete"
echo "Main CSV: $OUT"
echo "Summary CSV: results/assignment_summary.csv"
echo "Coverage CSV: results/assignment_coverage_summary.csv"
echo "Max concurrency CSV: results/max_concurrency_summary.csv"
echo "============================================================"
