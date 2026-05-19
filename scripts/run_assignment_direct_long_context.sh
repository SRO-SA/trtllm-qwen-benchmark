#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

# Direct TensorRT-LLM Python API runner for long-context diagnostics.
# This bypasses trtllm-serve/OpenAI endpoints to determine whether 64K/128K
# failures are caused by the serving wrapper or by TensorRT-LLM/model backend.

MODEL_PATH="${MODEL_PATH:-$FINAL_MODEL_PATH}"
MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-NVFP4}"
TP_SIZE="${TP_SIZE:-4}"
QUANTIZATION="${QUANTIZATION:-nvfp4}"
OUT="${OUT:-results/assignment_tensorrt_llm_qwen480b_direct.csv}"
PROMPT_TOKEN_RESERVE="${PROMPT_TOKEN_RESERVE:-1024}"
KV_DTYPE="${KV_DTYPE:-fp8}"
BENCHMARK_HEARTBEAT_S="${BENCHMARK_HEARTBEAT_S:-60}"
CASE_TIMEOUT_S="${CASE_TIMEOUT_S:-5400}"
DIRECT_CONTEXTS="${DIRECT_CONTEXTS:-65536}"
# Set DIRECT_CONTEXTS="65536 131072" to try both.

mkdir -p results results/direct_logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
export PLAN_MODEL="$MODEL_PATH"
export MODEL_PATH
export PROMPT_TOKEN_RESERVE
export BENCHMARK_HEARTBEAT_S

echo "============================================================"
echo "Direct TensorRT-LLM long-context benchmark"
echo "Model path: $MODEL_PATH"
echo "Contexts: $DIRECT_CONTEXTS"
echo "Output: $OUT"
echo "============================================================"

# Direct API should not run while trtllm-serve is still holding the GPUs.
bash scripts/stop_trtllm_server.sh || true

if [[ "${RESET_RESULTS:-0}" == "1" ]]; then
  rm -f "$OUT"
fi

run_direct_context() {
  local CONTEXT="$1"
  local CAP MAX_NEW KV_FRAC NUM_REQ TIMEOUT DISABLE_CG
  DISABLE_CG="${DIRECT_DISABLE_CUDA_GRAPH:-0}"

  case "$CONTEXT" in
    32768)
      CAP="${DIRECT_32K_CAP:-49152}"; MAX_NEW="${DIRECT_32K_MAX_NEW_TOKENS:-128}"; KV_FRAC="${DIRECT_32K_KV:-0.25}"; NUM_REQ="${DIRECT_32K_NUM_REQUESTS:-1}"; TIMEOUT="${DIRECT_32K_TIMEOUT_S:-3600}" ;;
    65536)
      CAP="${DIRECT_64K_CAP:-81920}"; MAX_NEW="${DIRECT_64K_MAX_NEW_TOKENS:-64}"; KV_FRAC="${DIRECT_64K_KV:-0.18}"; NUM_REQ="${DIRECT_64K_NUM_REQUESTS:-1}"; TIMEOUT="${DIRECT_64K_TIMEOUT_S:-5400}" ;;
    131072)
      CAP="${DIRECT_128K_CAP:-147456}"; MAX_NEW="${DIRECT_128K_MAX_NEW_TOKENS:-32}"; KV_FRAC="${DIRECT_128K_KV:-0.12}"; NUM_REQ="${DIRECT_128K_NUM_REQUESTS:-1}"; TIMEOUT="${DIRECT_128K_TIMEOUT_S:-7200}" ;;
    *)
      echo "Unsupported DIRECT_CONTEXT=$CONTEXT. Supported: 32768, 65536, 131072" >&2
      return 2 ;;
  esac

  echo "============================================================"
  echo "Direct context: $CONTEXT"
  echo "Cap/max_seq_len: $CAP"
  echo "Max new tokens: $MAX_NEW"
  echo "Num requests: $NUM_REQ"
  echo "KV fraction: $KV_FRAC"
  echo "Timeout: $TIMEOUT"
  echo "Disable CUDA graph: $DISABLE_CG"
  echo "============================================================"

  CMD=(
    "$PYTHON_BIN" benchmark/benchmark_trtllm_direct.py
    --model-path "$MODEL_PATH"
    --model-name "$MODEL_NAME"
    --tokenizer-path "$TOKENIZER_PATH"
    --tp-size "$TP_SIZE"
    --backend pytorch
    --context-len "$CONTEXT"
    --max-new-tokens "$MAX_NEW"
    --num-requests "$NUM_REQ"
    --max-seq-len "$CAP"
    --max-input-len "$CAP"
    --max-num-tokens "$CAP"
    --max-batch-size 1
    --kv-memory-fraction "$KV_FRAC"
    --kv-dtype "$KV_DTYPE"
    --cuda-graph-batch-sizes "1"
    --timeout-s "$TIMEOUT"
    --heartbeat-s "$BENCHMARK_HEARTBEAT_S"
    --output "$OUT"
    --quantization "$QUANTIZATION"
    --decode-mode baseline_direct
  )

  if [[ "$DISABLE_CG" == "1" ]]; then
    CMD+=(--disable-cuda-graph)
  fi
  if [[ "${DIRECT_ENABLE_CHUNKED_PREFILL:-0}" == "1" ]]; then
    CMD+=(--enable-chunked-prefill)
  fi

  local LOG="results/direct_logs/direct_${CONTEXT}.log"
  local STATUS=0
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "$CASE_TIMEOUT_S" "${CMD[@]}" 2>&1 | tee "$LOG" || STATUS=${PIPESTATUS[0]}
  else
    "${CMD[@]}" 2>&1 | tee "$LOG" || STATUS=${PIPESTATUS[0]}
  fi

  if [[ "$STATUS" != "0" ]]; then
    echo "WARNING: direct context $CONTEXT failed with status $STATUS. Log: $LOG"
    return "$STATUS"
  fi
}

STATUS=0
for C in $DIRECT_CONTEXTS; do
  run_direct_context "$C" || STATUS=$?
  # Ensure workers release between contexts.
  sleep 5
  nvidia-smi || true
done

echo "============================================================"
echo "Direct benchmark complete. Output: $OUT"
echo "============================================================"
if [[ -f "$OUT" ]]; then
  cat "$OUT"
fi
exit "$STATUS"
