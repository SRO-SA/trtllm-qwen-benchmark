#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_PATH="${MODEL_PATH:-$FINAL_MODEL_PATH}"
TP_SIZE="${TP_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
CONFIG_PATH="${CONFIG_PATH:-configs/final_draft_target_qwen480b.yaml}"

echo "Starting final TensorRT-LLM DraftTarget server"
echo "Model path: $MODEL_PATH"
echo "TP size: $TP_SIZE"
echo "Max seq len: $MAX_SEQ_LEN"
echo "Config: $CONFIG_PATH"

trtllm-serve serve \
  --backend pytorch \
  --host "$HOST" \
  --port "$PORT" \
  --tp_size "$TP_SIZE" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --extra_llm_api_options "$CONFIG_PATH" \
  "$MODEL_PATH"
