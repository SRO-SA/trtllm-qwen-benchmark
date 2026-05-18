#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_PATH="${MODEL_PATH:-$FINAL_MODEL_PATH}"
TP_SIZE="${TP_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
# TensorRT-LLM PyTorch backend keeps max_num_tokens separate from max_seq_len.
# If this stays at the default 8192, 32k/64k/128k prompts are rejected even when
# max_seq_len is large.
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-$MAX_SEQ_LEN}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-$MAX_SEQ_LEN}"
CONFIG_PATH="${CONFIG_PATH:-configs/final_baseline_qwen480b.yaml}"

EXTRA_ARGS=(
  --backend pytorch
  --host "$HOST"
  --port "$PORT"
  --tp_size "$TP_SIZE"
  --max_seq_len "$MAX_SEQ_LEN"
)

append_trtllm_option_if_supported EXTRA_ARGS --max_num_tokens "$MAX_NUM_TOKENS"
append_trtllm_option_if_supported EXTRA_ARGS --max_input_len "$MAX_INPUT_LEN"

# TensorRT-LLM 1.1.0 supports --extra_llm_api_options. Newer versions may also support --config.
if trtllm_supports_option --extra_llm_api_options; then
  EXTRA_ARGS+=(--extra_llm_api_options "$CONFIG_PATH")
elif trtllm_supports_option --config; then
  EXTRA_ARGS+=(--config "$CONFIG_PATH")
else
  echo "WARNING: no config/extra_llm_api_options flag found; launching without config."
fi

echo "Starting final TensorRT-LLM baseline server"
echo "Model path: $MODEL_PATH"
echo "TP size: $TP_SIZE"
echo "Max seq len: $MAX_SEQ_LEN"
echo "Max num tokens: $MAX_NUM_TOKENS"
echo "Max input len: $MAX_INPUT_LEN"
echo "Config: $CONFIG_PATH"
echo "Extra args: ${EXTRA_ARGS[*]}"

trtllm-serve serve "${EXTRA_ARGS[@]}" "$MODEL_PATH"
