#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

safe_install_benchmark_deps

mkdir -p "$MODEL_ROOT"
mkdir -p "$HF_HOME"

echo "Downloading small Qwen target model:"
echo "  $QWEN_TARGET_SMALL_ID -> $QWEN_TARGET_SMALL_PATH"

huggingface-cli download "$QWEN_TARGET_SMALL_ID" \
  --local-dir "$QWEN_TARGET_SMALL_PATH"

echo "Downloading small Qwen draft model:"
echo "  $QWEN_DRAFT_SMALL_ID -> $QWEN_DRAFT_SMALL_PATH"

huggingface-cli download "$QWEN_DRAFT_SMALL_ID" \
  --local-dir "$QWEN_DRAFT_SMALL_PATH"

echo "Verifying downloaded files..."
ls -lh "$QWEN_TARGET_SMALL_PATH" | head
ls -lh "$QWEN_DRAFT_SMALL_PATH" | head

echo "Checking for weight files..."
find "$QWEN_TARGET_SMALL_PATH" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" \) | head
find "$QWEN_DRAFT_SMALL_PATH" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" \) | head

echo "Small Qwen models downloaded successfully."
