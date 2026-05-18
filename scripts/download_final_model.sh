#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_ID="${MODEL_ID:-$FINAL_MODEL_ID}"
MODEL_PATH="${MODEL_PATH:-$FINAL_MODEL_PATH}"

safe_install_benchmark_deps

mkdir -p "$MODEL_ROOT"
mkdir -p "$HF_HOME"

echo "Downloading final model:"
echo "  $MODEL_ID"
echo "to:"
echo "  $MODEL_PATH"

huggingface-cli download "$MODEL_ID" \
  --local-dir "$MODEL_PATH"

echo "Download finished. Checking files:"
du -sh "$MODEL_PATH"
ls -lh "$MODEL_PATH" | head
