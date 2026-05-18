#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

echo "=============================="
echo "GPU check"
echo "=============================="
nvidia-smi || true

echo "=============================="
echo "Python check"
echo "=============================="
"$PYTHON_BIN" --version

echo "=============================="
echo "TensorRT-LLM command check"
echo "=============================="
which trtllm-serve || true
trtllm-serve --help | head -80 || true

echo "=============================="
echo "Repair/install safe dependency versions"
echo "=============================="
safe_install_benchmark_deps

echo "=============================="
echo "Python imports and versions"
echo "=============================="
"$PYTHON_BIN" - <<'PY'
import sys
print("Python:", sys.version)

import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

import tensorrt_llm
print("TensorRT-LLM import: OK")
PY

verify_core_versions

echo "Setup check finished."
