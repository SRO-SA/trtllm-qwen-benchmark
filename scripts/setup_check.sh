#!/bin/bash
set -euo pipefail

echo "=============================="
echo "GPU check"
echo "=============================="
nvidia-smi || true

echo "=============================="
echo "Python check"
echo "=============================="
python3 --version || true

echo "=============================="
echo "TensorRT-LLM command check"
echo "=============================="
which trtllm-serve || true
trtllm-serve --help | head -80 || true

echo "=============================="
echo "Python imports"
echo "=============================="
python3 - <<'PY'
import sys
print("Python:", sys.version)

try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
except Exception as e:
    print("Torch check failed:", repr(e))

try:
    import tensorrt_llm
    print("TensorRT-LLM import: OK")
except Exception as e:
    print("TensorRT-LLM import failed:", repr(e))
PY

echo "=============================="
echo "Install benchmark helper packages"
echo "=============================="
python3 -m pip install -U requests pandas numpy tqdm

echo "Setup check finished."