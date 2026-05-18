#!/bin/bash
set -euo pipefail

PORT="${PORT:-8000}"

echo "Checking TensorRT-LLM server processes..."

PIDS="$(pgrep -f "trtllm-serve" || true)"

if [[ -z "$PIDS" ]]; then
  echo "No trtllm-serve process found."
else
  echo "Found trtllm-serve process(es):"
  echo "$PIDS"

  echo "Trying graceful kill..."
  pkill -TERM -f "trtllm-serve" || true
  sleep 5

  PIDS_LEFT="$(pgrep -f "trtllm-serve" || true)"
  if [[ -n "$PIDS_LEFT" ]]; then
    echo "Some trtllm-serve processes are still alive:"
    echo "$PIDS_LEFT"
    echo "Force killing..."
    pkill -9 -f "trtllm-serve" || true
    sleep 3
  fi
fi

rm -f server.pid

echo "Checking whether port ${PORT} is free..."

python3 - <<PY
import socket
import sys

port = int("${PORT}")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.bind(("0.0.0.0", port))
    s.close()
    print(f"Port {port} is free.")
    sys.exit(0)
except OSError as e:
    print(f"Port {port} is still in use: {e}")
    sys.exit(1)
PY

echo "Done. It is safe to start a new TensorRT-LLM server."
