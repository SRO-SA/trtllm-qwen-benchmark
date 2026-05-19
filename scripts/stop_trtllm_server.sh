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

# Kill any remaining process that owns/binds PORT, even if its cmdline no longer
# contains trtllm-serve. This handles uvicorn/python workers left after Ctrl+C.
python3 - <<PY
import os, signal, pathlib, time
port = int("${PORT}")
hex_port = f"{port:04X}"

def find_port_pids():
    inodes = set()
    for path in ["/proc/net/tcp", "/proc/net/tcp6"]:
        try:
            lines = open(path).read().splitlines()[1:]
        except FileNotFoundError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local = parts[1]
            inode = parts[9]
            local_port = local.split(":")[-1]
            if local_port.upper() == hex_port:
                inodes.add(inode)
    pids = set()
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        fd_dir = proc / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    link = os.readlink(fd)
                except OSError:
                    continue
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    pids.add(int(proc.name))
        except OSError:
            continue
    return sorted(pids)

pids = find_port_pids()
if pids:
    print(f"Processes still using port {port}: {pids}")
    for pid in pids:
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except Exception:
            cmd = "<unknown>"
        print(f"Killing PID {pid}: {cmd}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pid in find_port_pids():
        try:
            print(f"Force killing PID {pid}")
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY

rm -f server.pid

echo "Checking whether port ${PORT} is free..."
python3 - <<PY
import socket, sys, time
port = int("${PORT}")
for attempt in range(10):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        s.close()
        print(f"Port {port} is free.")
        print("Done. It is safe to start a new TensorRT-LLM server.")
        sys.exit(0)
    except OSError as e:
        s.close()
        if attempt == 9:
            print(f"Port {port} is still in use: {e}")
            sys.exit(1)
        time.sleep(1)
PY
