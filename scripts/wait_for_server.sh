#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

TIMEOUT_S="${TIMEOUT_S:-1800}"
SLEEP_S="${SLEEP_S:-10}"
SERVER_LOG="${SERVER_LOG:-}"
PID_FILE="${PID_FILE:-server.pid}"
START_TIME=$(date +%s)

HEALTH_URL="http://localhost:${PORT}/health"
MODELS_URL="http://localhost:${PORT}/v1/models"

echo "Waiting for TensorRT-LLM server at ${HEALTH_URL}"
echo "Timeout: ${TIMEOUT_S}s"
if [[ -n "$SERVER_LOG" ]]; then
  echo "Server log: $SERVER_LOG"
fi

while true; do
  # If a PID file exists and the process already exited, fail early with diagnostics.
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" || true)"
    if [[ -n "${PID:-}" ]] && ! kill -0 "$PID" 2>/dev/null; then
      echo "ERROR: Server process PID ${PID} exited before health check became ready."
      SERVER_LOG="$SERVER_LOG" bash scripts/diagnose_server.sh || true
      exit 1
    fi
  fi

  HTTP_CODE=$(curl --max-time 5 --connect-timeout 2 -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || true)
  if [[ "$HTTP_CODE" == "200" ]]; then
    echo "Server health check passed."
    echo "Models:"
    curl --max-time 10 --connect-timeout 2 -sS "$MODELS_URL" || true
    echo
    exit 0
  fi

  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TIME))
  if (( ELAPSED >= TIMEOUT_S )); then
    echo "ERROR: Server did not become healthy within ${TIMEOUT_S}s. Last HTTP code: ${HTTP_CODE}"
    SERVER_LOG="$SERVER_LOG" bash scripts/diagnose_server.sh || true
    exit 1
  fi

  echo "Server not ready yet after ${ELAPSED}s. Last HTTP code: ${HTTP_CODE}. Sleeping ${SLEEP_S}s..."
  if [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]]; then
    echo "Last 8 log lines:"
    tail -n 8 "$SERVER_LOG" || true
  fi
  sleep "$SLEEP_S"
done
