#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

TIMEOUT_S="${TIMEOUT_S:-1800}"
SLEEP_S="${SLEEP_S:-10}"
START_TIME=$(date +%s)

HEALTH_URL="http://localhost:${PORT}/health"
MODELS_URL="http://localhost:${PORT}/v1/models"

echo "Waiting for TensorRT-LLM server at ${HEALTH_URL}"
echo "Timeout: ${TIMEOUT_S}s"

while true; do
  if curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" | grep -q "200"; then
    echo "Server health check passed."
    echo "Models:"
    curl -s "$MODELS_URL" || true
    echo
    exit 0
  fi

  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TIME))
  if (( ELAPSED >= TIMEOUT_S )); then
    echo "ERROR: Server did not become healthy within ${TIMEOUT_S}s."
    exit 1
  fi

  echo "Server not ready yet after ${ELAPSED}s. Sleeping ${SLEEP_S}s..."
  sleep "$SLEEP_S"
done
