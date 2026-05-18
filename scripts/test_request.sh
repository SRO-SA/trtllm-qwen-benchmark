#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

MODEL_NAME="${MODEL_NAME:-$TINY_MODEL}"

curl -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Write a short Python function that adds two numbers.\"}
    ],
    \"max_tokens\": 64,
    \"temperature\": 0
  }"
