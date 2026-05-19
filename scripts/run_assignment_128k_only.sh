#!/bin/bash
set -euo pipefail

# Convenience wrapper for debugging only the 128K assignment stage.
# It starts from smaller KV fractions to avoid repeating clearly OOM-prone attempts.
# The main runner still validates the actual KV window before sending any request.

export RUN_ONLY_STAGE="${RUN_ONLY_STAGE:-long_128k}"
export RESET_RESULTS="${RESET_RESULTS:-0}"
export LONG128_KV_LADDER="${LONG128_KV_LADDER:-0.50 0.45 0.40}"
export LONG128_STARTUP_NO_PROGRESS_TIMEOUT_S="${LONG128_STARTUP_NO_PROGRESS_TIMEOUT_S:-600}"
export LONG128_FIRST_TOKEN_TIMEOUT_S="${LONG128_FIRST_TOKEN_TIMEOUT_S:-1200}"

bash scripts/run_assignment_baseline.sh
