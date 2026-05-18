#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
INPUT="${INPUT:-results/assignment_tensorrt_llm_qwen480b_baseline.csv}"
OUTPUT="${OUTPUT:-results/assignment_summary.csv}"
"$PYTHON_BIN" scripts/summarize_assignment_results.py --input "$INPUT" --output "$OUTPUT"
