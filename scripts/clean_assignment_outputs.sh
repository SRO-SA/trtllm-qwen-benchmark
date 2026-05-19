#!/bin/bash
set -euo pipefail

mkdir -p results_backup
if [[ -d results ]]; then
  tar -czf "results_backup/results_before_clean_$(date +%Y%m%d_%H%M%S).tar.gz" results/ || true
fi

rm -rf results/runtime_configs results/server_logs results/diagnostics results/metrics
rm -f results/assignment_tensorrt_llm_qwen480b_baseline.csv results/assignment_summary.csv
rm -f results/plan_assignment_*.json
mkdir -p results/runtime_configs results/server_logs results/diagnostics results/metrics

echo "Cleaned generated assignment configs/logs/metrics/diagnostics and backed up previous results under results_backup/."
