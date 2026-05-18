#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

BASE_FILE="${BASE_FILE:-results/spec_smoke_baseline.csv}"
SPEC_FILE="${SPEC_FILE:-results/spec_smoke_draft_target.csv}"
OUT_FILE="${OUT_FILE:-results/spec_comparison_summary.csv}"

if [[ ! -f "$BASE_FILE" ]]; then
  echo "Missing $BASE_FILE"
  echo "Run baseline benchmark first, e.g.:"
  echo "  DECODE_MODE=baseline MODEL_NAME=Qwen3-1.7B bash scripts/run_spec_smoke_benchmark.sh"
  exit 1
fi

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "Missing $SPEC_FILE"
  echo "Start the DraftTarget server first, then run:"
  echo "  DECODE_MODE=draft_target MODEL_NAME=Qwen3-1.7B bash scripts/run_spec_smoke_benchmark.sh"
  exit 1
fi

"$PYTHON_BIN" - <<PY
import pandas as pd
from pathlib import Path

base_path = Path("$BASE_FILE")
spec_path = Path("$SPEC_FILE")
out = Path("$OUT_FILE")

base = pd.read_csv(base_path)
spec = pd.read_csv(spec_path)

if "decode_mode" not in base.columns:
    base["decode_mode"] = "baseline"
if "decode_mode" not in spec.columns:
    spec["decode_mode"] = "draft_target"

df = pd.concat([base, spec], ignore_index=True)

cols = [
    "decode_mode",
    "context_len",
    "concurrency",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tps_mean",
    "tps_p50",
    "tps_p99",
    "aggregate_tps",
    "vram_idle_gb",
    "vram_load_gb",
    "kv_cache_growth_gb",
    "gpu_util_mean_after",
    "runtime_stability",
    "error_count",
]
cols = [c for c in cols if c in df.columns]
summary = df[cols]
print(summary.to_string(index=False))

out.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(out, index=False)
print(f"\nSaved: {out}")
PY
