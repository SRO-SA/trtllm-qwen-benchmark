#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results/assignment_tensorrt_llm_qwen480b_baseline.csv")
parser.add_argument("--output", default="results/assignment_summary.csv")
args = parser.parse_args()

inp = Path(args.input)
if not inp.exists():
    raise FileNotFoundError(inp)

df = pd.read_csv(inp)

# Add scenario labels based on assignment context windows.
def scenario(row):
    c = int(row["context_len"])
    conc = int(row["concurrency"])
    if c <= 1024 and conc == 1:
        return "single_user_short_1k"
    if c <= 8192 and conc == 1:
        return "single_user_medium_8k"
    if c in {32768, 65536, 131072}:
        return "long_context_evaluation"
    if conc > 1:
        return "multi_user_concurrency"
    return "other"

df["assignment_scenario"] = df.apply(scenario, axis=1)

cols = [
    "assignment_scenario",
    "framework",
    "model",
    "quantization",
    "decode_mode",
    "gpu_type",
    "num_gpus",
    "context_len",
    "concurrency",
    "max_new_tokens",
    "num_requests",
    "successful_requests",
    "failed_requests",
    "ttft_mean_ms",
    "ttft_p99_ms",
    "tps_mean",
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
summary.to_csv(args.output, index=False)
print(summary.to_string(index=False))
print(f"\nSaved: {args.output}")
