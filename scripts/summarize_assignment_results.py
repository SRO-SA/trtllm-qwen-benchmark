#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results/assignment_tensorrt_llm_qwen480b_baseline.csv")
parser.add_argument("--output", default="results/assignment_summary.csv")
parser.add_argument("--coverage-output", default="results/assignment_coverage_summary.csv")
args = parser.parse_args()

inp = Path(args.input)
if not inp.exists():
    raise FileNotFoundError(inp)

df = pd.read_csv(inp)

if "scenario_name" not in df.columns:
    df["scenario_name"] = "unspecified"
if "workload_type" not in df.columns:
    df["workload_type"] = "unspecified"
if "prompt_profile" not in df.columns:
    df["prompt_profile"] = "unknown"
if "api_mode" not in df.columns:
    df["api_mode"] = "unknown"

# Assignment-oriented labels.
def assignment_scenario(row):
    existing = str(row.get("scenario_name", ""))
    if existing and existing != "unspecified":
        return existing
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

df["assignment_scenario"] = df.apply(assignment_scenario, axis=1)

cols = [
    "assignment_scenario",
    "scenario_name",
    "workload_type",
    "prompt_profile",
    "api_mode",
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
    "error_messages",
]
cols = [c for c in cols if c in df.columns]
summary = df[cols]
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(args.output, index=False)
print(summary.to_string(index=False))
print(f"\nSaved: {args.output}")

# Coverage table for assignment requirements.
def has_pass(mask):
    sub = df[mask]
    return bool(((sub.get("runtime_stability") == "pass") & (sub.get("successful_requests", 0) > 0)).any())

coverage_items = []
coverage_items.append({"requirement":"4-bit quantization", "covered": bool(df["quantization"].astype(str).str.contains("nvfp4|4", case=False, regex=True).any()), "evidence":"quantization column"})
coverage_items.append({"requirement":"RTX 6000 PRO Blackwell", "covered": bool(df["gpu_type"].astype(str).str.contains("Blackwell", case=False).any()), "evidence":"gpu_type column"})
coverage_items.append({"requirement":"single-user short prompts up to 1K", "covered": has_pass((df.context_len <= 1024) & (df.concurrency == 1)), "evidence":"context_len<=1024, concurrency=1"})
coverage_items.append({"requirement":"single-user medium prompts up to 8K", "covered": has_pass((df.context_len <= 8192) & (df.context_len > 1024) & (df.concurrency == 1)), "evidence":"context_len=8192, concurrency=1"})
coverage_items.append({"requirement":"multi-user inference", "covered": has_pass(df.concurrency > 1), "evidence":"concurrency>1"})
coverage_items.append({"requirement":"concurrent chat sessions", "covered": has_pass(df.workload_type.astype(str).str.contains("chat", case=False, na=False) & (df.concurrency > 1)), "evidence":"workload_type=chat, concurrency>1"})
coverage_items.append({"requirement":"parallel code generation", "covered": has_pass(df.workload_type.astype(str).str.contains("code", case=False, na=False) & (df.concurrency > 1)), "evidence":"workload_type=code_generation, concurrency>1"})
coverage_items.append({"requirement":"sustained throughput testing", "covered": has_pass(df.workload_type.astype(str).str.contains("sustained", case=False, na=False)), "evidence":"workload_type=sustained"})
for ctx in [32768, 65536, 131072]:
    coverage_items.append({"requirement":f"long-context {ctx//1024}K", "covered": has_pass(df.context_len == ctx), "evidence":f"context_len={ctx}"})
coverage_items.append({"requirement":"TTFT metric", "covered": "ttft_mean_ms" in df.columns, "evidence":"ttft_mean_ms column"})
coverage_items.append({"requirement":"TPS mean/P99", "covered": all(c in df.columns for c in ["tps_mean", "tps_p99"]), "evidence":"tps_mean,tps_p99 columns"})
coverage_items.append({"requirement":"VRAM idle/load", "covered": all(c in df.columns for c in ["vram_idle_gb", "vram_load_gb"]), "evidence":"vram_idle_gb,vram_load_gb columns"})
coverage_items.append({"requirement":"KV-cache growth", "covered": "kv_cache_growth_gb" in df.columns, "evidence":"kv_cache_growth_gb column"})
coverage_items.append({"requirement":"runtime stability", "covered": "runtime_stability" in df.columns, "evidence":"runtime_stability column"})

cov = pd.DataFrame(coverage_items)
cov.to_csv(args.coverage_output, index=False)
print("\nCoverage summary:")
print(cov.to_string(index=False))
print(f"\nSaved: {args.coverage_output}")

# Max passing concurrency per context/workload.
pass_df = df[(df.get("runtime_stability") == "pass") & (df.get("successful_requests", 0) > 0)].copy()
if not pass_df.empty:
    max_conc = pass_df.groupby(["workload_type", "context_len"], dropna=False)["concurrency"].max().reset_index()
    max_conc = max_conc.rename(columns={"concurrency":"max_passing_concurrency"})
    max_conc.to_csv("results/max_concurrency_summary.csv", index=False)
    print("\nMax passing concurrency:")
    print(max_conc.to_string(index=False))
    print("\nSaved: results/max_concurrency_summary.csv")
