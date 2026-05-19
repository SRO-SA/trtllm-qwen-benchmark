# TensorRT-LLM Qwen3-Coder-480B Benchmark

This repository contains scripts for deploying and benchmarking **TensorRT-LLM** on the **Qwen3-Coder-480B-A35B-Instruct-NVFP4** 4-bit model. It was built for the inference-framework benchmarking assignment comparing deployment behavior across vLLM, SGLang, and TensorRT-LLM.

The current repo focuses on the TensorRT-LLM side of the assignment.

## Assignment Target

| Item | Configuration |
|---|---|
| Framework | TensorRT-LLM |
| Model | Qwen3-Coder-480B-A35B-Instruct |
| Practical checkpoint | `nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4` |
| Quantization | 4-bit NVFP4 weights, FP8 KV cache from checkpoint config |
| GPU | RTX PRO 6000 Blackwell |
| Context windows | 1K, 8K, 32K, 64K, 128K |
| Metrics | TTFT, TPS mean/P99, aggregate TPS, max concurrency, VRAM idle/load, KV-cache growth, GPU utilization, runtime stability |
| Workloads | single-user, multi-user, concurrent chat, parallel code generation, long-context, sustained throughput |

Important practical note: 128K did not reliably fit on 4 GPUs in our real tests. The successful full assignment run used **8× RTX PRO 6000 Blackwell** with `TP_SIZE=8` and `LONG128_KV_LADDER="0.65 0.60 0.55"`.

## Important Environment Rule

Do **not** run unconstrained package upgrades inside the TensorRT-LLM container.

Avoid:

```bash
pip install -U ...
```

TensorRT-LLM 1.1.0 expects compatible dependency versions. In particular, keep:

```text
numpy < 2
huggingface_hub < 1.0
```

Use the setup checker instead:

```bash
bash scripts/setup_check.sh
```

This uses `scripts/common_env.sh` and installs/repairs only the safe benchmark helper packages.

## Repository Structure

```text
trtllm-qwen-benchmark/
├── benchmark/
│   ├── __init__.py
│   ├── benchmark_openai_stream.py
│   └── plan_safe_tests.py
├── configs/
│   ├── final_assignment_baseline.yaml
│   ├── final_baseline_qwen480b.yaml
│   ├── final_draft_target_qwen480b.yaml
│   ├── qwen_small_baseline.yaml
│   └── spec_draft_target_qwen_small.yaml
├── data/
│   ├── assignment_prompts.jsonl
│   └── assignment_scenarios.json
├── scripts/
│   ├── append_failure_row.py
│   ├── check_kv_window.py
│   ├── check_server_limits.py
│   ├── clean_assignment_outputs.sh
│   ├── common_env.sh
│   ├── compare_spec_results.sh
│   ├── diagnose_server.sh
│   ├── download_final_model.sh
│   ├── download_qwen_small_models.sh
│   ├── run_assignment_128k_only.sh
│   ├── run_assignment_baseline.sh
│   ├── run_assignment_extra_scenarios.sh
│   ├── run_assignment_full_suite.sh
│   ├── run_assignment_summary.sh
│   ├── run_benchmark_grid.sh
│   ├── run_final_baseline_benchmark.sh
│   ├── run_final_draft_target_benchmark.sh
│   ├── run_smoke_benchmark.sh
│   ├── run_spec_smoke_benchmark.sh
│   ├── serve_final_baseline.sh
│   ├── serve_final_draft_target.sh
│   ├── serve_qwen_small_baseline.sh
│   ├── serve_qwen_small_draft_target.sh
│   ├── serve_small.sh
│   ├── setup_check.sh
│   ├── setup_github_ssh.sh
│   ├── stop_trtllm_server.sh
│   ├── summarize_assignment_results.py
│   ├── test_prompt_lengths.py
│   ├── test_request.sh
│   └── wait_for_server.sh
└── results/
```

## Recommended Full Assignment Run on 8 GPUs

Use this after cloning the repo and downloading the model.

```bash
cd /workspace/trtllm-qwen-benchmark
bash scripts/stop_trtllm_server.sh || true
chmod +x scripts/*.sh
touch benchmark/__init__.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TP_SIZE=8 \
LONG128_KV_LADDER="0.65 0.60 0.55" \
MODEL_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
MODEL_NAME=Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
PLAN_MODEL=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
TOKENIZER_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
QUANTIZATION=nvfp4 \
PROMPT_TOKEN_RESERVE=1024 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PWD \
bash scripts/run_assignment_full_suite.sh
```

Main outputs:

```text
results/assignment_tensorrt_llm_qwen480b_full.csv
results/assignment_summary.csv
results/assignment_coverage_summary.csv
results/max_concurrency_summary.csv
results/server_logs/
results/metrics/
results/diagnostics/
```

## Validate the Full Assignment Results

After the run finishes:

```bash
cat results/assignment_coverage_summary.csv
cat results/max_concurrency_summary.csv
```

Check pass/fail rows:

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("results/assignment_tensorrt_llm_qwen480b_full.csv")
print("Rows:", len(df))
print("Contexts:", sorted(df["context_len"].dropna().unique()))
print("Runtime stability:")
print(df["runtime_stability"].value_counts(dropna=False))

bad = df[(df["runtime_stability"] != "pass") | (df["error_count"].fillna(0) > 0) | (df["failed_requests"].fillna(0) > 0)]
if len(bad) == 0:
    print("All rows passed.")
else:
    print(bad[["scenario_name", "workload_type", "context_len", "concurrency", "runtime_stability", "error_count", "error_messages"]].to_string(index=False))
PY
```

Check metric completeness:

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("results/assignment_tensorrt_llm_qwen480b_full.csv")
cols = ["ttft_mean_ms", "ttft_p99_ms", "tps_mean", "tps_p99", "aggregate_tps", "vram_idle_gb", "vram_load_gb", "kv_cache_growth_gb", "gpu_util_mean_after"]
print(df[cols].isna().sum())
PY
```

Known caveat: 64K and 128K use `/v1/completions` in non-streaming mode, so true first-token timing may be unavailable for those rows. Throughput, total latency, VRAM, KV-cache growth, GPU utilization, and runtime stability are still recorded.

## Download the Final Model

```bash
bash scripts/download_final_model.sh
```

By default this downloads:

```text
nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4
```

to:

```text
/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4
```

You can override with:

```bash
MODEL_ID=... MODEL_PATH=... bash scripts/download_final_model.sh
```

## 128K-Only Rerun

Use this when only the 128K long-context case needs to be retested:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TP_SIZE=8 \
LONG128_KV_LADDER="0.65 0.60 0.55" \
MODEL_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
MODEL_NAME=Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
PLAN_MODEL=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
TOKENIZER_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
QUANTIZATION=nvfp4 \
PYTHONPATH=$PWD \
bash scripts/run_assignment_128k_only.sh
```

The 128K stage uses a KV-cache retry ladder. It only sends a request if the TensorRT-LLM log reports a KV-cache window large enough for:

```text
context_len + max_new_tokens + safety_tokens
```

## Script Reference

### Core benchmark clients

| File | Purpose |
|---|---|
| `benchmark/benchmark_openai_stream.py` | Main benchmark client. Sends OpenAI-compatible requests to `trtllm-serve`, builds tokenizer-aware prompts, supports chat and completion endpoints, records TTFT/TPS/VRAM/KV growth/GPU utilization/runtime stability, and appends rows to CSV. |
| `benchmark/plan_safe_tests.py` | Pre-run planner. Checks `(context_len, concurrency)` pairs against `server_max_seq_len`, `server_max_num_tokens`, safety buffer, and approximate KV-cache memory. Saves a JSON plan and prevents obviously unsafe requests. |
| `benchmark/__init__.py` | Makes `benchmark/` importable so helper scripts such as `test_prompt_lengths.py` can import prompt-generation utilities. |

### Full assignment runners

| File | Purpose |
|---|---|
| `scripts/run_assignment_full_suite.sh` | Main end-to-end assignment runner. Runs baseline context scaling, long-context tests, concurrent chat, parallel code generation, sustained throughput, and summary generation. Use this for final results. |
| `scripts/run_assignment_baseline.sh` | Assignment baseline runner for short/medium/long context scaling. Starts fresh servers per context group and handles 32K/64K/128K special settings. |
| `scripts/run_assignment_extra_scenarios.sh` | Adds assignment-specific workload scenarios: concurrent chat sessions, parallel code generation, and sustained throughput. Appends to the full assignment CSV. |
| `scripts/run_assignment_128k_only.sh` | Convenience wrapper for debugging or rerunning only the 128K long-context stage. Supports `LONG128_KV_LADDER`. |
| `scripts/run_assignment_summary.sh` | Wrapper around `summarize_assignment_results.py`; generates `assignment_summary.csv`, `assignment_coverage_summary.csv`, and `max_concurrency_summary.csv`. |
| `scripts/summarize_assignment_results.py` | Python summarizer that checks coverage, summarizes pass/fail rows, and computes max passing concurrency per workload/context. |

### Generic benchmark runners

| File | Purpose |
|---|---|
| `scripts/run_benchmark_grid.sh` | Shared benchmark-grid runner. Calls the planner, runs safe cases, enforces case timeouts, saves plan JSON, and appends failure rows when needed. Used by smoke, speculative, final, and assignment runners. |
| `scripts/run_smoke_benchmark.sh` | Runs a small TinyLlama smoke benchmark to verify the server and client path. |
| `scripts/run_spec_smoke_benchmark.sh` | Runs small Qwen baseline or DraftTarget speculative decoding smoke tests. |
| `scripts/run_final_baseline_benchmark.sh` | Generic final-model baseline benchmark runner. Useful for custom context/concurrency experiments outside the full assignment suite. |
| `scripts/run_final_draft_target_benchmark.sh` | Generic final-model DraftTarget speculative-decoding benchmark runner. Use only after baseline is stable. |
| `scripts/compare_spec_results.sh` | Compares small baseline and DraftTarget smoke-test CSVs and writes `results/spec_comparison_summary.csv`. |

### Server launchers

| File | Purpose |
|---|---|
| `scripts/serve_small.sh` | Starts a TinyLlama TensorRT-LLM server for quick smoke testing. |
| `scripts/serve_qwen_small_baseline.sh` | Starts a small Qwen baseline server for DraftTarget compatibility tests. |
| `scripts/serve_qwen_small_draft_target.sh` | Starts a small Qwen DraftTarget speculative decoding server using `configs/spec_draft_target_qwen_small.yaml`. |
| `scripts/serve_final_baseline.sh` | Starts the final Qwen3-Coder-480B NVFP4 baseline server. Supports `TP_SIZE`, `MAX_SEQ_LEN`, `MAX_NUM_TOKENS`, `MAX_INPUT_LEN`, `MAX_BATCH_SIZE`, and `CONFIG_PATH`. |
| `scripts/serve_final_draft_target.sh` | Starts the final-model DraftTarget speculative decoding server. Experimental; not used for the final baseline assignment table. |

### Diagnostics, safety, and cleanup

| File | Purpose |
|---|---|
| `scripts/wait_for_server.sh` | Waits for `/health`; detects early process exit, timeout, and no-progress startup hangs. Calls diagnostics and cleanup when needed. |
| `scripts/stop_trtllm_server.sh` | Stops `trtllm-serve`, orphaned `mpi4py.futures.server` workers, model-related Python workers, and processes holding port 8000. Use before every clean run. |
| `scripts/diagnose_server.sh` | Collects a diagnostic report: `nvidia-smi`, relevant processes, port-bind status, `/health`, `/v1/models`, `/metrics`, and server-log tail. |
| `scripts/clean_assignment_outputs.sh` | Backs up previous `results/` and cleans generated assignment configs, logs, diagnostics, metrics, and assignment CSV files. |
| `scripts/check_server_limits.py` | Parses server logs for `max_seq_len`, `max_num_tokens`, and `max_input_len`; fails if the server is not configured for the required stage. |
| `scripts/check_kv_window.py` | Parses TensorRT-LLM logs for `window size=...`; verifies the actual KV-cache window is large enough for a requested long-context stage. |
| `scripts/append_failure_row.py` | Appends a structured failure row to a results CSV when a server or benchmark case fails before the normal benchmark client can write a row. |
| `scripts/test_prompt_lengths.py` | Uses the tokenizer and benchmark prompt generator to confirm generated prompt lengths match requested contexts. |
| `scripts/test_request.sh` | Sends one simple chat-completion request to a running server. Useful for quick API sanity checks. |

### Setup and download helpers

| File | Purpose |
|---|---|
| `scripts/common_env.sh` | Shared environment and defaults: model IDs/paths, ports, Python executable, safe helper functions, Hugging Face cache paths, and dependency constraints. Almost every script sources this file. |
| `scripts/setup_check.sh` | Checks GPU, Python, `trtllm-serve`, TensorRT-LLM import, CUDA availability, and compatible package versions. Run this first on a new instance. |
| `scripts/download_final_model.sh` | Downloads the NVFP4 Qwen3-Coder-480B model to `/workspace/models`. |
| `scripts/download_qwen_small_models.sh` | Downloads small Qwen target/draft models for DraftTarget compatibility smoke tests. |
| `scripts/setup_github_ssh.sh` | Sets up GitHub SSH access on Vast. Supports raw private key paste, base64 private key, or private key file. |

## Config and Data Reference

| File | Purpose |
|---|---|
| `data/assignment_prompts.jsonl` | Prompt templates for assignment scenarios: baseline, concurrent chat, parallel code generation, sustained throughput, and long-context code review. |
| `data/assignment_scenarios.json` | Scenario metadata/configuration for assignment runs. |
| `configs/final_assignment_baseline.yaml` | Base TensorRT-LLM YAML for assignment stages. Runtime configs are generated under `results/runtime_configs/`. |
| `configs/final_baseline_qwen480b.yaml` | Baseline YAML for final-model generic serving. |
| `configs/final_draft_target_qwen480b.yaml` | Experimental DraftTarget YAML for the final model. |
| `configs/qwen_small_baseline.yaml` | Small-Qwen baseline serving config. |
| `configs/spec_draft_target_qwen_small.yaml` | Small-Qwen DraftTarget serving config. |

## Results Files

| File/Directory | Meaning |
|---|---|
| `results/assignment_tensorrt_llm_qwen480b_full.csv` | Main full-suite assignment results. |
| `results/assignment_summary.csv` | Summary of key rows and metrics. |
| `results/assignment_coverage_summary.csv` | Boolean coverage checklist for assignment requirements. |
| `results/max_concurrency_summary.csv` | Largest passing concurrency per workload/context group. |
| `results/server_logs/` | Server logs for each stage. Keep these for debugging and final reporting. |
| `results/metrics/` | Saved `/metrics` snapshots when available. |
| `results/diagnostics/` | Diagnostic reports from failed, hung, or timed-out stages. |
| `results/runtime_configs/` | Generated per-stage TensorRT-LLM YAML configs. |

## Small Smoke Test Workflow

Use this only to verify the environment before the 480B run:

```bash
bash scripts/setup_check.sh

nohup bash scripts/serve_small.sh > server.log 2>&1 &
echo $! > server.pid
bash scripts/wait_for_server.sh
bash scripts/run_smoke_benchmark.sh
bash scripts/stop_trtllm_server.sh
```

## Small DraftTarget Compatibility Test

```bash
bash scripts/download_qwen_small_models.sh

nohup bash scripts/serve_qwen_small_baseline.sh > server_baseline.log 2>&1 &
echo $! > server.pid
bash scripts/wait_for_server.sh
DECODE_MODE=baseline MODEL_NAME=Qwen3-1.7B bash scripts/run_spec_smoke_benchmark.sh
bash scripts/stop_trtllm_server.sh

nohup bash scripts/serve_qwen_small_draft_target.sh > server_draft_target.log 2>&1 &
echo $! > server.pid
bash scripts/wait_for_server.sh
DECODE_MODE=draft_target MODEL_NAME=Qwen3-1.7B bash scripts/run_spec_smoke_benchmark.sh
bash scripts/stop_trtllm_server.sh

bash scripts/compare_spec_results.sh
```

## Useful Commands

Check GPUs:

```bash
nvidia-smi
watch -n 5 nvidia-smi
```

Check TensorRT-LLM CLI:

```bash
which trtllm-serve
trtllm-serve serve --help | grep -E "tp_size|max_seq_len|max_num_tokens|max_batch_size|extra_llm_api_options"
```

Check server health:

```bash
curl -i http://localhost:8000/health
curl http://localhost:8000/v1/models
curl -sS http://localhost:8000/metrics | head
```

Stop all TensorRT-LLM workers:

```bash
bash scripts/stop_trtllm_server.sh
```

Package final results:

```bash
mkdir -p final_assignment_results
cp results/assignment_tensorrt_llm_qwen480b_full.csv final_assignment_results/
cp results/assignment_summary.csv final_assignment_results/
cp results/assignment_coverage_summary.csv final_assignment_results/
cp results/max_concurrency_summary.csv final_assignment_results/
cp -r results/server_logs final_assignment_results/
cp -r results/metrics final_assignment_results/ 2>/dev/null || true
cp -r results/diagnostics final_assignment_results/ 2>/dev/null || true

tar -czf final_assignment_results_$(date +%Y%m%d_%H%M%S).tar.gz final_assignment_results
ls -lh final_assignment_results_*.tar.gz
```

## Notes Learned From Real Runs

- 64K and 128K are sent through `/v1/completions` instead of `/v1/chat/completions` because TensorRT-LLM 1.1.0 chat serving can mishandle very long prompts.
- Long-context completion rows may not have true TTFT if the endpoint returns a non-streaming response.
- 128K needs enough KV-cache window. The script validates this from the actual server log before sending the request.
- On 4 GPUs, 128K can hit startup OOM or no-progress startup hangs. On 8 GPUs, 128K completed successfully with `LONG128_KV_LADDER="0.65 0.60 0.55"`.
- Always run `scripts/stop_trtllm_server.sh` before starting a new server; TensorRT-LLM can leave orphaned MPI workers after failed long-context attempts.
