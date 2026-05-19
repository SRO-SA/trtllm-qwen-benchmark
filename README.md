# TensorRT-LLM Qwen Benchmark

This repository contains scripts for testing and benchmarking TensorRT-LLM for Qwen3-Coder-480B 4-bit inference.

The target assignment setup is:

- Framework: TensorRT-LLM
- Main model: Qwen3-Coder-480B
- Quantization: 4-bit offline/pre-made quantization
- Practical TensorRT-LLM checkpoint: `nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4`
- Target GPU: 4× RTX PRO 6000 Blackwell
- Context windows: 1k, 8k, 32k, 64k, 128k
- Metrics: TTFT, TPS mean/P99, aggregate TPS, max concurrency, VRAM idle/load, KV-cache growth, GPU utilization, runtime stability

## Important Environment Rule

Do **not** run unconstrained package upgrades inside the TensorRT-LLM container.

Avoid:

```bash
pip install -U ...
```

TensorRT-LLM 1.1.0 requires `numpy < 2`, and the included Transformers stack requires `huggingface_hub < 1.0`.

Use:

```bash
bash scripts/setup_check.sh
```

This repairs/install safe dependency versions through `scripts/common_env.sh`.

## Repository Structure

```text
trtllm-qwen-benchmark/
├── benchmark/
│   └── benchmark_openai_stream.py
├── configs/
│   ├── final_baseline_qwen480b.yaml
│   ├── final_draft_target_qwen480b.yaml
│   └── spec_draft_target_qwen_small.yaml
├── scripts/
│   ├── common_env.sh
│   ├── setup_check.sh
│   ├── download_qwen_small_models.sh
│   ├── download_final_model.sh
│   ├── serve_small.sh
│   ├── serve_qwen_small_baseline.sh
│   ├── serve_qwen_small_draft_target.sh
│   ├── serve_final_baseline.sh
│   ├── serve_final_draft_target.sh
│   ├── run_smoke_benchmark.sh
│   ├── run_spec_smoke_benchmark.sh
│   ├── run_final_baseline_benchmark.sh
│   ├── run_final_draft_target_benchmark.sh
│   ├── compare_spec_results.sh
│   ├── setup_github_ssh.sh
│   └── test_request.sh
└── results/
```

## 1. Basic Environment Check

```bash
bash scripts/setup_check.sh
```

Expected signs:

```text
trtllm-serve found
CUDA available: True
TensorRT-LLM import: OK
numpy: 1.26.x
huggingface_hub: 0.x
```

## 2. TinyLlama Smoke Test

Start server in background:

```bash
nohup bash scripts/serve_small.sh > server.log 2>&1 &
echo $! > server.pid
tail -f server.log
```

Stop watching log with `Ctrl+C`.

Check health:

```bash
curl -i http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Run benchmark:

```bash
bash scripts/run_smoke_benchmark.sh
```

Stop server:

```bash
kill $(cat server.pid)
# or
pkill -f trtllm-serve
```

## 3. Small DraftTarget Compatibility Test

This verifies TensorRT-LLM DraftTarget speculative decoding before using the huge model.

Download small local models:

```bash
bash scripts/download_qwen_small_models.sh
```

Start baseline:

```bash
nohup bash scripts/serve_qwen_small_baseline.sh > server_baseline.log 2>&1 &
echo $! > server.pid
tail -f server_baseline.log
```

Run baseline benchmark after `/health` is OK:

```bash
DECODE_MODE=baseline MODEL_NAME=Qwen3-1.7B bash scripts/run_spec_smoke_benchmark.sh
kill $(cat server.pid)
```

Start DraftTarget:

```bash
nohup bash scripts/serve_qwen_small_draft_target.sh > server_draft_target.log 2>&1 &
echo $! > server.pid
tail -f server_draft_target.log
```

Run DraftTarget benchmark after `/health` is OK:

```bash
DECODE_MODE=draft_target MODEL_NAME=Qwen3-1.7B bash scripts/run_spec_smoke_benchmark.sh
kill $(cat server.pid)
```

Compare:

```bash
bash scripts/compare_spec_results.sh
```

Notes:

- TensorRT-LLM 1.1.0 uses `speculative_model_dir`, not `speculative_model`.
- The draft model must be a **local directory**, not a Hugging Face repo ID.
- The config is in `configs/spec_draft_target_qwen_small.yaml`.

## 4. Final Baseline Benchmark on 4× RTX PRO 6000 Blackwell

Recommended Vast settings:

- GPU: 4× RTX PRO 6000 Blackwell
- Disk: at least 700GB, preferably 1TB
- Image: NVIDIA TensorRT-LLM container
- Port: 8000
- Launch mode: SSH or Jupyter + SSH

Clone repo and check environment:

```bash
cd /workspace
git clone git@github.com:SRO-SA/trtllm-qwen-benchmark.git
cd trtllm-qwen-benchmark
chmod +x scripts/*.sh
bash scripts/setup_check.sh
```

Download the model:

```bash
bash scripts/download_final_model.sh
```

Start baseline server:

```bash
MAX_SEQ_LEN=32768 TP_SIZE=4 \
nohup bash scripts/serve_final_baseline.sh > server_final_baseline_32k.log 2>&1 &

echo $! > server.pid
tail -f server_final_baseline_32k.log
```

Check:

```bash
curl -i http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Run conservative baseline first:

```bash
MODEL_NAME=Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
OUT=results/final_baseline_32k.csv \
CONTEXTS="1024 8192 32768" \
CONCURRENCIES="1 2" \
bash scripts/run_final_baseline_benchmark.sh
```

After stable baseline, extend:

```bash
CONTEXTS="65536 131072" CONCURRENCIES="1" \
OUT=results/final_baseline_long.csv \
bash scripts/run_final_baseline_benchmark.sh
```

## 5. Final DraftTarget Add-on Experiment

Run only after baseline is stable.

Start with a small subset:

```bash
MAX_SEQ_LEN=8192 TP_SIZE=4 \
nohup bash scripts/serve_final_draft_target.sh > server_final_draft_target.log 2>&1 &

echo $! > server.pid
tail -f server_final_draft_target.log
```

Then:

```bash
MODEL_NAME=Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
OUT=results/final_draft_target.csv \
CONTEXTS="1024 8192" \
CONCURRENCIES="1" \
bash scripts/run_final_draft_target_benchmark.sh
```

Do not start directly with 128k + speculative decoding.

## Useful Commands

### Check GPU

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Check TensorRT-LLM flags

```bash
which trtllm-serve
trtllm-serve --help
trtllm-serve serve --help | grep -E "tp_size|max_seq_len|extra_llm_api_options|config"
```

### Start a server in the background

```bash
nohup bash scripts/serve_small.sh > server.log 2>&1 &
echo $! > server.pid
tail -f server.log
```

### Stop a server

```bash
kill $(cat server.pid)
# or
pkill -f trtllm-serve
```

### Save metrics

```bash
curl -m 10 --connect-timeout 2 -sS \
  http://localhost:8000/metrics \
  -o results/tensorrt_metrics.json || true
```

## GitHub SSH Setup on Vast

The script supports a raw private deploy key, a base64 private key, or a key file:

```bash
bash scripts/setup_github_ssh.sh
```

If pasting a raw private key, paste the full key and press `Ctrl+D`.

Private key starts with:

```text
-----BEGIN OPENSSH PRIVATE KEY-----
```

Public key starts with:

```text
ssh-ed25519 ...
```

The Vast instance needs the **private** key. GitHub receives the **public** key as a repo deploy key.


## Safe Benchmark Planning

All benchmark runner scripts now use the shared planner:

```bash
benchmark/plan_safe_tests.py
```

The planner does **not** change any files by itself. It reads the current GPU memory from `nvidia-smi`, estimates the model KV-cache cost from the model config when possible, and marks each `(context_len, concurrency)` pair as `RUN` or `SKIP`.

The main safety rule is:

```text
context_len + max_new_tokens + safety_tokens <= server_max_seq_len
```

This prevents hangs such as running `context_len=512` with `max_new_tokens=64` on a server started with `MAX_SEQ_LEN=512`.

The shared runner:

```bash
scripts/run_benchmark_grid.sh
```

calls the planner first, saves the full plan JSON under `results/`, and only runs the safe cases. The following scripts use the same safety logic:

- `scripts/run_smoke_benchmark.sh`
- `scripts/run_spec_smoke_benchmark.sh`
- `scripts/run_final_baseline_benchmark.sh`
- `scripts/run_final_draft_target_benchmark.sh`

Example for a small server started with `MAX_SEQ_LEN=512`:

```bash
SERVER_MAX_SEQ_LEN=512 \
DECODE_MODE=draft_target \
MODEL_NAME=Qwen3-1.7B \
bash scripts/run_spec_smoke_benchmark.sh
```

The planner will skip unsafe cases like `context_len=512` because the prompt plus generated tokens and safety buffer exceed the server limit.

To inspect a plan without running the benchmark:

```bash
python3 benchmark/plan_safe_tests.py \
  --model /workspace/models/Qwen3-1.7B \
  --tp-size 1 \
  --server-max-seq-len 512 \
  --max-new-tokens 64 \
  --contexts 128,256,512,1024 \
  --concurrency 1,2 \
  --format summary
```

## Assignment Runner: TensorRT-LLM + Qwen3-Coder-480B NVFP4

The assignment requires TensorRT-LLM evaluation on the Qwen3-Coder-480B 4-bit model with RTX 6000 PRO Blackwell GPUs. The required context windows are 1k, 8k, 32k, 64k, and 128k, and the required metrics include TTFT, TPS mean/P99, max concurrency, VRAM idle/load, KV-cache growth, runtime stability, GPU utilization, multi-user scalability, and latency/throughput tradeoffs.

Use the assignment-specific baseline runner instead of the generic smoke-test scripts:

```bash
RESET_RESULTS=1 bash scripts/run_assignment_baseline.sh
```

This script explicitly runs these assignment stages:

```text
short_1k_multiuser:     context 1k,    concurrency 1/2/4/8
medium_8k_multiuser:    context 8k,    concurrency 1/2/4
long_32k:               context 32k,   concurrency 1/2
long_64k:               context 64k,   concurrency 1
long_128k:              context 128k,  concurrency 1
```

The script starts a fresh TensorRT-LLM server for each context group with a large enough `MAX_SEQ_LEN`, then runs the benchmark. The safe planner is still used, but it should not skip the required contexts unless the case is truly unsafe due to memory or configuration. If 64k or 128k fails, keep the server log and report it as the scalability/stability limit.

After the run, summarize the assignment metrics:

```bash
bash scripts/run_assignment_summary.sh
```

Main outputs:

```text
results/assignment_tensorrt_llm_qwen480b_baseline.csv
results/assignment_summary.csv
results/server_logs/
results/metrics/
```

## Stuck/Timeout Detection

The assignment runner includes watchdog-style checks because long-context TensorRT-LLM runs can fail in several ways: the server may never become healthy, a benchmark request may hang, or port 8000 may remain occupied by an old server.

Key scripts:

- `scripts/wait_for_server.sh`: waits for `/health` with a timeout. If the server process exits early or the timeout is reached, it prints diagnostics and the tail of the server log.
- `scripts/run_benchmark_grid.sh`: wraps each benchmark case with a wall-clock timeout (`CASE_TIMEOUT_S`). If a case times out, it appends a failure row to the CSV instead of silently hanging forever.
- `scripts/diagnose_server.sh`: collects `nvidia-smi`, relevant processes, port bind status, health/model/metrics endpoint checks, and the last server log lines.
- `scripts/stop_trtllm_server.sh`: kills old TensorRT-LLM processes and verifies that port 8000 is free using Python, so it does not require `lsof` or `ss`.

Useful knobs:

```bash
WAIT_TIMEOUT_S=3600      # max time to wait for model/server loading
TIMEOUT_S=1800           # request timeout inside benchmark client
CASE_TIMEOUT_S=2100      # wall-clock timeout for a whole benchmark case
RESET_RESULTS=1          # remove old assignment CSV before starting
```

Example:

```bash
RESET_RESULTS=1 \
WAIT_TIMEOUT_S=3600 \
TIMEOUT_S=1800 \
CASE_TIMEOUT_S=2400 \
bash scripts/run_assignment_baseline.sh
```

If a required 64k or 128k assignment case fails, the CSV will contain a `runtime_stability=fail` row and `error_messages` will record the failure reason such as `case_timeout_after_2400s`. The corresponding logs and diagnostic files are saved under `results/server_logs/`, `results/metrics/`, and `results/diagnostics/`.

## Real 480B assignment-runner fixes

The final assignment runner includes two important safety fixes learned from the real 4× RTX PRO 6000 Blackwell test:

1. **Tokenizer-aware prompt generation.** The benchmark client uses the local model tokenizer through `TOKENIZER_PATH` and keeps prompt length below the requested context window using `PROMPT_TOKEN_RESERVE`. This avoids accidental 50k+ token prompts when the target context is 32k.

2. **Separate `max_num_tokens` control.** TensorRT-LLM PyTorch backend may keep `max_num_tokens=8192` even when `max_seq_len` is much larger. The final server scripts now pass `MAX_NUM_TOKENS` and `MAX_INPUT_LEN` when supported by `trtllm-serve`, and the assignment runner checks the parsed server log before sending benchmark requests.

Before rerunning the assignment benchmark on the real machine, test prompt lengths:

```bash
TOKENIZER_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
PROMPT_TOKEN_RESERVE=1024 \
python3 scripts/test_prompt_lengths.py \
  --tokenizer-path /workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
  --contexts "1024 8192 32768"
```

Then rerun the baseline assignment benchmark:

```bash
bash scripts/stop_trtllm_server.sh

RESET_RESULTS=1 \
MODEL_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
MODEL_NAME=Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
PLAN_MODEL=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
TOKENIZER_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
PROMPT_TOKEN_RESERVE=1024 \
TP_SIZE=4 \
QUANTIZATION=nvfp4 \
WAIT_TIMEOUT_S=3600 \
TIMEOUT_S=1800 \
CASE_TIMEOUT_S=2400 \
bash scripts/run_assignment_baseline.sh
```

## Latest assignment-runner fixes

This repo version includes the final fixes for the 4× RTX PRO 6000 Blackwell + TensorRT-LLM + Qwen3-Coder-480B NVFP4 assignment run:

- `run_assignment_baseline.sh` now keeps the required assignment input contexts unchanged: 1k, 8k, 32k, 64k, and 128k.
- Long-context stages now use stage-specific workloads so they do not run 8 repeated huge requests:
  - 32k: `LONG32_NUM_REQUESTS=4`, `LONG32_MAX_NEW_TOKENS=128`
  - 64k: `LONG64_NUM_REQUESTS=1`, `LONG64_MAX_NEW_TOKENS=64`
  - 128k: `LONG128_NUM_REQUESTS=1`, `LONG128_MAX_NEW_TOKENS=32`
- Generated `cuda_graph_config` no longer sets both `batch_sizes` and `max_batch_size`, because TensorRT-LLM 1.1.0 rejects that combination.
- `stop_trtllm_server.sh` now also kills any leftover process owning port 8000, even if its command line no longer contains `trtllm-serve`.
- `benchmark_openai_stream.py` prints per-request progress, which makes long-context runs easier to distinguish from a true hang.
- `scripts/clean_assignment_outputs.sh` backs up and cleans generated results/config/log folders before a clean rerun.

Recommended clean rerun on Vast:

```bash
cd ~/workspace/trtllm-qwen-benchmark
bash scripts/stop_trtllm_server.sh || true
bash scripts/clean_assignment_outputs.sh

RESET_RESULTS=1 \
MODEL_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
MODEL_NAME=Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
PLAN_MODEL=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
TOKENIZER_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 \
PROMPT_TOKEN_RESERVE=1024 \
TP_SIZE=4 \
QUANTIZATION=nvfp4 \
WAIT_TIMEOUT_S=3600 \
TIMEOUT_S=1800 \
CASE_TIMEOUT_S=3600 \
PYTHONPATH=$PWD \
bash scripts/run_assignment_baseline.sh
```

When the 64k stage starts, verify the log prints:

```text
Stage max new tokens: 64
Stage num requests: 1
MAX_NEW_TOKENS=64
NUM_REQUESTS=1
```

If it still prints `MAX_NEW_TOKENS=256` or `NUM_REQUESTS=8` during the 64k stage, the updated script is not being used.


## May 19 long-context stability patch

This repo version fixes a TensorRT-LLM 1.1.0 long-context stall observed at 64K. The server accepted the request with HTTP 200, but logged a negative `default_max_tokens` warning because the effective internal input length stayed around 32K even though `--max_seq_len` and `--max_num_tokens` were larger. The assignment runner now writes `max_input_len`, `max_seq_len`, and `max_num_tokens` into each generated `--extra_llm_api_options` YAML, including the mirrored `build_config` values.

The assignment context lengths are unchanged: 1K, 8K, 32K, 64K, and 128K. Long-context stages use fewer repeated requests to avoid extremely long wall-clock runs: 32K uses 4 requests / 128 output tokens, 64K uses 1 request / 64 output tokens, and 128K uses 1 request / 32 output tokens.

Before each full run:

```bash
bash scripts/stop_trtllm_server.sh || true
bash scripts/clean_assignment_outputs.sh
```

Then run:

```bash
RESET_RESULTS=1 MODEL_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 MODEL_NAME=Qwen3-Coder-480B-A35B-Instruct-NVFP4 PLAN_MODEL=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 TOKENIZER_PATH=/workspace/models/Qwen3-Coder-480B-A35B-Instruct-NVFP4 PROMPT_TOKEN_RESERVE=1024 TP_SIZE=4 QUANTIZATION=nvfp4 WAIT_TIMEOUT_S=3600 TIMEOUT_S=1800 CASE_TIMEOUT_S=3600 BENCHMARK_HEARTBEAT_S=60 DEBUG_BENCHMARK_REQUESTS=1 PYTHONPATH=$PWD bash scripts/run_assignment_baseline.sh
```

For 64K, verify the generated config contains the long input capacity:

```bash
grep -nE "max_input_len|max_seq_len|max_num_tokens" results/runtime_configs/config_long_64k.yaml
```

Expected 64K benchmark printout should include `MAX_NEW_TOKENS=64` and `NUM_REQUESTS=1`.


## Long-context note: chunked prefill

For the final Qwen3-Coder-480B NVFP4 assignment runner, long-context stages use `LONG_CHUNKED_PREFILL=false` by default. In our TensorRT-LLM 1.1.0 PyTorch-backend tests, enabling chunked prefill for the 64K stage allowed the server to return HTTP 200 but then stalled before first token with a warning like:

```text
default_max_tokens (...) = max_seq_len (~33056) - splited_prompt_len (~64519)
```

This indicates an internal chunked-prefill/scheduler limit around the 32K range, even though top-level `max_input_len`, `max_seq_len`, and `max_num_tokens` are configured correctly. To keep assignment context lengths unchanged while avoiding that stall, the runner disables chunked prefill for long-context stages by default. You can override with `LONG_CHUNKED_PREFILL=true` for experiments.


## Long-context endpoint behavior

The assignment runner keeps the required input context lengths unchanged (1k, 8k, 32k, 64k, 128k). For the 64k and 128k stages it uses `OPENAI_API_MODE=completion` by default so the benchmark sends requests to `/v1/completions` instead of `/v1/chat/completions`. This is intentional: on TensorRT-LLM 1.1.0 PyTorch backend, the chat endpoint can accept a very long prompt with HTTP 200 but then compute a negative `default_max_tokens` internally for ~64k prompts, causing the request to wait forever with no GPU compute.

The short, medium, and 32k stages still default to chat mode. You can override these if needed:

```bash
LONG64_API_MODE=chat LONG128_API_MODE=chat bash scripts/run_assignment_baseline.sh
```

The benchmark payload also includes both `max_tokens` and `max_completion_tokens` for chat requests for compatibility across OpenAI-compatible servers.


## KV cache fix for 64K/128K OpenAI serving

Long-context OpenAI/trtllm-serve stages use context-aware KV-cache reservations. A previous 64K run with `LONG64_KV=0.18` initialized the model but only allocated a KV-cache window of about 33K tokens, which caused `default_max_tokens < 0` and a no-progress stall after HTTP 200. The assignment runner now defaults to `LONG64_KV=0.45` and `LONG128_KV=0.75` and includes log-based detection for `window size=...` and negative `default_max_tokens` warnings.

If 64K or 128K OOMs at server initialization, reduce only that stage's KV fraction and report the result as a runtime-stability/scalability limit. Do not reduce the assignment context length.

### Note on HTTP 400 from chat endpoint
TensorRT-LLM 1.1.0 may reject `max_completion_tokens` on `/v1/chat/completions` with HTTP 400. The benchmark now sends only `max_tokens` by default for chat mode. To test a newer server that requires `max_completion_tokens`, set `INCLUDE_MAX_COMPLETION_TOKENS=1`.

## 128K KV retry ladder

The 128K assignment stage now uses a KV-cache retry ladder instead of a single fixed `free_gpu_memory_fraction`. This is necessary because 128K has two opposite failure modes:

- If KV reservation is too small, the server may start but the actual KV-cache window is smaller than the 128K prompt, which causes a request stall.
- If KV reservation is too large, TensorRT-LLM may fail during executor creation with CUDA OOM.

By default, `run_assignment_baseline.sh` tries:

```bash
LONG128_KV_LADDER="0.75 0.70 0.65 0.60 0.55 0.50 0.45"
```

For each attempt, the runner checks the TensorRT-LLM server log using `scripts/check_kv_window.py`. It only sends the 128K request if the reported KV-cache `window size` is at least:

```text
context_len + max_new_tokens + safety_tokens
```

If all attempts fail, the script records one clean failure row for the 128K stage instead of multiple duplicate rows.


### 128K startup hang protection

The assignment runner includes a startup no-progress watchdog. If `trtllm-serve` stays alive but `/health` never becomes ready and the server log stops changing, the runner records `server_start_hung_no_log_progress`, kills the hung attempt, and moves to the next 128K KV-cache retry value. This prevents the 128K ladder from waiting indefinitely after `Model init total -- ...` with no further log output. Tunables:

```bash
LONG128_STARTUP_NO_PROGRESS_TIMEOUT_S=300   # default for 128K attempts
STARTUP_NO_PROGRESS_TIMEOUT_S=600          # default for other stages
STARTUP_NO_PROGRESS_MIN_ELAPSED_S=180
```

## Patch note: long-context completions parsing

For 64K and 128K assignment stages, the runner uses `/v1/completions` rather than `/v1/chat/completions` because the TensorRT-LLM 1.1.0 OpenAI chat wrapper can mis-handle very long prompts. The benchmark now applies the model tokenizer's chat template before sending a completion request, uses non-streaming completions by default for long contexts, and parses both `choices[].text` and chat-like `delta.content`/`message.content` fields. This avoids the earlier `no_output_tokens_returned_or_parsed` false failure when the server completed a long-context request but the benchmark did not extract generated text correctly.

## Full Assignment Scenario Suite

The assignment requires TensorRT-LLM results for Qwen3-Coder-480B NVFP4 on RTX PRO 6000 Blackwell across:

- Single-user inference
- Multi-user/concurrent inference
- Long-context evaluation at 32K, 64K, and 128K
- Concurrent chat sessions
- Parallel code generation
- Sustained throughput testing

This patched repo includes prompt templates in:

```bash
data/assignment_prompts.jsonl
```

Each benchmark row now records:

```text
scenario_name, workload_type, prompt_profile, api_mode, duration_s_requested,
TTFT, TPS mean/P99, VRAM idle/load, KV-cache growth, GPU utilization, runtime stability
```

### Recommended 8-GPU full run

Use this after the model is downloaded and the repo is set up:

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

```bash
results/assignment_tensorrt_llm_qwen480b_full.csv
results/assignment_summary.csv
results/assignment_coverage_summary.csv
results/max_concurrency_summary.csv
results/server_logs/
results/diagnostics/
```

### 128K-only rerun

If you only need to rerun the 128K long-context case:

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
