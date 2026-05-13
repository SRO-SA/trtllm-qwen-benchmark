# TensorRT-LLM Qwen Benchmark

This repository contains scripts for testing and benchmarking TensorRT-LLM for the Qwen3-Coder-480B 4-bit inference task.

The final goal is to run TensorRT-LLM on a 4× RTX PRO 6000 Blackwell machine and collect system metrics for different context lengths and concurrency levels.

## Assignment Target

Framework:

- TensorRT-LLM

Model:

- Qwen3-Coder-480B
- 4-bit offline/pre-made quantization, likely AWQ

Target GPU:

- 4× RTX PRO 6000 Blackwell

Context windows:

- 1k
- 8k
- 32k
- 64k
- 128k

Required metrics:

- TTFT
- TPS mean
- TPS P99
- Max concurrency
- VRAM idle/load
- KV-cache growth
- Runtime stability
- GPU utilization
- Throughput degradation under concurrency
- Multi-user scalability limits
- Latency/throughput tradeoff

## Current Status

A smoke test was completed on a cheaper Vast.ai instance before moving to the final 4-GPU setup.

Smoke-test setup:

- GPU: 1× NVIDIA RTX 6000 Ada Generation
- Framework: TensorRT-LLM
- TensorRT-LLM version: 1.1.0
- Backend: PyTorch backend
- Test model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

The smoke test verified:

- TensorRT-LLM installation
- `trtllm-serve` availability
- CUDA/GPU visibility
- OpenAI-compatible API
- `/health`
- `/v1/models`
- `/v1/chat/completions`
- `/metrics`
- Benchmark CSV generation

## Repository Structure

```text
trtllm-qwen-benchmark/
├── scripts/
│   ├── setup_check.sh
│   ├── serve_small.sh
│   ├── test_request.sh
│   ├── run_smoke_benchmark.sh
│   └── setup_github_ssh.sh
├── benchmark/
│   └── benchmark_openai_stream.py
├── results/
│   └── smoke_results.csv
└── README.md
```

## Scripts

### `scripts/setup_check.sh`

Checks whether the environment is ready.

It verifies:

- GPU visibility with `nvidia-smi`
- Python version
- `trtllm-serve` availability
- PyTorch CUDA access
- TensorRT-LLM Python import
- Required helper packages such as `requests`, `pandas`, `numpy`, and `tqdm`

Run:

```bash
bash scripts/setup_check.sh
```

Expected successful signs:

```text
trtllm-serve found
CUDA available: True
TensorRT-LLM import: OK
```

---

### `scripts/serve_small.sh`

Starts a small TensorRT-LLM test server using TinyLlama.

Default model:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Run in foreground:

```bash
bash scripts/serve_small.sh
```

Run in background:

```bash
nohup bash scripts/serve_small.sh > server.log 2>&1 &
echo $! > server.pid
```

Check server log:

```bash
tail -f server.log
```

Stop watching the log with:

```text
Ctrl + C
```

This only stops `tail`, not the server.

Stop the server:

```bash
kill $(cat server.pid)
```

If needed:

```bash
pkill -f trtllm-serve
```

---

### `scripts/test_request.sh`

Sends one test request to the running TensorRT-LLM server.

Run:

```bash
bash scripts/test_request.sh
```

This checks whether the OpenAI-compatible `/v1/chat/completions` endpoint works.

---

### `scripts/run_smoke_benchmark.sh`

Runs a small benchmark against the local TensorRT-LLM server.

Current smoke-test settings:

- model: TinyLlama
- context length: small test contexts
- concurrency: 1, 2, 4
- output: `results/smoke_results.csv`

Run:

```bash
bash scripts/run_smoke_benchmark.sh
```

View results:

```bash
cat results/smoke_results.csv
```

The CSV includes:

- framework
- model
- quantization
- GPU type
- number of GPUs
- context length
- concurrency
- max new tokens
- number of requests
- successful requests
- failed requests
- TTFT mean/P50/P99
- TPS mean/P50/P99
- total output tokens
- total runtime
- VRAM idle/load
- approximate KV-cache growth
- GPU utilization
- runtime stability
- error count
- error messages

---

### `benchmark/benchmark_openai_stream.py`

Main benchmark script.

It sends requests to an OpenAI-compatible server and measures:

- time to first token
- total request time
- generated tokens per second
- successful/failed requests
- GPU memory before and after benchmark
- approximate KV-cache growth
- runtime stability

Example direct usage:

```bash
python3 benchmark/benchmark_openai_stream.py \
  --host localhost \
  --port 8000 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --framework tensorrt-llm \
  --quantization smoke-test \
  --context-len 1024 \
  --concurrency 1 \
  --num-requests 8 \
  --max-tokens 64 \
  --output results/smoke_results.csv
```

---

### `scripts/setup_github_ssh.sh`

Optional script for setting up GitHub SSH access on a new Vast.ai instance.

Important:

- Do not commit private SSH keys into this repository.
- Use a repo-specific deploy key if possible.
- The script expects the private key to be provided at runtime, preferably as a base64 string.

Run:

```bash
bash scripts/setup_github_ssh.sh
```

## Useful Commands

### Check GPU

```bash
nvidia-smi
```

Continuous GPU monitoring:

```bash
watch -n 1 nvidia-smi
```

### Check TensorRT-LLM

```bash
which trtllm-serve
trtllm-serve --help
trtllm-serve serve --help
```

### Start Server in Background

```bash
nohup bash scripts/serve_small.sh > server.log 2>&1 &
echo $! > server.pid
```

### Check Server Log

```bash
tail -f server.log
```

### Check Whether Server Is Running

```bash
ps aux | grep trtllm
```

### Health Check

```bash
curl -i http://localhost:8000/health
```

Expected:

```text
HTTP/1.1 200 OK
```

### List Served Models

```bash
curl http://localhost:8000/v1/models
```

### Send One Chat Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "user", "content": "Write a short Python function that adds two numbers."}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

### Test Streaming Response

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 32,
    "temperature": 0,
    "stream": true
  }'
```

### Check TensorRT-LLM Metrics

```bash
curl http://localhost:8000/metrics
```

Save metrics with timeout:

```bash
curl -m 10 --connect-timeout 2 -sS \
  http://localhost:8000/metrics \
  -o results/tensorrt_metrics.json
```

If `/metrics` hangs, stop it with:

```text
Ctrl + C
```

Then restart the server.

### Stop Server

```bash
kill $(cat server.pid)
```

If that does not work:

```bash
pkill -f trtllm-serve
```

### Clean Old Results

```bash
rm -f results/smoke_results.csv
rm -f results/tensorrt_metrics.json
```

## Smoke-Test Result

The smoke test successfully produced the following result for 1024 context length:

| Context | Concurrency | TTFT Mean (ms) | TPS Mean | TPS P99 | VRAM Load (GB) | Stability |
|---:|---:|---:|---:|---:|---:|---|
| 1024 | 1 | 13.00 | 320.17 | 334.06 | 42.71 | pass |
| 1024 | 2 | 11.47 | 313.40 | 315.45 | 42.71 | pass |
| 1024 | 4 | 27.03 | 285.41 | 305.16 | 42.71 | pass |

This confirms that the benchmark pipeline works.

## Notes About the Smoke Test

The TinyLlama smoke test is only for validating the environment and scripts. It is not intended to represent final TensorRT-LLM performance.

The 2048-context smoke test stalled with TinyLlama, likely because the approximate prompt length plus generated tokens reached the model/server context limit. For this reason, the smoke test should stay small.

The final long-context benchmark should be done with Qwen3-Coder-480B on the real 4× RTX PRO 6000 Blackwell machine.

## Final Benchmark Plan

For the real benchmark, use:

- 4× RTX PRO 6000 Blackwell
- Qwen3-Coder-480B 4-bit quantized model
- TensorRT-LLM
- Tensor parallel size 4
- Local model path under `/workspace/models`

Suggested order:

```text
1k context, concurrency 1
8k context, concurrency 1
32k context, concurrency 1
32k context, concurrency 2
64k context, concurrency 1
128k context, concurrency 1
```

Then gradually increase concurrency.

Do not start directly with 128k context and high concurrency.

## Final Model Serving Placeholder

The final serving command will depend on the exact Qwen3-Coder-480B 4-bit checkpoint format.

Expected shape:

```bash
MODEL_PATH="/workspace/models/Qwen3-Coder-480B-AWQ"

trtllm-serve serve \
  --backend pytorch \
  --host 0.0.0.0 \
  --port 8000 \
  --tp_size 4 \
  "${MODEL_PATH}"
```

Before running the final model, check the available TensorRT-LLM flags:

```bash
trtllm-serve serve --help
```

Useful flags may include:

```text
--tp_size
--max_seq_len
--backend
--host
--port
--extra_llm_api_options
```

## Recommended Final Workflow

1. Rent the 4× RTX PRO 6000 Blackwell instance.
2. Clone this repository.
3. Run `scripts/setup_check.sh`.
4. Download or sync the Qwen3-Coder-480B 4-bit model to local disk.
5. Start TensorRT-LLM server in background.
6. Run a 1k context sanity request.
7. Run the benchmark from small context to long context.
8. Save CSV results and TensorRT metrics.
9. Upload results to GitHub or cloud storage.
10. Stop/destroy the Vast instance when finished.

## Safety Note

Do not store GitHub private keys, Hugging Face tokens, cloud credentials, or model access tokens directly in this repository.

Use environment variables, Vast secrets, or runtime prompts instead.
