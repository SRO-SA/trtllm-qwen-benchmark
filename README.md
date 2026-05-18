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
