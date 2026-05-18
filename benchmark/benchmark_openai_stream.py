import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import requests

_TOKENIZER = None
_TOKENIZER_PATH = None


def gpu_snapshot():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True)
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                rows.append(
                    {
                        "gpu_index": parts[0],
                        "gpu_name": parts[1],
                        "mem_used_mb": float(parts[2]),
                        "mem_total_mb": float(parts[3]),
                        "gpu_util_percent": float(parts[4]),
                    }
                )
        return rows
    except Exception as e:
        return [{"error": repr(e)}]


def sum_gpu_mem_gb(snapshot):
    total_mb = 0.0
    for row in snapshot:
        if "mem_used_mb" in row:
            total_mb += row["mem_used_mb"]
    return total_mb / 1024.0


def mean_gpu_util(snapshot):
    vals = [row["gpu_util_percent"] for row in snapshot if "gpu_util_percent" in row]
    return statistics.mean(vals) if vals else None


def _get_tokenizer(tokenizer_path: Optional[str]):
    global _TOKENIZER, _TOKENIZER_PATH
    if not tokenizer_path:
        return None
    if _TOKENIZER is not None and _TOKENIZER_PATH == tokenizer_path:
        return _TOKENIZER
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
        )
        _TOKENIZER_PATH = tokenizer_path
        return _TOKENIZER
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer from {tokenizer_path}: {repr(e)}")
        return None


def _encode_len(tokenizer, text: str) -> int:
    if tokenizer is None:
        return -1
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_prompt(context_len: int) -> Tuple[str, int, int]:
    """Create a synthetic code prompt with tokenizer-aware length control.

    context_len is the assignment target window (1k, 8k, 32k, 64k, 128k). We
    create a prompt that is safely below that target, leaving reserve tokens for
    chat-template overhead and generated output.

    This is important for TensorRT-LLM: the server validates the *tokenized*
    prompt against max_num_tokens. Character-based estimates can overshoot badly
    for Qwen/Qwen3-Coder and caused 32k tests to become 52k+ token prompts.
    """
    tokenizer_path = (
        os.environ.get("TOKENIZER_PATH")
        or os.environ.get("PLAN_MODEL")
        or os.environ.get("MODEL_PATH")
    )
    reserve_tokens = int(os.environ.get("PROMPT_TOKEN_RESERVE", "1024"))
    target_prompt_tokens = max(32, int(context_len) - reserve_tokens)

    header = (
        "You are a coding assistant. Read the following synthetic Python code context. "
        "Answer only the final question.\n\n"
    )
    unit = (
        "def transform_value(x):\n"
        "    y = x + 1\n"
        "    z = y * 2\n"
        "    if z % 3 == 0:\n"
        "        return z - 1\n"
        "    return z + 1\n\n"
    )
    footer = "\nQuestion: In one sentence, summarize what transform_value repeatedly does."

    tokenizer = _get_tokenizer(tokenizer_path)

    if tokenizer is None:
        # Conservative fallback: use fewer chars/token than before to avoid huge overshoot.
        target_chars = max(1, target_prompt_tokens) * 2
        body = unit * max(1, target_chars // len(unit))
        prompt = header + body + footer
        return prompt, -1, target_prompt_tokens

    fixed = header + footer
    fixed_ids = tokenizer.encode(fixed, add_special_tokens=False)
    unit_ids = tokenizer.encode(unit, add_special_tokens=False)
    remaining = max(1, target_prompt_tokens - len(fixed_ids))

    # Build exact-ish body in token space, then decode back to text.
    repeated_ids = []
    while len(repeated_ids) < remaining:
        repeated_ids.extend(unit_ids)
    repeated_ids = repeated_ids[:remaining]

    body = tokenizer.decode(repeated_ids, skip_special_tokens=True)
    prompt = header + body + footer
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Final trim if decode/re-encode produced a slight overrun.
    if len(ids) > target_prompt_tokens:
        ids = ids[:target_prompt_tokens]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)

    return prompt, len(ids), target_prompt_tokens


def run_one_request(url, model, context_len, max_tokens, request_id, timeout_s):
    prompt, prompt_tokens_est, target_prompt_tokens = make_prompt(context_len)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    first_token_time = None
    end = None
    output_text = ""
    completion_tokens = None
    prompt_tokens_reported = None
    error = None
    status_code = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
            status_code = r.status_code
            r.raise_for_status()

            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                line = raw_line.strip()
                if not line.startswith("data: "):
                    continue

                data = line[len("data: ") :]
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except Exception:
                    continue

                if "usage" in obj and obj["usage"]:
                    completion_tokens = obj["usage"].get("completion_tokens", completion_tokens)
                    prompt_tokens_reported = obj["usage"].get("prompt_tokens", prompt_tokens_reported)

                choices = obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        output_text += content

            end = time.perf_counter()

    except Exception as e:
        end = time.perf_counter()
        error = repr(e)

    ttft_ms = None
    if first_token_time is not None:
        ttft_ms = (first_token_time - start) * 1000.0

    total_time_s = max((end or time.perf_counter()) - start, 1e-9)

    # If server does not return usage in streaming mode, use a rough fallback.
    if completion_tokens is None:
        completion_tokens = max(1, len(output_text.split())) if output_text else 0

    tps = completion_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "request_id": request_id,
        "success": error is None,
        "status_code": status_code,
        "error": error or "",
        "ttft_ms": ttft_ms,
        "total_time_s": total_time_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens_est": prompt_tokens_est,
        "target_prompt_tokens": target_prompt_tokens,
        "prompt_tokens_reported": prompt_tokens_reported,
        "tps": tps,
        "output_chars": len(output_text),
    }


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = int(q * (len(values) - 1))
    return values[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--framework", default="tensorrt-llm")
    parser.add_argument("--quantization", default="smoke-test")
    parser.add_argument("--decode-mode", default="baseline", help="baseline, draft_target, eagle3, etc.")
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--output", default="results/smoke_results.csv")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    idle_gpu = gpu_snapshot()
    vram_idle_gb = sum_gpu_mem_gb(idle_gpu)

    results = []
    start_wall = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(
                run_one_request,
                url,
                args.model,
                args.context_len,
                args.max_tokens,
                i,
                args.timeout_s,
            )
            for i in range(args.num_requests)
        ]

        for fut in as_completed(futures):
            results.append(fut.result())

    end_wall = time.perf_counter()

    load_gpu = gpu_snapshot()
    vram_load_gb = sum_gpu_mem_gb(load_gpu)

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    ttfts = [r["ttft_ms"] for r in successes if r["ttft_ms"] is not None]
    tps_vals = [r["tps"] for r in successes if r["tps"] is not None]
    total_output_tokens = sum(r["completion_tokens"] for r in successes)
    total_time_s = max(end_wall - start_wall, 1e-9)
    aggregate_tps = total_output_tokens / total_time_s

    prompt_est_values = [r.get("prompt_tokens_est") for r in results if r.get("prompt_tokens_est") not in (None, -1)]
    prompt_reported_values = [r.get("prompt_tokens_reported") for r in results if r.get("prompt_tokens_reported") is not None]
    target_prompt_values = [r.get("target_prompt_tokens") for r in results if r.get("target_prompt_tokens") is not None]

    row = {
        "framework": args.framework,
        "model": args.model,
        "quantization": args.quantization,
        "decode_mode": args.decode_mode,
        "gpu_type": "; ".join(sorted(set(g.get("gpu_name", "unknown") for g in load_gpu))),
        "num_gpus": len([g for g in load_gpu if "gpu_index" in g]),
        "context_len": args.context_len,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_tokens,
        "target_prompt_tokens": max(target_prompt_values) if target_prompt_values else "",
        "prompt_tokens_est": max(prompt_est_values) if prompt_est_values else "",
        "prompt_tokens_reported": max(prompt_reported_values) if prompt_reported_values else "",
        "num_requests": args.num_requests,
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "ttft_mean_ms": statistics.mean(ttfts) if ttfts else "",
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else "",
        "ttft_p99_ms": percentile(ttfts, 0.99) if ttfts else "",
        "tps_mean": statistics.mean(tps_vals) if tps_vals else "",
        "tps_p50": statistics.median(tps_vals) if tps_vals else "",
        "tps_p99": percentile(tps_vals, 0.99) if tps_vals else "",
        "aggregate_tps": aggregate_tps,
        "total_output_tokens": total_output_tokens,
        "total_time_s": total_time_s,
        "vram_idle_gb": vram_idle_gb,
        "vram_load_gb": vram_load_gb,
        "kv_cache_growth_gb": max(0.0, vram_load_gb - vram_idle_gb),
        "gpu_util_mean_after": mean_gpu_util(load_gpu),
        "runtime_stability": "pass" if len(failures) == 0 else "fail",
        "error_count": len(failures),
        "error_messages": " | ".join(sorted(set(f["error"] for f in failures if f["error"]))),
    }

    fieldnames = list(row.keys())
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
