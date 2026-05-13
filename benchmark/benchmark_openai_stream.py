import argparse
import csv
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


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


def make_prompt(context_len):
    # Approximate token length for smoke testing.
    # For real final benchmarking, replace this with tokenizer-based prompt generation.
    base = "You are a coding assistant. Analyze the following synthetic context and answer briefly.\n\n"
    repeated = "def foo(x): return x + 1\n"
    target_chars = context_len * 4
    body = repeated * max(1, target_chars // len(repeated))
    return base + body + "\nQuestion: Write one sentence summarizing what the code does."


def run_one_request(url, model, context_len, max_tokens, request_id):
    prompt = make_prompt(context_len)
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
    error = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as r:
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

    total_time_s = max(end - start, 1e-9)

    # If server does not return usage in streaming mode, use a rough fallback.
    if completion_tokens is None:
        completion_tokens = max(1, len(output_text.split()))

    tps = completion_tokens / total_time_s

    return {
        "request_id": request_id,
        "success": error is None,
        "error": error or "",
        "ttft_ms": ttft_ms,
        "total_time_s": total_time_s,
        "completion_tokens": completion_tokens,
        "tps": tps,
        "output_chars": len(output_text),
    }


def p99(values):
    if not values:
        return None
    values = sorted(values)
    idx = int(0.99 * (len(values) - 1))
    return values[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--framework", default="tensorrt-llm")
    parser.add_argument("--quantization", default="smoke-test")
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
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

    row = {
        "framework": args.framework,
        "model": args.model,
        "quantization": args.quantization,
        "gpu_type": "; ".join(sorted(set(g.get("gpu_name", "unknown") for g in load_gpu))),
        "num_gpus": len([g for g in load_gpu if "gpu_index" in g]),
        "context_len": args.context_len,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_tokens,
        "num_requests": args.num_requests,
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "ttft_mean_ms": statistics.mean(ttfts) if ttfts else "",
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else "",
        "ttft_p99_ms": p99(ttfts) if ttfts else "",
        "tps_mean": statistics.mean(tps_vals) if tps_vals else "",
        "tps_p50": statistics.median(tps_vals) if tps_vals else "",
        "tps_p99": p99(tps_vals) if tps_vals else "",
        "total_output_tokens": sum(r["completion_tokens"] for r in successes),
        "total_time_s": end_wall - start_wall,
        "vram_idle_gb": vram_idle_gb,
        "vram_load_gb": vram_load_gb,
        "kv_cache_growth_gb": max(0.0, vram_load_gb - vram_idle_gb),
        "gpu_util_mean_after": mean_gpu_util(load_gpu),
        "runtime_stability": "pass" if len(failures) == 0 else "fail",
        "error_count": len(failures),
        "error_messages": " | ".join(sorted(set(f["error"] for f in failures if f["error"]))),
    }

    fieldnames = list(row.keys())

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()