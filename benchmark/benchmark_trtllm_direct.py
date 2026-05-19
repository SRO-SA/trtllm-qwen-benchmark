#!/usr/bin/env python3
"""Direct TensorRT-LLM API benchmark for long-context diagnostics.

This bypasses trtllm-serve/OpenAI endpoints and calls TensorRT-LLM's Python
LLM API directly. It is intended for the 64K/128K assignment cases where the
OpenAI-compatible serving path can accept HTTP 200 but stall before first token.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.benchmark_openai_stream import make_prompt, gpu_snapshot, sum_gpu_mem_gb, mean_gpu_util


def _debug(msg: str) -> None:
    if os.environ.get("DEBUG_BENCHMARK_REQUESTS", "1").lower() not in {"0", "false", "no", "off"}:
        print(msg, flush=True)


def _heartbeat(stop_event: threading.Event, start_time: float, interval_s: float) -> None:
    while not stop_event.wait(interval_s):
        snap = gpu_snapshot()
        util = mean_gpu_util(snap)
        vram = sum_gpu_mem_gb(snap)
        _debug(
            f"[direct heartbeat] elapsed={time.perf_counter() - start_time:.1f}s; "
            f"vram_used_total_gb={vram:.2f}; gpu_util_mean={util}"
        )


def _import_trtllm():
    try:
        from tensorrt_llm import LLM, SamplingParams  # type: ignore
        return LLM, SamplingParams
    except Exception:
        from tensorrt_llm.llmapi import LLM, SamplingParams  # type: ignore
        return LLM, SamplingParams


def _try_import_config(name: str):
    for module in ("tensorrt_llm.llmapi", "tensorrt_llm.llmapi.llm_args", "tensorrt_llm"):
        try:
            mod = __import__(module, fromlist=[name])
            return getattr(mod, name)
        except Exception:
            pass
    return None


def _make_kv_cache_config(kv_fraction: float, kv_dtype: str, tokens_per_block: int) -> Any:
    data = {
        "free_gpu_memory_fraction": kv_fraction,
        "dtype": kv_dtype,
        "tokens_per_block": tokens_per_block,
    }
    cls = _try_import_config("KvCacheConfig")
    if cls is not None:
        try:
            return cls(**data)
        except Exception:
            pass
    return data


def _make_cuda_graph_config(batch_sizes: List[int]) -> Any:
    data = {
        "batch_sizes": batch_sizes,
        "enable_padding": False,
    }
    cls = _try_import_config("CudaGraphConfig")
    if cls is not None:
        try:
            return cls(**data)
        except Exception:
            pass
    return data


def _make_sampling_params(SamplingParams, max_tokens: int):
    # TensorRT-LLM versions have used slightly different accepted fields. Try the
    # standard max_tokens form first, then fall back to max_new_tokens.
    for kwargs in (
        {"max_tokens": max_tokens, "temperature": 0.0},
        {"max_new_tokens": max_tokens, "temperature": 0.0},
        {"max_tokens": max_tokens},
    ):
        try:
            return SamplingParams(**kwargs)
        except Exception:
            continue
    raise RuntimeError("Could not construct TensorRT-LLM SamplingParams")


def _extract_text_and_tokens(output: Any) -> tuple[str, Optional[int]]:
    """Best-effort parser for TensorRT-LLM output objects across versions."""
    text = ""
    tokens = None

    try:
        if isinstance(output, (list, tuple)) and output:
            output = output[0]

        if hasattr(output, "outputs") and output.outputs:
            first = output.outputs[0]
            text = getattr(first, "text", "") or ""
            for attr in ("token_ids", "tokens", "output_token_ids"):
                vals = getattr(first, attr, None)
                if vals is not None:
                    try:
                        tokens = len(vals)
                        break
                    except Exception:
                        pass
        elif hasattr(output, "text"):
            text = getattr(output, "text", "") or ""
            vals = getattr(output, "token_ids", None)
            if vals is not None:
                try:
                    tokens = len(vals)
                except Exception:
                    pass
        elif isinstance(output, dict):
            text = output.get("text") or output.get("output_text") or ""
            vals = output.get("token_ids") or output.get("tokens")
            if vals is not None:
                tokens = len(vals)
    except Exception:
        pass

    if tokens is None:
        tokens = len(text.split()) if text else 0
    return text, tokens


def append_row(path: str, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="Qwen3-Coder-480B-A35B-Instruct-NVFP4")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--backend", default="pytorch")
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--max-input-len", type=int, default=0)
    parser.add_argument("--max-num-tokens", type=int, required=True)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--kv-memory-fraction", type=float, default=0.18)
    parser.add_argument("--kv-dtype", default="auto")
    parser.add_argument("--tokens-per-block", type=int, default=32)
    parser.add_argument("--cuda-graph-batch-sizes", default="1")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--heartbeat-s", type=float, default=float(os.environ.get("BENCHMARK_HEARTBEAT_S", "60")))
    parser.add_argument("--output", default="results/assignment_tensorrt_llm_qwen480b_direct.csv")
    parser.add_argument("--quantization", default="nvfp4")
    parser.add_argument("--decode-mode", default="baseline_direct")
    args = parser.parse_args()

    os.environ["TOKENIZER_PATH"] = args.tokenizer_path or args.model_path
    os.environ.setdefault("PLAN_MODEL", args.model_path)
    os.environ.setdefault("MODEL_PATH", args.model_path)

    idle_gpu = gpu_snapshot()
    vram_idle_gb = sum_gpu_mem_gb(idle_gpu)
    start_all = time.perf_counter()
    stop_event = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(stop_event, start_all, args.heartbeat_s), daemon=True)
    hb.start()

    successes: List[Dict[str, Any]] = []
    failures: List[str] = []
    total_output_tokens = 0
    model_init_s = None

    prompt, prompt_tokens_est, target_prompt_tokens = make_prompt(args.context_len)
    _debug(
        f"[direct] Prompt ready: context={args.context_len}, target_prompt_tokens={target_prompt_tokens}, "
        f"prompt_tokens_est={prompt_tokens_est}, prompt_chars={len(prompt)}"
    )

    try:
        LLM, SamplingParams = _import_trtllm()
        batch_sizes = [int(x) for x in args.cuda_graph_batch_sizes.replace(",", " ").split() if x.strip()]
        max_input_len = args.max_input_len or args.max_seq_len

        llm_kwargs: Dict[str, Any] = {
            "model": args.model_path,
            "tokenizer": args.tokenizer_path or args.model_path,
            "tensor_parallel_size": args.tp_size,
            "backend": args.backend,
            "max_input_len": max_input_len,
            "max_seq_len": args.max_seq_len,
            "max_num_tokens": args.max_num_tokens,
            "max_batch_size": args.max_batch_size,
            "enable_chunked_prefill": bool(args.enable_chunked_prefill),
            "kv_cache_config": _make_kv_cache_config(args.kv_memory_fraction, args.kv_dtype, args.tokens_per_block),
            "enable_iter_perf_stats": True,
        }
        if not args.disable_cuda_graph:
            llm_kwargs["cuda_graph_config"] = _make_cuda_graph_config(batch_sizes)

        _debug("[direct] Constructing TensorRT-LLM LLM with kwargs:")
        safe_kwargs = {k: str(v) if k.endswith("config") else v for k, v in llm_kwargs.items()}
        _debug(json.dumps(safe_kwargs, indent=2, default=str))

        t0 = time.perf_counter()
        llm = LLM(**llm_kwargs)
        model_init_s = time.perf_counter() - t0
        _debug(f"[direct] LLM initialized in {model_init_s:.2f}s")

        sampling_params = _make_sampling_params(SamplingParams, args.max_new_tokens)

        for i in range(args.num_requests):
            _debug(f"[direct request {i}] Starting generate(...)")
            req_start = time.perf_counter()
            try:
                output = llm.generate([prompt], sampling_params=sampling_params)
                req_end = time.perf_counter()
                text, comp_tokens = _extract_text_and_tokens(output)
                total_output_tokens += int(comp_tokens or 0)
                successes.append(
                    {
                        "request_id": i,
                        "total_time_s": req_end - req_start,
                        "completion_tokens": int(comp_tokens or 0),
                        "output_chars": len(text),
                    }
                )
                _debug(
                    f"[direct request {i}] Done in {req_end - req_start:.2f}s; "
                    f"completion_tokens={comp_tokens}; output_chars={len(text)}"
                )
            except Exception as e:
                failures.append(repr(e))
                _debug(f"[direct request {i}] Failed: {repr(e)}")

    except Exception as e:
        failures.append(repr(e))
        _debug(f"[direct] Failed before/during model init or generation: {repr(e)}")
    finally:
        stop_event.set()

    end_all = time.perf_counter()
    load_gpu = gpu_snapshot()
    vram_load_gb = sum_gpu_mem_gb(load_gpu)
    total_time_s = max(end_all - start_all, 1e-9)
    gen_times = [r["total_time_s"] for r in successes]
    tps_vals = [r["completion_tokens"] / max(r["total_time_s"], 1e-9) for r in successes]

    row = {
        "framework": "tensorrt-llm-direct",
        "model": args.model_name,
        "quantization": args.quantization,
        "decode_mode": args.decode_mode,
        "gpu_type": "; ".join(sorted(set(g.get("gpu_name", "unknown") for g in load_gpu))),
        "num_gpus": len([g for g in load_gpu if "gpu_index" in g]),
        "context_len": args.context_len,
        "concurrency": 1,
        "max_new_tokens": args.max_new_tokens,
        "target_prompt_tokens": target_prompt_tokens,
        "prompt_tokens_est": prompt_tokens_est,
        "prompt_tokens_reported": "",
        "num_requests": args.num_requests,
        "successful_requests": len(successes),
        "failed_requests": len(failures) + max(0, args.num_requests - len(successes) - len(failures)),
        "ttft_mean_ms": "",  # Direct non-streaming API does not expose TTFT.
        "ttft_p50_ms": "",
        "ttft_p99_ms": "",
        "tps_mean": statistics.mean(tps_vals) if tps_vals else "",
        "tps_p50": statistics.median(tps_vals) if tps_vals else "",
        "tps_p99": sorted(tps_vals)[int(0.99 * (len(tps_vals) - 1))] if tps_vals else "",
        "aggregate_tps": total_output_tokens / total_time_s,
        "total_output_tokens": total_output_tokens,
        "total_time_s": total_time_s,
        "model_init_s": model_init_s if model_init_s is not None else "",
        "direct_generation_time_mean_s": statistics.mean(gen_times) if gen_times else "",
        "vram_idle_gb": vram_idle_gb,
        "vram_load_gb": vram_load_gb,
        "kv_cache_growth_gb": max(0.0, vram_load_gb - vram_idle_gb),
        "gpu_util_mean_after": mean_gpu_util(load_gpu),
        "runtime_stability": "pass" if not failures and len(successes) == args.num_requests else "fail",
        "error_count": len(failures),
        "error_messages": " | ".join(sorted(set(failures))),
    }

    append_row(args.output, row)
    print(json.dumps(row, indent=2), flush=True)

    if row["runtime_stability"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
