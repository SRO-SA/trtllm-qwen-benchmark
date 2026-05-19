#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path


def extract_last_int(patterns, text):
    """Return the integer from the chronologically last regex match."""
    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            try:
                matches.append((m.start(), int(m.group(1))))
            except Exception:
                pass
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--required-seq-len", type=int, required=True)
    p.add_argument("--required-num-tokens", type=int, required=True)
    args = p.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"ERROR: server log not found: {path}")
        return 2

    text = path.read_text(errors="replace")

    max_seq_len = extract_last_int([
        r"max_seq_len=(\d+)",
        r"Max seq len:\s*(\d+)",
        r"build_config=BuildConfig\([^\n]*max_seq_len=(\d+)",
    ], text)
    max_num_tokens = extract_last_int([
        r"max_num_tokens=(\d+)",
        r"Max num tokens:\s*(\d+)",
        r"build_config=BuildConfig\([^\n]*max_num_tokens=(\d+)",
    ], text)
    max_input_len = extract_last_int([
        r"max_input_len=(\d+)",
        r"Max input len:\s*(\d+)",
        r"build_config=BuildConfig\([^\n]*max_input_len=(\d+)",
    ], text)

    print("Parsed TensorRT-LLM server limits:")
    print(f"  max_seq_len={max_seq_len}")
    print(f"  max_num_tokens={max_num_tokens}")
    print(f"  max_input_len={max_input_len}")
    print("Required limits:")
    print(f"  required_seq_len={args.required_seq_len}")
    print(f"  required_num_tokens={args.required_num_tokens}")

    errors = []
    if max_seq_len is None:
        errors.append("could_not_parse_max_seq_len")
    elif max_seq_len < args.required_seq_len:
        errors.append(f"max_seq_len_too_small:{max_seq_len}<{args.required_seq_len}")

    if max_num_tokens is None:
        errors.append("could_not_parse_max_num_tokens")
    elif max_num_tokens < args.required_num_tokens:
        errors.append(f"max_num_tokens_too_small:{max_num_tokens}<{args.required_num_tokens}")

    # For long-context serving this is required. In TensorRT-LLM 1.1.0, the CLI may
    # not expose --max_input_len, so run_assignment_baseline.sh writes it into the
    # extra LLM YAML. If it remains at 1024/32768, 64k/128k can get HTTP 200 but stall.
    if max_input_len is None:
        errors.append("could_not_parse_max_input_len")
    elif max_input_len < args.required_num_tokens:
        errors.append(f"max_input_len_too_small:{max_input_len}<{args.required_num_tokens}")

    if errors:
        print("ERROR: server limits are not sufficient: " + ";".join(errors))
        return 1

    # Detect the known internal stall warning early if checking after a request.
    if re.search(r"default_max_tokens \(-?\d+\).*splited_prompt_len", text):
        print("ERROR: server log contains TensorRT-LLM negative default_max_tokens warning; long-context request likely stalled.")
        return 1

    print("Server limits are sufficient for this stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
