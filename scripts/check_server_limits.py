#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path


def extract_last_int(patterns, text):
    last = None
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            last = int(m.group(1))
    return last


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
    ], text)
    max_num_tokens = extract_last_int([
        r"max_num_tokens=(\d+)",
        r"Max num tokens:\s*(\d+)",
    ], text)
    max_input_len = extract_last_int([
        r"max_input_len=(\d+)",
        r"Max input len:\s*(\d+)",
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

    # max_input_len is useful but not always enforced/reported the same way.
    if max_input_len is not None and max_input_len < args.required_num_tokens:
        errors.append(f"max_input_len_too_small:{max_input_len}<{args.required_num_tokens}")

    if errors:
        print("ERROR: server limits are not sufficient: " + ";".join(errors))
        return 1

    print("Server limits are sufficient for this stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
