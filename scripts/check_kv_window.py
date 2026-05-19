#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--log', required=True)
parser.add_argument('--required-tokens', type=int, required=True)
args = parser.parse_args()

text = Path(args.log).read_text(errors='ignore') if Path(args.log).exists() else ''
windows = [int(x) for x in re.findall(r'window size=(\d+)', text)]
if not windows:
    print('No KV-cache window size found in log yet.')
    sys.exit(2)

w = windows[-1]
print(f'Latest KV-cache window size: {w}')
print(f'Required tokens: {args.required_tokens}')
if w < args.required_tokens:
    print(f'FAIL: KV-cache window too small: window={w}, required={args.required_tokens}')
    sys.exit(1)
print('PASS: KV-cache window is sufficient.')
