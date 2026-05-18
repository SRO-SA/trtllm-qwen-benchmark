#!/usr/bin/env python3
import argparse
import os
from transformers import AutoTokenizer
from benchmark.benchmark_openai_stream import make_prompt


def parse_contexts(s):
    return [int(x) for x in s.replace(',', ' ').split() if x]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer-path', required=True)
    p.add_argument('--contexts', default='1024 8192 32768 65536 131072')
    p.add_argument('--reserve', type=int, default=1024)
    args = p.parse_args()

    os.environ['TOKENIZER_PATH'] = args.tokenizer_path
    os.environ['PROMPT_TOKEN_RESERVE'] = str(args.reserve)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=True)
    print(f'tokenizer={args.tokenizer_path}')
    print(f'reserve={args.reserve}')
    for ctx in parse_contexts(args.contexts):
        prompt, est, target = make_prompt(ctx)
        actual = len(tok.encode(prompt, add_special_tokens=False))
        print(f'context={ctx}\ttarget_prompt={target}\test={est}\tactual_raw_prompt={actual}\tsafe={actual <= target}')


if __name__ == '__main__':
    main()
