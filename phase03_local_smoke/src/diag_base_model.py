"""
Quick test: base model (no adapter) chunked generation vs answer_only adapter.
Tests if the base model already knows the boxed format, or if the adapter is needed.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import SYSTEM_PROMPT

BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
BOX_APPROX_RE = re.compile(r"\\box(?:ed)?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def build_prompt(tokenizer, row):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["prompt"].strip()}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def gen_chunked(model, tokenizer, input_ids, max_new=100):
    generated = input_ids
    eos_id    = tokenizer.eos_token_id
    for step in range(max_new):
        with torch.no_grad():
            out = model(input_ids=generated, use_cache=False, return_dict=True)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat([generated, nxt], dim=1)
        if eos_id is not None and (nxt == eos_id).all():
            break
    return generated


def run_eval(model, tokenizer, rows, label, max_new=100):
    model.eval()
    sep = "=" * 64
    print(f"\n{sep}\n  {label}\n{sep}")
    n_boxed, n_approx, n_correct = 0, 0, 0
    for i, row in enumerate(rows):
        prompt = build_prompt(tokenizer, row)
        ids    = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out    = gen_chunked(model, tokenizer, ids, max_new=max_new)
        gen    = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        boxed_exact  = BOX_RE.findall(gen)
        boxed_approx = BOX_APPROX_RE.findall(gen)
        ext     = boxed_exact[-1].strip() if boxed_exact else (boxed_approx[-1].strip() if boxed_approx else "")
        gold    = str(row.get("gold_answer", row.get("answer", ""))).strip()
        correct = ext.strip().lower() == gold.strip().lower() if ext else False

        if boxed_exact:
            n_boxed += 1
        if boxed_approx:
            n_approx += 1
        if correct:
            n_correct += 1

        print(f"\n  [{i}] task={row.get('task_type','?')}  gold={gold!r}")
        print(f"       gen: {gen[:200]!r}")
        print(f"       ext: {ext!r}  exact_boxed={bool(boxed_exact)}  approx_boxed={bool(boxed_approx)}  correct={correct}")

    n = len(rows)
    print(f"\n  SUMMARY [{label}]: exact_boxed={n_boxed}/{n}  approx_boxed={n_approx}/{n}  correct={n_correct}/{n}")
    return n_boxed, n_approx, n_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val",         required=True)
    parser.add_argument("--n",           type=int, default=5)
    parser.add_argument("--max-new",     type=int, default=100)
    parser.add_argument("--adapter-dir", default=None, help="If set, test adapter too")
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.val) as f:
        rows = [json.loads(l) for l in f if l.strip()][:args.n]

    print(f"Loading base model: {BASE_MODEL}")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
    )
    model.eval()

    run_eval(model, tokenizer, rows, "BASE MODEL (chunked)", args.max_new)

    if args.adapter_dir:
        from train_lora_v2 import load_adapter_weights
        print(f"\nLoading adapter: {args.adapter_dir}")
        load_adapter_weights(model, args.adapter_dir)
        run_eval(model, tokenizer, rows, f"ADAPTER (chunked)", args.max_new)


if __name__ == "__main__":
    main()
