"""
Diagnostic: chunked (no-cache) generation vs cached generation for NemotronH LoRA.

Hypothesis: sequential `selective_state_update` diverges from training path even for
Attn+MLP-only adapters; chunked `mamba_chunk_scan_combined` stays consistent.

Usage:
    python phase03_local_smoke/src/diag_chunked_gen.py \
        --adapter-dir phase03_local_smoke/outputs/adapters/variants_1k/answer_only/final_adapter_answer_only \
        --val phase02_data_generation/data/merged/val.jsonl \
        --n 10
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import SYSTEM_PROMPT
from train_lora_v2 import (
    load_adapter_weights, _fix_adapter_key_names,
    _patch_hybrid_cache, _patch_block_forward, _init_cache
)

BOX_RE = re.compile(r"\\box(?:ed)?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def build_prompt(tokenizer, row):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["prompt"].strip()}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def gen_cached(model, tokenizer, input_ids, max_new=150):
    """Original cached generation with selective_state_update at step 1+."""
    _patch_hybrid_cache(model)
    _patch_block_forward()
    cache     = _init_cache(model)
    generated = input_ids
    eos_id    = tokenizer.eos_token_id

    for step in range(max_new):
        cur = generated if step == 0 else generated[:, -1:]
        pos = (torch.arange(generated.shape[1], device=generated.device)
               if step == 0 else
               torch.tensor([generated.shape[1] - 1], device=generated.device))
        with torch.no_grad():
            out = model(input_ids=cur, cache_params=cache,
                        cache_position=pos, use_cache=True, return_dict=True)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat([generated, nxt], dim=1)
        if eos_id is not None and (nxt == eos_id).all():
            break
    return generated


def gen_chunked(model, tokenizer, input_ids, max_new=150):
    """Chunked (no-cache) generation. Re-processes full sequence every step.
    Always uses mamba_chunk_scan_combined (BF16) — no selective_state_update.
    O(n^2) tokens but avoids FP32/BF16 path mismatch."""
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


def run_eval(model, tokenizer, rows, gen_fn, label, max_new=150):
    model.eval()
    sep = "=" * 64
    print(f"\n{sep}\n  {label}\n{sep}")
    n_boxed, n_correct = 0, 0
    for i, row in enumerate(rows):
        prompt = build_prompt(tokenizer, row)
        ids    = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out    = gen_fn(model, tokenizer, ids, max_new=max_new)
        gen    = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        boxed   = BOX_RE.findall(gen)
        ext     = boxed[-1].strip() if boxed else ""
        gold    = str(row.get("gold_answer", row.get("answer", ""))).strip()
        correct = ext.strip().lower() == gold.strip().lower() if ext else False

        if boxed:
            n_boxed += 1
        if correct:
            n_correct += 1

        print(f"\n  [{i}] task={row.get('task_type','?')}  gold={gold!r}")
        print(f"       gen: {gen[:200]!r}")
        print(f"       ext: {ext!r}  boxed={bool(boxed)}  correct={correct}")

    n = len(rows)
    print(f"\n  SUMMARY [{label}]: boxed={n_boxed}/{n}  correct={n_correct}/{n}")
    return n_boxed, n_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--val",         required=True)
    parser.add_argument("--n",           type=int, default=10)
    parser.add_argument("--max-new",     type=int, default=150)
    parser.add_argument("--test-base",   action="store_true",
                        help="Also test base model (no adapter) as control")
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.val) as f:
        rows = [json.loads(l) for l in f if l.strip()][:args.n]

    print(f"Loading adapter: {args.adapter_dir}")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
    )
    load_adapter_weights(model, args.adapter_dir)
    model.eval()

    print(f"\nTesting {len(rows)} samples with max_new={args.max_new}")

    # Test 1: Cached generation (original, broken)
    b1, c1 = run_eval(model, tokenizer, rows, gen_cached, "CACHED (original)", args.max_new)

    # Test 2: Chunked generation (no SSM cache)
    b2, c2 = run_eval(model, tokenizer, rows, gen_chunked, "CHUNKED (no SSM cache)", args.max_new)

    print("\n" + "=" * 64)
    print("  COMPARISON SUMMARY")
    print("=" * 64)
    print(f"  Cached   : boxed={b1}/{len(rows)}  correct={c1}/{len(rows)}")
    print(f"  Chunked  : boxed={b2}/{len(rows)}  correct={c2}/{len(rows)}")
    print("=" * 64)

    if args.test_base:
        print("\nLoading base model (no adapter)...")
        model_base, _ = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=2048,
            load_in_4bit=False,
            dtype=None,
            trust_remote_code=True,
        )
        model_base.eval()
        b3, c3 = run_eval(model_base, tokenizer, rows, gen_chunked, "BASE MODEL (chunked)", args.max_new)
        b4, c4 = run_eval(model_base, tokenizer, rows, gen_cached,  "BASE MODEL (cached)", args.max_new)
        print(f"\n  Base chunked: boxed={b3}/{len(rows)}  correct={c3}/{len(rows)}")
        print(f"  Base cached : boxed={b4}/{len(rows)}  correct={c4}/{len(rows)}")


if __name__ == "__main__":
    main()
