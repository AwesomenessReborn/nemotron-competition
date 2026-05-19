"""
Direct inference eval for a LoRA-adapted Nemotron-H model.
No vLLM server required — loads model + adapter and runs inference locally.

Usage:
    python phase03_local_smoke/src/eval_direct.py \
        --adapter-dir phase03_local_smoke/outputs/adapters/local_4b/final_adapter \
        --val phase02_data_generation/data/merged/val.jsonl \
        --out phase03_local_smoke/outputs/evals/eval_results.json \
        [--n 100]       # optional: evaluate only N samples (default: all)
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import SYSTEM_PROMPT

BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_answer(text: str) -> str:
    boxed = BOX_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip().splitlines()[-1].strip()


def score(pred: str, gold: str, task_type: str) -> bool:
    pred, gold = pred.strip(), str(gold).strip()
    if task_type == "roman":
        return pred.upper() == gold.upper()
    if task_type in ("gravity", "unit_conversion"):
        try:
            return abs(float(pred) - float(gold)) / max(abs(float(gold)), 1) < 0.01
        except ValueError:
            return pred == gold
    return pred == gold


def load_val(path, n):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if n and n < len(rows):
        # Sample evenly across task types
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_type"]].append(r)
        per_task = n // len(by_task)
        sampled = []
        for tt in sorted(by_task):
            sampled.extend(by_task[tt][:per_task])
        rows = sampled[:n]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--val",         required=True)
    parser.add_argument("--out",         required=True)
    parser.add_argument("--n", type=int, default=None, help="Max eval samples (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    print(f"\nLoading adapter from: {args.adapter_dir}")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
    )
    # Merge LoRA weights into base model so generate() uses the native
    # Nemotron-H path (including NemotronHHybridDynamicCache init).
    # The PEFT wrapper's generate() doesn't initialize that cache.
    print("Merging LoRA weights into base model...", flush=True)
    model = model.merge_and_unload()
    model.eval()

    # transformers 5.5.0 passes an empty DynamicCache (not None) as past_key_values
    # on the first _prefill call, so prepare_inputs_for_generation sees
    # empty_past_kv=False and tries cache_position[-1] which is None → crash.
    # Patch: reset past_key_values to None when cache_position is absent so the
    # model's else-branch runs and initializes HybridMambaAttentionDynamicCache.
    _orig_prep = model.prepare_inputs_for_generation
    def _patched_prep(input_ids, past_key_values=None, cache_position=None, **kwargs):
        if cache_position is None:
            past_key_values = None
        return _orig_prep(input_ids, past_key_values=past_key_values,
                          cache_position=cache_position, **kwargs)
    model.prepare_inputs_for_generation = _patched_prep

    print(f"Loaded + merged in {time.time() - t0:.1f}s\n", flush=True)

    rows = load_val(args.val, args.n)
    print(f"Evaluating {len(rows)} samples...\n")

    results = []
    by_task = defaultdict(lambda: {"correct": 0, "total": 0, "boxed": 0})

    for i, row in enumerate(rows, 1):
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": row["prompt"].strip()},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[{i:03d}/{len(rows)}] OOM — skipping sample", flush=True)
            output_text = ""
        finally:
            torch.cuda.empty_cache()

        pred = extract_answer(output_text)
        gold = str(row.get("gold_answer", row.get("answer", ""))).strip()
        task = row.get("task_type", "unknown")
        is_correct = score(pred, gold, task)
        has_boxed = bool(BOX_RE.search(output_text))

        by_task[task]["total"] += 1
        by_task[task]["correct"] += int(is_correct)
        by_task[task]["boxed"] += int(has_boxed)

        tick = "✓" if is_correct else "✗"
        print(f"[{i:03d}/{len(rows)}] {task:<22} {tick} | gold={gold!r:>12} pred={pred!r}", flush=True)

        results.append({
            "id":         row.get("id"),
            "task_type":  task,
            "gold":       gold,
            "pred":       pred,
            "correct":    is_correct,
            "has_boxed":  has_boxed,
            "output":     output_text,
        })

    total_correct = sum(r["correct"] for r in results)
    total_boxed   = sum(r["has_boxed"] for r in results)
    accuracy      = total_correct / len(results)
    boxed_rate    = total_boxed / len(results)

    print(f"\n{'='*60}")
    print(f"Overall accuracy: {total_correct}/{len(results)}  {accuracy:.1%}")
    print(f"Boxed rate:       {total_boxed}/{len(results)}  {boxed_rate:.1%}")
    print(f"\nPer-task accuracy:")
    for task in sorted(by_task):
        v = by_task[task]
        print(f"  {task:<22} {v['correct']}/{v['total']}  {v['correct']/v['total']:.1%}  (boxed {v['boxed']}/{v['total']})")

    summary = {
        "adapter_dir":    args.adapter_dir,
        "n_samples":      len(results),
        "accuracy":       round(accuracy, 4),
        "boxed_rate":     round(boxed_rate, 4),
        "by_task": {
            task: {
                "accuracy":   round(v["correct"] / v["total"], 4),
                "boxed_rate": round(v["boxed"] / v["total"], 4),
                "n":          v["total"],
            }
            for task, v in by_task.items()
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
