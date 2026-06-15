"""
Post-training evaluation: chunked (no-cache) generation with per-task breakdown.

Usage:
    python phase03_local_smoke/src/eval_chunked_full.py \
        --adapter-dir phase03_local_smoke/outputs/adapters/full_haiku_9500/final_adapter_haiku_reasoning \
        --val phase02_data_generation/data/merged/val.jsonl \
        --n-per-task 17 --max-new 250 \
        --output-dir phase03_local_smoke/outputs/evals

Comparison baseline (1k-row haiku_reasoning at max_new=250, n=50):
  parse=88% (44/50)  accuracy=12% (6/50)
  roman: correct; numerical tasks: 0-13%
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_lora_v2 import load_adapter_weights, BOX_RE, build_prompt

BASELINE_1K = {
    "label": "haiku_reasoning 1k/3ep max_new=250 n=50",
    "parse_pct": 88.0,
    "accuracy_pct": 12.0,
    "by_task": {
        "roman":           {"parse_pct": 100, "accuracy_pct": 80},
        "unit_conversion": {"parse_pct": 100, "accuracy_pct": 13},
        "gravity":         {"parse_pct": 100, "accuracy_pct":  0},
        "bit_manipulation":{"parse_pct":  50, "accuracy_pct":  0},
        "symbol_transform":{"parse_pct":  83, "accuracy_pct":  0},
        "cipher_text":     {"parse_pct":   0, "accuracy_pct":  0},
    },
}

# V8 local Gemma clean baseline — eval_final_adapter_haiku_reasoning_20260609_155738.json
# adapter: v8_local_gemma_clean/final_adapter_haiku_reasoning, val: merged/val.jsonl, n_per_task=17
BASELINE_V8_LOCAL_GEMMA = {
    "label": "v8_local_gemma_clean haiku_reasoning/5ep max_new=250 n=102",
    "parse_pct": 96.1,
    "accuracy_pct": 24.5,
    "by_task": {
        "roman":           {"parse_pct": 100.0, "accuracy_pct": 100.0},
        "cipher_text":     {"parse_pct": 100.0, "accuracy_pct":  35.3},
        "gravity":         {"parse_pct": 100.0, "accuracy_pct":   0.0},
        "unit_conversion": {"parse_pct": 100.0, "accuracy_pct":   5.9},
        "bit_manipulation":{"parse_pct": 100.0, "accuracy_pct":   5.9},
        "symbol_transform":{"parse_pct":  76.5, "accuracy_pct":   0.0},
    },
}

# Full 9,500-row Haiku dataset baseline — eval_final_adapter_haiku_reasoning_20260604_124340.json
# adapter: full_haiku_9500/final_adapter_haiku_reasoning, val: merged/val.jsonl, n_per_task=17
BASELINE_FULL_HAIKU_9500 = {
    "label": "full_haiku_9500 haiku_reasoning/5ep max_new=250 n=102",
    "parse_pct": 86.3,
    "accuracy_pct": 19.6,
    "by_task": {
        "roman":           {"parse_pct": 100.0, "accuracy_pct": 100.0},
        "cipher_text":     {"parse_pct":  94.1, "accuracy_pct":  17.6},
        "gravity":         {"parse_pct": 100.0, "accuracy_pct":   0.0},
        "unit_conversion": {"parse_pct": 100.0, "accuracy_pct":   0.0},
        "bit_manipulation":{"parse_pct":  64.7, "accuracy_pct":   0.0},
        "symbol_transform":{"parse_pct":  58.8, "accuracy_pct":   0.0},
    },
}


def gen_chunked(model, tokenizer, input_ids, max_new=250):
    generated = input_ids
    eos_id    = tokenizer.eos_token_id
    for _ in range(max_new):
        with torch.no_grad():
            out = model(input_ids=generated, use_cache=False, return_dict=True)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat([generated, nxt], dim=1)
        if eos_id is not None and (nxt == eos_id).all():
            break
    return generated


def stratified_sample(rows, n_per_task):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "unknown")].append(r)
    selected = []
    for task in sorted(by_task):
        task_rows = by_task[task]
        selected.extend(task_rows[:n_per_task])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--val",         required=True)
    parser.add_argument("--n-per-task",  type=int, default=17,
                        help="Rows per task type (6 tasks → N*6 total)")
    parser.add_argument("--max-new",     type=int, default=250)
    parser.add_argument("--output-dir",  default="phase03_local_smoke/outputs/evals")
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer

    print(f"\n{'='*64}")
    print(f"  eval_chunked_full.py")
    print(f"  adapter:    {args.adapter_dir}")
    print(f"  val data:   {args.val}")
    print(f"  n_per_task: {args.n_per_task}  max_new: {args.max_new}")
    print(f"{'='*64}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_val = [json.loads(l) for l in open(args.val) if l.strip()]
    rows    = stratified_sample(all_val, args.n_per_task)
    print(f"Sampled {len(rows)} rows ({args.n_per_task} per task across "
          f"{len(set(r.get('task_type') for r in rows))} task types)\n")

    print(f"Loading adapter from {args.adapter_dir}...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
    )
    load_adapter_weights(model, args.adapter_dir)
    model.eval()

    # --- Run generation ---
    t0 = time.time()
    task_boxed   = defaultdict(int)
    task_correct = defaultdict(int)
    task_total   = defaultdict(int)
    task_examples = defaultdict(lambda: {"correct": [], "incorrect": []})

    total_boxed = 0
    total_correct = 0

    for i, row in enumerate(rows):
        task = row.get("task_type", "unknown")
        gold = str(row.get("gold_answer", row.get("answer", ""))).strip()

        prompt = build_prompt(tokenizer, row)
        ids    = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out    = gen_chunked(model, tokenizer, ids, max_new=args.max_new)
        gen    = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        boxed   = BOX_RE.findall(gen)
        ext     = boxed[-1].strip() if boxed else ""
        correct = ext.strip().lower() == gold.strip().lower() if ext else False

        task_total[task]  += 1
        if boxed:
            task_boxed[task] += 1
            total_boxed += 1
        if correct:
            task_correct[task] += 1
            total_correct += 1

        ex = {"gold": gold, "extracted": ext, "correct": correct,
              "gen_snippet": gen[:300]}
        bucket = "correct" if correct else "incorrect"
        if len(task_examples[task][bucket]) < 2:
            task_examples[task][bucket].append(ex)

        elapsed = time.time() - t0
        print(f"  [{i+1:3d}/{len(rows)}] task={task:<20}  "
              f"gold={gold!r:<25}  ext={ext!r:<20}  "
              f"correct={correct}  elapsed={elapsed:.0f}s")

    elapsed_total = time.time() - t0
    n = len(rows)

    # --- Print summary ---
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  FINAL RESULTS  (n={n}, max_new={args.max_new})")
    print(sep)
    print(f"  parse rate : {total_boxed}/{n}  ({100*total_boxed/n:.1f}%)")
    print(f"  accuracy   : {total_correct}/{n}  ({100*total_correct/n:.1f}%)")
    print(f"  elapsed    : {elapsed_total/60:.1f} min")
    print(f"\n  Per-task breakdown:")
    print(f"  {'task':<20}  {'parse':>12}  {'accuracy':>12}  Δvs_haiku9500  Δvs_v8gemma")
    for task in sorted(task_total):
        nt = task_total[task]
        bp = 100*task_boxed[task]/nt
        ap = 100*task_correct[task]/nt
        b1 = BASELINE_FULL_HAIKU_9500["by_task"].get(task, {})
        b2 = BASELINE_V8_LOCAL_GEMMA["by_task"].get(task, {})
        d1 = f"{ap - b1['accuracy_pct']:+.0f}%" if "accuracy_pct" in b1 else "—"
        d2 = f"{ap - b2['accuracy_pct']:+.0f}%" if "accuracy_pct" in b2 else "—"
        print(f"  {task:<20}  {task_boxed[task]:>4}/{nt} ({bp:>5.1f}%)  "
              f"{task_correct[task]:>4}/{nt} ({ap:>5.1f}%)  "
              f"{d1:>13}  {d2:>11}")
    print(f"\n  Baseline comparison:")
    print(f"    haiku_9500  parse={BASELINE_FULL_HAIKU_9500['parse_pct']}%  "
          f"accuracy={BASELINE_FULL_HAIKU_9500['accuracy_pct']}%")
    print(f"    v8_gemma    parse={BASELINE_V8_LOCAL_GEMMA['parse_pct']}%  "
          f"accuracy={BASELINE_V8_LOCAL_GEMMA['accuracy_pct']}%")
    print(f"    this run    parse={100*total_boxed/n:.1f}%  "
          f"accuracy={100*total_correct/n:.1f}%  "
          f"(Δ vs v8_gemma: acc={100*total_correct/n - BASELINE_V8_LOCAL_GEMMA['accuracy_pct']:+.1f}%)")
    print(sep)

    # --- Per-task examples ---
    print(f"\n  EXAMPLES BY TASK:")
    for task in sorted(task_examples):
        print(f"\n  [{task}]")
        for bucket in ("correct", "incorrect"):
            for ex in task_examples[task][bucket]:
                marker = "CORRECT  " if bucket == "correct" else "INCORRECT"
                print(f"    {marker}  gold={ex['gold']!r}  ext={ex['extracted']!r}")
                print(f"             gen: {ex['gen_snippet'][:200]!r}")

    # --- Save JSON ---
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    adapter_name = Path(args.adapter_dir).name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"eval_{adapter_name}_{ts}.json"

    result = {
        "adapter_dir":    args.adapter_dir,
        "val_data":       args.val,
        "n_per_task":     args.n_per_task,
        "max_new":        args.max_new,
        "timestamp":      ts,
        "n_total":        n,
        "elapsed_min":    round(elapsed_total / 60, 2),
        "overall": {
            "boxed":        total_boxed,
            "boxed_pct":    round(100*total_boxed/n, 1),
            "correct":      total_correct,
            "accuracy_pct": round(100*total_correct/n, 1),
        },
        "by_task": {
            task: {
                "n":           task_total[task],
                "boxed":       task_boxed[task],
                "boxed_pct":   round(100*task_boxed[task]/task_total[task], 1),
                "correct":     task_correct[task],
                "accuracy_pct": round(100*task_correct[task]/task_total[task], 1),
            }
            for task in sorted(task_total)
        },
        "examples": {
            task: task_examples[task]
            for task in sorted(task_examples)
        },
        "baseline_1k": BASELINE_1K,
        "baseline_full_haiku_9500": BASELINE_FULL_HAIKU_9500,
        "baseline_v8_local_gemma": BASELINE_V8_LOCAL_GEMMA,
    }

    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
