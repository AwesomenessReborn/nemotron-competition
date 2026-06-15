#!/usr/bin/env python3
"""
Controlled same-eval-set comparison between V8 and V8.1 adapters.

Fixes the stratified 102-sample eval set once, then evaluates both adapters
on identical rows so results are directly comparable.

Outputs:
  phase03_local_smoke/outputs/evals/v8_vs_v8_1_same_eval_ids.json
  phase03_local_smoke/outputs/evals/v8_same_v8_1_eval_results.json
  phase03_local_smoke/outputs/evals/v8_1_same_eval_results.json
  phase03_local_smoke/outputs/evals/v8_vs_v8_1_same_eval_report.md

Run from project root:
  python phase03_local_smoke/src/eval_compare_v8_v8_1.py
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_lora_v2 import load_adapter_weights, BOX_RE, build_prompt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

V8_ADAPTER   = "phase03_local_smoke/outputs/adapters/v8_local_gemma_clean/final_adapter_haiku_reasoning"
V81_ADAPTER  = "phase03_local_smoke/outputs/adapters/v8_1_local_gemma_clean/final_adapter_haiku_reasoning"
VAL_DATA     = "phase02_data_generation/data/v8/val_reasoning_v8_1_local_gemma_clean.jsonl"

N_PER_TASK   = 17
MAX_NEW      = 250
OUTPUT_DIR   = Path("phase03_local_smoke/outputs/evals")

REPAIR_PHRASES = [
    "the correct symbol sequence is provided",
    "copy it exactly",
    "post-hoc training trace",
    "i copy it",
    "so i copy",
]

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def gen_chunked(model, tokenizer, input_ids, max_new=250):
    generated = input_ids
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new):
        with torch.no_grad():
            out = model(input_ids=generated, use_cache=False, return_dict=True)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat([generated, nxt], dim=1)
        if eos_id is not None and (nxt == eos_id).all():
            break
    return generated

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stratified_sample(rows, n_per_task):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "unknown")].append(r)
    selected = []
    for task in sorted(by_task):
        selected.extend(by_task[task][:n_per_task])
    return selected

# ---------------------------------------------------------------------------
# Repair phrase scan
# ---------------------------------------------------------------------------

def check_repair_phrase(text):
    lower = text.lower()
    for phrase in REPAIR_PHRASES:
        if phrase in lower:
            return phrase
    return None

# ---------------------------------------------------------------------------
# Evaluate one adapter on fixed rows
# ---------------------------------------------------------------------------

def evaluate_adapter(adapter_dir, rows, tokenizer, label):
    from unsloth import FastLanguageModel

    print(f"\n{'='*64}")
    print(f"  Evaluating: {label}")
    print(f"  Adapter:    {adapter_dir}")
    print(f"  Rows:       {len(rows)}")
    print(f"{'='*64}\n")

    print(f"Loading adapter...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
    )
    load_adapter_weights(model, adapter_dir)
    model.eval()

    t0 = time.time()
    task_boxed    = defaultdict(int)
    task_correct  = defaultdict(int)
    task_total    = defaultdict(int)
    task_examples = defaultdict(lambda: {"correct": [], "incorrect": []})
    repair_hits   = []

    all_results = []
    total_boxed = 0
    total_correct = 0

    for i, row in enumerate(rows):
        task = row.get("task_type", "unknown")
        gold = str(row.get("gold_answer", row.get("answer", ""))).strip()
        row_id = str(row.get("id", ""))

        prompt = build_prompt(tokenizer, row)
        ids    = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out    = gen_chunked(model, tokenizer, ids, max_new=MAX_NEW)
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

        # Repair phrase check
        repair = check_repair_phrase(gen)
        if repair:
            repair_hits.append({"id": row_id, "task": task, "phrase": repair,
                                  "gen_snippet": gen[:200]})

        row_result = {
            "id":        row_id,
            "task":      task,
            "gold":      gold,
            "extracted": ext,
            "correct":   correct,
            "boxed":     bool(boxed),
            "gen":       gen[:400],
        }
        all_results.append(row_result)

        ex = {"id": row_id, "gold": gold, "extracted": ext, "correct": correct,
              "gen_snippet": gen[:300]}
        bucket = "correct" if correct else "incorrect"
        if len(task_examples[task][bucket]) < 3:
            task_examples[task][bucket].append(ex)

        elapsed = time.time() - t0
        print(f"  [{i+1:3d}/{len(rows)}] {task:<20}  gold={gold!r:<20}  "
              f"ext={ext!r:<20}  ok={correct}  {elapsed:.0f}s")

    elapsed_total = time.time() - t0
    n = len(rows)

    # Print summary
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  RESULTS — {label}")
    print(f"  parse    : {total_boxed}/{n}  ({100*total_boxed/n:.1f}%)")
    print(f"  accuracy : {total_correct}/{n}  ({100*total_correct/n:.1f}%)")
    print(f"  elapsed  : {elapsed_total/60:.1f} min")
    print(f"\n  Per-task:")
    print(f"  {'task':<20}  {'parse':>10}  {'accuracy':>10}")
    for task in sorted(task_total):
        nt = task_total[task]
        print(f"  {task:<20}  {task_boxed[task]:>3}/{nt} ({100*task_boxed[task]/nt:>5.1f}%)  "
              f"{task_correct[task]:>3}/{nt} ({100*task_correct[task]/nt:>5.1f}%)")
    if repair_hits:
        print(f"\n  !! Repair phrase hits: {len(repair_hits)}")
        for h in repair_hits[:3]:
            print(f"     id={h['id']}  task={h['task']}  phrase={h['phrase']!r}")
    else:
        print(f"\n  Repair phrase hits: 0 (clean)")
    print(sep)

    # Free model memory before loading next adapter
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label":         label,
        "adapter_dir":   adapter_dir,
        "val_data":      VAL_DATA,
        "n_per_task":    N_PER_TASK,
        "max_new":       MAX_NEW,
        "timestamp":     datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_total":       n,
        "elapsed_min":   round(elapsed_total / 60, 2),
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
        "repair_phrase_hits": repair_hits,
        "examples": {task: dict(task_examples[task]) for task in sorted(task_examples)},
        "all_results": all_results,
    }

# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def write_report(r8, r81, eval_ids, out_path):
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    def fmt_delta(v81, v8):
        d = v81 - v8
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}%"

    lines = [
        "# V8 vs V8.1 — Controlled Same-Eval-Set Comparison",
        "",
        f"**Generated:** {ts}",
        f"**Eval source:** `{VAL_DATA}`",
        f"**Sample:** {len(eval_ids)} rows (17 per task, 6 tasks) — fixed IDs, both adapters identical input",
        f"**max_new:** {MAX_NEW}  |  **Generation:** chunked (no-cache)",
        "",
        "## Overall Results",
        "",
        f"| Metric | V8 | V8.1 | Δ (V8.1 − V8) |",
        f"|--------|-----|------|----------------|",
        f"| Parse % | {r8['overall']['boxed_pct']}% | {r81['overall']['boxed_pct']}% "
        f"| {fmt_delta(r81['overall']['boxed_pct'], r8['overall']['boxed_pct'])} |",
        f"| Accuracy % | {r8['overall']['accuracy_pct']}% | {r81['overall']['accuracy_pct']}% "
        f"| {fmt_delta(r81['overall']['accuracy_pct'], r8['overall']['accuracy_pct'])} |",
        "",
        "## Per-Task Results",
        "",
        "| Task | V8 parse | V8.1 parse | Δparse | V8 acc | V8.1 acc | Δacc |",
        "|------|----------|-----------|--------|--------|----------|------|",
    ]

    for task in sorted(r8["by_task"]):
        t8  = r8["by_task"][task]
        t81 = r81["by_task"].get(task, {})
        p8   = t8["boxed_pct"]
        p81  = t81.get("boxed_pct", 0)
        a8   = t8["accuracy_pct"]
        a81  = t81.get("accuracy_pct", 0)
        lines.append(
            f"| {task} | {p8}% | {p81}% | {fmt_delta(p81, p8)} "
            f"| {a8}% | {a81}% | {fmt_delta(a81, a8)} |"
        )

    # Repair phrase section
    lines += [
        "",
        "## Repair Phrase Leakage Check",
        "",
        f"| Adapter | Hits |",
        f"|---------|------|",
        f"| V8 | {len(r8['repair_phrase_hits'])} |",
        f"| V8.1 | {len(r81['repair_phrase_hits'])} |",
    ]
    if r8["repair_phrase_hits"]:
        lines += ["", "**V8 hits:**"]
        for h in r8["repair_phrase_hits"][:5]:
            lines.append(f"- `{h['id']}` ({h['task']}): `{h['phrase']}`")
            lines.append(f"  > {h['gen_snippet'][:150]}")
    if r81["repair_phrase_hits"]:
        lines += ["", "**V8.1 hits:**"]
        for h in r81["repair_phrase_hits"][:5]:
            lines.append(f"- `{h['id']}` ({h['task']}): `{h['phrase']}`")
            lines.append(f"  > {h['gen_snippet'][:150]}")

    # Top 10 failures per adapter
    for label, r in [("V8", r8), ("V8.1", r81)]:
        failures = [x for x in r["all_results"] if not x["correct"]][:10]
        lines += [
            "",
            f"## Top 10 Failure Examples — {label}",
            "",
            f"| # | task | gold | extracted | gen snippet |",
            f"|---|------|------|-----------|-------------|",
        ]
        for i, f in enumerate(failures, 1):
            snippet = f["gen"][:80].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {i} | {f['task']} | `{f['gold'][:20]}` "
                f"| `{f['extracted'][:20]}` | {snippet} |"
            )

    # Per-task sample failures for symbol_transform and gravity
    for task in ("symbol_transform", "gravity", "cipher_text"):
        lines += ["", f"## {task} — Side-by-Side Examples", ""]
        for label, r in [("V8", r8), ("V8.1", r81)]:
            exs = r["examples"].get(task, {})
            lines += [f"**{label}:**"]
            for ex in exs.get("correct", [])[:2]:
                lines.append(f"- CORRECT gold=`{ex['gold']}` → `{ex['extracted']}`")
                lines.append(f"  > {ex['gen_snippet'][:200]}")
            for ex in exs.get("incorrect", [])[:3]:
                lines.append(f"- WRONG   gold=`{ex['gold']}` → `{ex['extracted']}`")
                lines.append(f"  > {ex['gen_snippet'][:200]}")
            lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written → {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load val data and fix eval sample ----------------------------------
    print("Loading val data and fixing eval sample...")
    all_val = [json.loads(l) for l in open(VAL_DATA) if l.strip()]
    rows    = stratified_sample(all_val, N_PER_TASK)
    print(f"  Fixed {len(rows)} rows ({N_PER_TASK} per task across "
          f"{len(set(r.get('task_type') for r in rows))} task types)")

    # Save eval IDs
    eval_ids = [str(r.get("id", "")) for r in rows]
    ids_by_task = defaultdict(list)
    for r in rows:
        ids_by_task[r.get("task_type", "unknown")].append(str(r.get("id", "")))
    ids_path = OUTPUT_DIR / "v8_vs_v8_1_same_eval_ids.json"
    with open(ids_path, "w") as f:
        json.dump({"n_total": len(eval_ids), "n_per_task": N_PER_TASK,
                   "val_source": VAL_DATA, "ids": eval_ids,
                   "ids_by_task": dict(ids_by_task)}, f, indent=2)
    print(f"  Eval IDs saved → {ids_path}")

    # Load tokenizer once (both adapters share the same base tokenizer)
    print(f"\nLoading tokenizer from {V8_ADAPTER}...")
    tokenizer = AutoTokenizer.from_pretrained(V8_ADAPTER, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Evaluate V8 --------------------------------------------------------
    v8_path = OUTPUT_DIR / "v8_same_v8_1_eval_results.json"
    if v8_path.exists():
        print(f"\nV8 results already exist — loading from {v8_path}")
        with open(v8_path) as f:
            r8 = json.load(f)
        print(f"  V8: parse={r8['overall']['boxed_pct']}%  acc={r8['overall']['accuracy_pct']}%")
    else:
        r8 = evaluate_adapter(V8_ADAPTER, rows, tokenizer, label="V8 (v8_local_gemma_clean)")
        with open(v8_path, "w") as f:
            json.dump(r8, f, indent=2)
        print(f"V8 results saved → {v8_path}")

    # ---- Evaluate V8.1 -------------------------------------------------------
    r81 = evaluate_adapter(V81_ADAPTER, rows, tokenizer, label="V8.1 (v8_1_local_gemma_clean)")

    v81_path = OUTPUT_DIR / "v8_1_same_eval_results.json"
    with open(v81_path, "w") as f:
        json.dump(r81, f, indent=2)
    print(f"V8.1 results saved → {v81_path}")

    # ---- Comparison report ---------------------------------------------------
    report_path = OUTPUT_DIR / "v8_vs_v8_1_same_eval_report.md"
    write_report(r8, r81, eval_ids, report_path)

    # ---- Console summary ----------------------------------------------------
    sep = "=" * 64
    print(f"\n{sep}")
    print("  FINAL COMPARISON (same 102 rows, both adapters)")
    print(sep)
    print(f"  {'Metric':<25}  {'V8':>8}  {'V8.1':>8}  {'Δ':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Parse %':<25}  {r8['overall']['boxed_pct']:>7.1f}%  "
          f"{r81['overall']['boxed_pct']:>7.1f}%  "
          f"{r81['overall']['boxed_pct']-r8['overall']['boxed_pct']:>+7.1f}%")
    print(f"  {'Accuracy %':<25}  {r8['overall']['accuracy_pct']:>7.1f}%  "
          f"{r81['overall']['accuracy_pct']:>7.1f}%  "
          f"{r81['overall']['accuracy_pct']-r8['overall']['accuracy_pct']:>+7.1f}%")
    print()
    for task in sorted(r8["by_task"]):
        a8  = r8["by_task"][task]["accuracy_pct"]
        a81 = r81["by_task"].get(task, {}).get("accuracy_pct", 0)
        print(f"  {'acc_'+task:<25}  {a8:>7.1f}%  {a81:>7.1f}%  {a81-a8:>+7.1f}%")
    print()
    print(f"  Repair phrase hits — V8: {len(r8['repair_phrase_hits'])}  "
          f"V8.1: {len(r81['repair_phrase_hits'])}")
    print(sep)
    print("\nOutputs:")
    for p in [ids_path, v8_path, v81_path, report_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
