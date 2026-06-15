#!/usr/bin/env python3
"""
Controlled eval of V8.3-lite adapter on the same 102 fixed rows used for V8/V8.1/V8.2.
Loads existing V8, V8.1, V8.2 results, runs V8.3-lite, writes 4-way comparison report.

Outputs (max_new=250):
  phase03_local_smoke/outputs/evals/v8_3_lite_controlled_max250_results.json
  phase03_local_smoke/outputs/evals/v8_3_lite_controlled_max250_report.md

Run from project root:
  python phase03_local_smoke/src/eval_v8_3_controlled.py
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

V83_ADAPTER = "phase03_local_smoke/outputs/adapters/v8_3_solver_lite_clean/final_adapter_haiku_reasoning"
VAL_DATA    = "phase02_data_generation/data/v8/val_reasoning_v8_1_local_gemma_clean.jsonl"
IDS_PATH    = "phase03_local_smoke/outputs/evals/v8_vs_v8_1_same_eval_ids.json"
V8_RESULTS  = "phase03_local_smoke/outputs/evals/v8_same_v8_1_eval_results.json"
V81_RESULTS = "phase03_local_smoke/outputs/evals/v8_1_same_eval_results.json"
V82_RESULTS = "phase03_local_smoke/outputs/evals/v8_2_same_eval_results_maxnew500.json"

MAX_NEW    = 250
OUTPUT_DIR = Path("phase03_local_smoke/outputs/evals")

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
# Repair phrase check
# ---------------------------------------------------------------------------

def check_repair_phrase(text):
    lower = text.lower()
    for phrase in REPAIR_PHRASES:
        if phrase in lower:
            return phrase
    return None

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate_v83(rows, tokenizer):
    from unsloth import FastLanguageModel

    print(f"\n{'='*64}")
    print(f"  Evaluating: V8.3-lite (v8_3_solver_lite_clean)")
    print(f"  Adapter:    {V83_ADAPTER}")
    print(f"  Rows:       {len(rows)}")
    print(f"  max_new:    {MAX_NEW}")
    print(f"{'='*64}\n")

    model, _ = FastLanguageModel.from_pretrained(
        model_name=V83_ADAPTER,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
    )
    load_adapter_weights(model, V83_ADAPTER)
    model.eval()

    t0 = time.time()
    task_boxed    = defaultdict(int)
    task_correct  = defaultdict(int)
    task_total    = defaultdict(int)
    task_examples = defaultdict(lambda: {"correct": [], "incorrect": []})
    repair_hits   = []
    all_results   = []
    total_boxed = total_correct = 0

    for i, row in enumerate(rows):
        task  = row.get("task_type", "unknown")
        gold  = str(row.get("gold_answer", row.get("answer", ""))).strip()
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

        repair = check_repair_phrase(gen)
        if repair:
            repair_hits.append({"id": row_id, "task": task, "phrase": repair,
                                 "gen_snippet": gen[:200]})

        row_result = {
            "id": row_id, "task": task, "gold": gold,
            "extracted": ext, "correct": correct, "boxed": bool(boxed),
            "gen": gen[:800],
        }
        all_results.append(row_result)

        ex = {"id": row_id, "gold": gold, "extracted": ext, "correct": correct,
              "gen_snippet": gen[:600]}
        bucket = "correct" if correct else "incorrect"
        if len(task_examples[task][bucket]) < 3:
            task_examples[task][bucket].append(ex)

        elapsed = time.time() - t0
        print(f"  [{i+1:3d}/{len(rows)}] {task:<20}  gold={gold!r:<20}  "
              f"ext={ext!r:<20}  ok={correct}  {elapsed:.0f}s")

    elapsed_total = time.time() - t0
    n = len(rows)

    print(f"\n{'='*64}")
    print(f"  RESULTS — V8.3-lite")
    print(f"  parse    : {total_boxed}/{n}  ({100*total_boxed/n:.1f}%)")
    print(f"  accuracy : {total_correct}/{n}  ({100*total_correct/n:.1f}%)")
    print(f"  elapsed  : {elapsed_total/60:.1f} min")
    print(f"\n  Per-task:")
    for task in sorted(task_total):
        nt = task_total[task]
        print(f"  {task:<20}  parse={task_boxed[task]:>3}/{nt} ({100*task_boxed[task]/nt:>5.1f}%)  "
              f"acc={task_correct[task]:>3}/{nt} ({100*task_correct[task]/nt:>5.1f}%)")
    if repair_hits:
        print(f"\n  !! Repair phrase hits: {len(repair_hits)}")
    else:
        print(f"\n  Repair phrase hits: 0 (clean)")
    print(f"{'='*64}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label":        "V8.3-lite (v8_3_solver_lite_clean)",
        "adapter_dir":  V83_ADAPTER,
        "val_data":     VAL_DATA,
        "max_new":      MAX_NEW,
        "timestamp":    datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_total":      n,
        "elapsed_min":  round(elapsed_total / 60, 2),
        "overall": {
            "boxed":        total_boxed,
            "boxed_pct":    round(100*total_boxed/n, 1),
            "correct":      total_correct,
            "accuracy_pct": round(100*total_correct/n, 1),
        },
        "by_task": {
            task: {
                "n":            task_total[task],
                "boxed":        task_boxed[task],
                "boxed_pct":    round(100*task_boxed[task]/task_total[task], 1),
                "correct":      task_correct[task],
                "accuracy_pct": round(100*task_correct[task]/task_total[task], 1),
            }
            for task in sorted(task_total)
        },
        "repair_phrase_hits": repair_hits,
        "examples":    {task: dict(task_examples[task]) for task in sorted(task_examples)},
        "all_results": all_results,
    }

# ---------------------------------------------------------------------------
# 4-way comparison report
# ---------------------------------------------------------------------------

def write_4way_report(r8, r81, r82, r83, eval_ids, out_path):
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    def d(new, base):
        diff = new - base
        return f"{'+' if diff >= 0 else ''}{diff:.1f}%"

    lines = [
        "# V8 vs V8.1 vs V8.2 vs V8.3-lite — Controlled Same-Eval-Set Comparison",
        "",
        f"**Generated:** {ts}",
        f"**Eval source:** `{VAL_DATA}`",
        f"**Sample:** {len(eval_ids)} rows (17 per task, 6 tasks) — fixed IDs across all adapters",
        f"**max_new (V8.3-lite):** {MAX_NEW}  |  **Generation:** chunked (no-cache)",
        f"**V8.2 ref:** max_new=500 (from prior controlled run)",
        "",
        "## Overall Results",
        "",
        "| Metric | V8 | V8.1 | V8.2 | V8.3-lite | Δ(V8.3−V8.1) |",
        "|--------|----|------|------|-----------|--------------|",
        f"| Parse % | {r8['overall']['boxed_pct']}% | {r81['overall']['boxed_pct']}% "
        f"| {r82['overall']['boxed_pct']}% | {r83['overall']['boxed_pct']}% "
        f"| {d(r83['overall']['boxed_pct'], r81['overall']['boxed_pct'])} |",
        f"| Accuracy % | {r8['overall']['accuracy_pct']}% | {r81['overall']['accuracy_pct']}% "
        f"| {r82['overall']['accuracy_pct']}% | {r83['overall']['accuracy_pct']}% "
        f"| {d(r83['overall']['accuracy_pct'], r81['overall']['accuracy_pct'])} |",
        "",
        "## Per-Task Results",
        "",
        "| Task | V8 acc | V8.1 acc | V8.2 acc | V8.3-lite acc | V8.3-lite parse | Δ(V8.3−V8.1) |",
        "|------|--------|----------|----------|---------------|-----------------|--------------|",
    ]

    for task in sorted(r8["by_task"]):
        t8  = r8["by_task"][task]
        t81 = r81["by_task"].get(task, {})
        t82 = r82["by_task"].get(task, {})
        t83 = r83["by_task"].get(task, {})
        lines.append(
            f"| {task} | {t8['accuracy_pct']}% | {t81.get('accuracy_pct',0)}% "
            f"| {t82.get('accuracy_pct',0)}% | {t83.get('accuracy_pct',0)}% "
            f"| {t83.get('boxed_pct',0)}% "
            f"| {d(t83.get('accuracy_pct',0), t81.get('accuracy_pct',0))} |"
        )

    # Repair phrase section
    lines += [
        "",
        "## Repair Phrase Leakage Check",
        "",
        "| Adapter | Hits |",
        "|---------|------|",
        f"| V8 | {len(r8['repair_phrase_hits'])} |",
        f"| V8.1 | {len(r81['repair_phrase_hits'])} |",
        f"| V8.2 | {len(r82['repair_phrase_hits'])} |",
        f"| V8.3-lite | {len(r83['repair_phrase_hits'])} |",
    ]

    # Top 10 failures for V8.3-lite
    failures = [x for x in r83["all_results"] if not x["correct"]][:10]
    lines += [
        "",
        "## Top 10 Failure Examples — V8.3-lite",
        "",
        "| # | task | gold | extracted | gen snippet |",
        "|---|------|------|-----------|-------------|",
    ]
    for i, f in enumerate(failures, 1):
        snippet = f["gen"][:100].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | {f['task']} | `{f['gold'][:20]}` "
            f"| `{f['extracted'][:20]}` | {snippet} |"
        )

    # Deep dive per task: gravity, unit_conversion, then others
    for task in ("gravity", "unit_conversion", "bit_manipulation", "roman",
                 "cipher_text", "symbol_transform"):
        lines += ["", f"## {task} — Examples", ""]
        for label, r in [("V8.1", r81), ("V8.2", r82), ("V8.3-lite", r83)]:
            exs = r["examples"].get(task, {})
            if not exs:
                continue
            lines += [f"**{label}:**"]
            for ex in exs.get("correct", [])[:2]:
                lines.append(f"- CORRECT  gold=`{ex['gold']}` → `{ex['extracted']}`")
                lines.append(f"  > {ex['gen_snippet'][:400]}")
            for ex in exs.get("incorrect", [])[:3]:
                lines.append(f"- WRONG    gold=`{ex['gold']}` → `{ex['extracted']}`")
                lines.append(f"  > {ex['gen_snippet'][:400]}")
            lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written → {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from transformers import AutoTokenizer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load fixed eval IDs and rows
    ids_data  = json.load(open(IDS_PATH))
    eval_ids  = set(ids_data["ids"])
    print(f"Loaded {len(eval_ids)} fixed eval IDs from {IDS_PATH}")

    all_val = [json.loads(l) for l in open(VAL_DATA) if l.strip()]
    rows    = [r for r in all_val if str(r.get("id", "")) in eval_ids]
    print(f"Matched {len(rows)} rows from {VAL_DATA}")

    # Load prior results
    r8  = json.load(open(V8_RESULTS))
    r81 = json.load(open(V81_RESULTS))
    r82 = json.load(open(V82_RESULTS))
    print(f"Loaded V8  results: parse={r8['overall']['boxed_pct']}%  acc={r8['overall']['accuracy_pct']}%")
    print(f"Loaded V8.1 results: parse={r81['overall']['boxed_pct']}%  acc={r81['overall']['accuracy_pct']}%")
    print(f"Loaded V8.2 results: parse={r82['overall']['boxed_pct']}%  acc={r82['overall']['accuracy_pct']}%  (max_new=500 ref)")

    # Load tokenizer
    print(f"\nLoading tokenizer from {V83_ADAPTER}...")
    tokenizer = AutoTokenizer.from_pretrained(V83_ADAPTER, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate V8.3-lite
    r83 = evaluate_v83(rows, tokenizer)

    results_path = OUTPUT_DIR / "v8_3_lite_controlled_max250_results.json"
    with open(results_path, "w") as f:
        json.dump(r83, f, indent=2)
    print(f"V8.3-lite results saved → {results_path}")

    # 4-way report
    report_path = OUTPUT_DIR / "v8_3_lite_controlled_max250_report.md"
    write_4way_report(r8, r81, r82, r83, ids_data["ids"], report_path)

    # Console summary
    sep = "=" * 72
    print(f"\n{sep}")
    print("  FINAL 4-WAY COMPARISON (same 102 rows)")
    print(sep)
    print(f"  {'Metric':<25}  {'V8':>7}  {'V8.1':>7}  {'V8.2':>7}  {'V8.3-lite':>9}  {'Δ(V8.3-V8.1)':>13}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*13}")
    for label, key in [("Parse %", "boxed_pct"), ("Accuracy %", "accuracy_pct")]:
        v8  = r8["overall"][key]
        v81 = r81["overall"][key]
        v82 = r82["overall"][key]
        v83 = r83["overall"][key]
        print(f"  {label:<25}  {v8:>6.1f}%  {v81:>6.1f}%  {v82:>6.1f}%  {v83:>8.1f}%  {v83-v81:>+12.1f}%")
    print()
    for task in sorted(r8["by_task"]):
        a8  = r8["by_task"][task]["accuracy_pct"]
        a81 = r81["by_task"].get(task, {}).get("accuracy_pct", 0)
        a82 = r82["by_task"].get(task, {}).get("accuracy_pct", 0)
        a83 = r83["by_task"].get(task, {}).get("accuracy_pct", 0)
        p83 = r83["by_task"].get(task, {}).get("boxed_pct", 0)
        print(f"  {'acc_'+task:<25}  {a8:>6.1f}%  {a81:>6.1f}%  {a82:>6.1f}%  {a83:>8.1f}%  {a83-a81:>+12.1f}%  [parse={p83:.1f}%]")
    print()
    print(f"  Repair hits — V8:{len(r8['repair_phrase_hits'])}  V8.1:{len(r81['repair_phrase_hits'])}  "
          f"V8.2:{len(r82['repair_phrase_hits'])}  V8.3-lite:{len(r83['repair_phrase_hits'])}")
    print(sep)
    print(f"\nOutputs (max_new={MAX_NEW}):")
    print(f"  {results_path}")
    print(f"  {report_path}")
    print("\nCOMPLETED")


if __name__ == "__main__":
    main()
