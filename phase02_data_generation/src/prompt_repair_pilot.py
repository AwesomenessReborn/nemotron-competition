#!/usr/bin/env python3
"""
Prompt-repair pilot — 30 hard rows, 3 prompt variants.

Tests three post-hoc rationale prompt variants on rows that failed in the
100-row Fireworks pilot (bit_manipulation, cipher_text, unit_conversion,
symbol_transform). Goal: find a variant that achieves >=95% parse AND
>=95% answer-copy rate before scaling to full generation.

Variants:
  A  strict-copy JSON  — explicit CORRECT_ANSWER header, JSON format, no placeholders
  B  answer-first JSON — answer key first, brief rationale
  C  copy-only JSON    — no reasoning, just {"answer":"..."}, max_tokens=128

Usage (from project root):
    python phase02_data_generation/src/prompt_repair_pilot.py --confirm-paid-api

Hard stop: $0.10 total across all variants (30 rows × 3 variants = 90 calls max).
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL       = "https://api.fireworks.ai/inference/v1"
MODEL_ID       = "accounts/fireworks/models/deepseek-v4-flash"
PILOT_JSONL    = Path("phase02_data_generation/data/v8/fireworks_pilot_100_outputs.jsonl")
POOL_CSV       = Path("phase02_data_generation/data/v8/rejected_v8_pool.csv")
OUTPUT_DIR     = Path("phase02_data_generation/data/v8")
SUBSET_CSV     = OUTPUT_DIR / "repair_pilot_30.csv"
OUTPUT_REPORT  = OUTPUT_DIR / "prompt_repair_pilot_report.json"
HARD_TYPES     = {"bit_manipulation", "cipher_text", "unit_conversion", "symbol_transform"}
TARGET_N       = 30
WORKERS        = 2
TEMPERATURE    = 0.0
COST_HARD_STOP = 0.10           # total across all variants
DEFAULT_COST_IN  = 0.22 / 1_000_000
DEFAULT_COST_OUT = 0.88 / 1_000_000

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANT_A_SYSTEM = """You will receive a problem and its CORRECT_ANSWER.
Your ONLY job is to output a JSON object — nothing else.

Rules (violating any fails the task):
1. Do NOT re-solve, re-derive, re-verify, or re-compute anything.
2. "answer" MUST be copied character-for-character from CORRECT_ANSWER.
3. "reasoning" MUST be ≤60 words. State the rule/method, confirm the answer fits. No math.

Output exactly this JSON (no markdown fences, no extra text):
{"reasoning": "The rule [state it briefly]. Applying it to the input yields the given answer.", "answer": "CORRECT_ANSWER_HERE"}

Replace CORRECT_ANSWER_HERE with the CORRECT_ANSWER value. Replace the reasoning template with your ≤60-word explanation.
"""

VARIANT_B_SYSTEM = """You will receive a problem and its CORRECT_ANSWER.
Your ONLY job is to output a JSON object — nothing else.

Rules:
1. Output keys in this exact order: answer, reasoning.
2. "answer" is copied first — it must be an exact copy of CORRECT_ANSWER.
3. "reasoning" is ≤40 words. One sentence max.
4. No calculations, no re-derivation, no step-by-step analysis.

Output exactly:
{"answer": "CORRECT_ANSWER_COPIED_HERE", "reasoning": "one-sentence explanation ≤40 words"}
"""

VARIANT_C_SYSTEM = """Output a single JSON object with one key: "answer".
Copy the CORRECT_ANSWER value exactly into "answer".
No other text. No markdown. No explanation.

Example output: {"answer": "the exact answer here"}
"""

VARIANTS = {
    "A": {
        "label":        "strict_copy_json",
        "system":       VARIANT_A_SYSTEM,
        "max_tokens":   512,
        "answer_first": False,
    },
    "B": {
        "label":        "answer_first_json",
        "system":       VARIANT_B_SYSTEM,
        "max_tokens":   512,
        "answer_first": True,
    },
    "C": {
        "label":        "copy_only_json",
        "system":       VARIANT_C_SYSTEM,
        "max_tokens":   128,
        "answer_first": True,
    },
}

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def user_content_A(row):
    return (
        f"CORRECT_ANSWER: {row['answer']}\n\n"
        f"Problem:\n{row['prompt']}\n\n"
        f"Write a brief rationale (≤60 words) and copy CORRECT_ANSWER exactly. "
        f"Do NOT solve, verify, derive, or recompute."
    )

def user_content_B(row):
    return (
        f"CORRECT_ANSWER: {row['answer']}\n\n"
        f"Problem:\n{row['prompt']}\n\n"
        f"Output JSON with keys answer then reasoning. "
        f"Copy CORRECT_ANSWER into answer first."
    )

def user_content_C(row):
    return (
        f"CORRECT_ANSWER: {row['answer']}\n\n"
        f"Problem:\n{row['prompt']}\n\n"
        f"Output: {{\"answer\":\"{row['answer']}\"}}\n"
        f"(Copy the answer above exactly.)"
    )

CONTENT_FN = {"A": user_content_A, "B": user_content_B, "C": user_content_C}

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_json_response(text, variant):
    """Parse a JSON response, returning (reasoning, answer, parse_ok)."""
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        obj = json.loads(text)
        if variant in ("A",):
            reasoning = str(obj.get("reasoning", "")).strip()
            answer    = str(obj.get("answer", "")).strip()
        else:  # B and C
            answer    = str(obj.get("answer", "")).strip()
            reasoning = str(obj.get("reasoning", "")).strip()
        if answer:
            return reasoning, answer, True
    except json.JSONDecodeError:
        pass
    # Fallback: try to extract "answer" with regex
    am = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if am:
        return "", am.group(1), True
    return text, "PARSE_ERROR", False


def answer_copy_ok(pred, gold, task_type):
    pred, gold = str(pred).strip(), str(gold).strip()
    if task_type == "roman":
        return pred.upper() == gold.upper()
    if task_type in ("gravity", "unit_conversion"):
        try:
            p, g = float(pred), float(gold)
            return abs(p - g) / max(abs(g), 1) < 0.01
        except ValueError:
            return pred == gold
    return pred == gold

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_row(row, client, variant_key):
    cfg       = VARIANTS[variant_key]
    content_fn = CONTENT_FN[variant_key]
    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL_ID,
        max_tokens=cfg["max_tokens"],
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": cfg["system"]},
            {"role": "user",   "content": content_fn(row)},
        ],
    )
    elapsed    = round(time.time() - t0, 2)
    raw        = response.choices[0].message.content
    tokens_in  = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    reasoning, answer, parse_ok = parse_json_response(raw, variant_key)
    copy_ok = answer_copy_ok(answer, str(row["answer"]), row["task_type"])
    return {
        "id":           str(row["id"]),
        "task_type":    row["task_type"],
        "variant":      variant_key,
        "gold_answer":  str(row["answer"]),
        "raw_response": raw,
        "reasoning":    reasoning,
        "answer":       answer,
        "parse_ok":     parse_ok,
        "copy_ok":      copy_ok,
        "tokens_in":    tokens_in,
        "tokens_out":   tokens_out,
        "elapsed":      elapsed,
    }

# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------

def run_variant(rows, client, variant_key, cost_in, cost_out, shared_cost, cost_lock):
    label   = VARIANTS[variant_key]["label"]
    results = []
    errors  = 0

    def process(row):
        nonlocal errors
        for attempt in range(3):
            try:
                r = call_row(row, client, variant_key)
                with cost_lock:
                    shared_cost[0] += r["tokens_in"] * cost_in + r["tokens_out"] * cost_out
                    current = shared_cost[0]
                if current > COST_HARD_STOP:
                    print(f"  HARD STOP: total cost ${current:.4f} exceeds ${COST_HARD_STOP}.",
                          flush=True)
                    return None
                return r
            except Exception as e:
                err = str(e)
                wait = 30 if any(k in err.lower() for k in ("429", "rate")) else 5
                print(f"  [{row['id']}] attempt {attempt+1} error: {e}", flush=True)
                time.sleep(wait)
        errors += 1
        return {
            "id": str(row["id"]), "task_type": row["task_type"], "variant": variant_key,
            "gold_answer": str(row["answer"]), "raw_response": "",
            "reasoning": "GENERATION_FAILED", "answer": "ERROR",
            "parse_ok": False, "copy_ok": False,
            "tokens_in": 0, "tokens_out": 0, "elapsed": -1,
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process, row): row for _, row in rows.iterrows()}
        for f in as_completed(futures):
            r = f.result()
            if r is None:
                break
            results.append(r)
            n_done = len(results)
            if n_done % 10 == 0 or n_done == len(rows):
                cp = sum(r["copy_ok"] for r in results)
                pp = sum(r["parse_ok"] for r in results)
                print(f"  [{variant_key}] {n_done}/{len(rows)} done  "
                      f"parse={pp}/{n_done}  copy={cp}/{n_done}", flush=True)
    elapsed = time.time() - t0

    # Aggregate
    n       = len(results)
    parse_n = sum(r["parse_ok"] for r in results)
    copy_n  = sum(r["copy_ok"]  for r in results)
    t_in    = sum(r["tokens_in"]  for r in results)
    t_out   = sum(r["tokens_out"] for r in results)
    cost    = t_in * cost_in + t_out * cost_out

    by_task = defaultdict(lambda: {"done":0,"parse":0,"copy":0,"t_out":0})
    for r in results:
        tt = r["task_type"]
        by_task[tt]["done"] += 1
        by_task[tt]["t_out"] += r["tokens_out"]
        if r["parse_ok"]: by_task[tt]["parse"] += 1
        if r["copy_ok"]:  by_task[tt]["copy"]  += 1

    per_task = {}
    for tt, v in sorted(by_task.items()):
        d = v["done"]
        per_task[tt] = {
            "done": d,
            "parse_ok": v["parse"],
            "parse_rate": round(v["parse"]/d, 4) if d else 0,
            "copy_ok": v["copy"],
            "copy_rate": round(v["copy"]/d, 4) if d else 0,
            "avg_tokens_out": round(v["t_out"]/d, 1) if d else 0,
        }

    failures = [r for r in results if not r["parse_ok"] or not r["copy_ok"]]
    fail_examples = []
    for r in failures[:5]:
        fail_examples.append({
            "id":           r["id"],
            "task_type":    r["task_type"],
            "parse_ok":     r["parse_ok"],
            "copy_ok":      r["copy_ok"],
            "gold_answer":  r["gold_answer"],
            "model_answer": r["answer"][:80],
            "tokens_out":   r["tokens_out"],
            "raw_snippet":  r["raw_response"][:200],
        })

    return {
        "variant":         variant_key,
        "label":           label,
        "rows":            n,
        "errors":          errors,
        "parse_ok":        parse_n,
        "parse_rate":      round(parse_n/n, 4) if n else 0,
        "copy_ok":         copy_n,
        "copy_rate":       round(copy_n/n, 4) if n else 0,
        "avg_tokens_in":   round(t_in/n, 1) if n else 0,
        "avg_tokens_out":  round(t_out/n, 1) if n else 0,
        "cost_usd":        round(cost, 6),
        "elapsed_seconds": round(elapsed, 2),
        "per_task":        per_task,
        "fail_examples":   fail_examples,
        "meets_gate":      (parse_n/n >= 0.95 and copy_n/n >= 0.95) if n else False,
    }

# ---------------------------------------------------------------------------
# Subset builder
# ---------------------------------------------------------------------------

def build_subset():
    """All pilot failures from hard types + top-up to 30 from successes."""
    if SUBSET_CSV.exists():
        print(f"  Reusing existing subset: {SUBSET_CSV}")
        return pd.read_csv(SUBSET_CSV)

    # Load pilot outputs
    pilot_rows = []
    with open(PILOT_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                pilot_rows.append(json.loads(line))

    fail_ids = {r["id"] for r in pilot_rows
                if r["task_type"] in HARD_TYPES
                and (not r["parse_success"] or not r["answer_copy_ok"])}
    ok_ids   = {r["id"] for r in pilot_rows
                if r["task_type"] in HARD_TYPES
                and r["parse_success"] and r["answer_copy_ok"]}

    pool = pd.read_csv(POOL_CSV)
    pool["id"] = pool["id"].astype(str)
    pool = pool[pool["task_type"].isin(HARD_TYPES)]

    failures = pool[pool["id"].isin(fail_ids)]
    successes = pool[pool["id"].isin(ok_ids)]

    n_fail    = len(failures)
    n_topup   = max(0, TARGET_N - n_fail)
    topup     = successes.sample(n=min(n_topup, len(successes)), random_state=42)
    subset    = pd.concat([failures, topup], ignore_index=True)

    print(f"  Subset: {n_fail} failures + {len(topup)} success top-up = {len(subset)} rows")
    subset.to_csv(SUBSET_CSV, index=False)
    return subset

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prompt-repair pilot: 3 variants × 30 hard rows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--confirm-paid-api", action="store_true",
                        help="Required — acknowledges real API calls.")
    parser.add_argument("--variants", default="A,B,C",
                        help="Comma-separated variant keys to run (default: A,B,C)")
    parser.add_argument("--cost-per-m-in",  type=float, default=0.22)
    parser.add_argument("--cost-per-m-out", type=float, default=0.88)
    args = parser.parse_args()

    cost_in  = args.cost_per_m_in  / 1_000_000
    cost_out = args.cost_per_m_out / 1_000_000
    variants_to_run = [v.strip().upper() for v in args.variants.split(",")]

    print("\n" + "=" * 65)
    print("Prompt-repair pilot — 3 variants × ~30 hard rows")
    print("=" * 65)
    print(f"  Model    : {MODEL_ID}")
    print(f"  Workers  : {WORKERS}  temperature: {TEMPERATURE}")
    print(f"  Variants : {variants_to_run}")
    print(f"  Cost rates: ${args.cost_per_m_in:.2f}/M in  "
          f"${args.cost_per_m_out:.2f}/M out")
    print(f"  Hard stop: ${COST_HARD_STOP} total across all variants")
    print(f"  Cost note: verify at https://fireworks.ai/pricing")

    if not args.confirm_paid_api:
        print("\n  Add --confirm-paid-api to run. Exiting.\n")
        sys.exit(0)

    load_dotenv()
    key = os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        raise SystemExit("\nERROR: FIREWORKS_API_KEY not set.\n")
    masked = key[:6] + "*" * max(4, len(key) - 10) + key[-4:]
    print(f"\n  API key  : present ({masked})")

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=BASE_URL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subset = build_subset()
    print(f"  Subset   : {len(subset)} rows  ({', '.join(sorted(subset['task_type'].unique()))})")

    shared_cost = [0.0]
    cost_lock   = threading.Lock()

    all_variant_results = {}
    for vkey in variants_to_run:
        if vkey not in VARIANTS:
            print(f"  Unknown variant {vkey!r}, skipping.")
            continue
        with cost_lock:
            if shared_cost[0] > COST_HARD_STOP:
                print(f"  Hard stop hit — skipping variant {vkey}.")
                break
        print(f"\n  --- Variant {vkey}: {VARIANTS[vkey]['label']} ---")
        print(f"  max_tokens={VARIANTS[vkey]['max_tokens']}")
        result = run_variant(subset, client, vkey, cost_in, cost_out,
                             shared_cost, cost_lock)
        all_variant_results[vkey] = result
        print(f"  Variant {vkey} done: "
              f"parse={result['parse_rate']:.1%} "
              f"copy={result['copy_rate']:.1%} "
              f"avg_out={result['avg_tokens_out']:.0f}tok "
              f"cost=${result['cost_usd']:.4f}")

    # Determine recommendation
    passing = [k for k, v in all_variant_results.items() if v["meets_gate"]]
    # Prefer lowest avg_tokens_out among passing
    if passing:
        best = min(passing, key=lambda k: all_variant_results[k]["avg_tokens_out"])
        recommendation = (
            f"Variant {best} ({VARIANTS[best]['label']}) passes the gate "
            f"(parse≥95% copy≥95%). Recommend using this prompt for full generation."
        )
    else:
        best_copy = max(all_variant_results, key=lambda k: all_variant_results[k]["copy_rate"])
        recommendation = (
            f"No variant passes the gate. Best copy rate: Variant {best_copy} "
            f"({all_variant_results[best_copy]['copy_rate']:.1%}). "
            f"Do not proceed to full generation — prompt needs further work."
        )

    report = {
        "mode":                 "prompt_repair_pilot",
        "timestamp_utc":        datetime.now(timezone.utc).isoformat(),
        "model":                MODEL_ID,
        "subset_rows":          len(subset),
        "total_cost_usd":       round(shared_cost[0], 6),
        "hard_stop_limit_usd":  COST_HARD_STOP,
        "cost_rates":           {"in_per_m": args.cost_per_m_in, "out_per_m": args.cost_per_m_out},
        "variants":             all_variant_results,
        "recommendation":       recommendation,
        "gate":                 {"parse_min": 0.95, "copy_min": 0.95},
    }

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    # Print comparison table
    print(f"\n{'='*65}")
    print("REPAIR PILOT SUMMARY")
    print(f"{'='*65}")
    print(f"{'Variant':<8} {'label':<22} {'parse':<8} {'copy':<8} "
          f"{'avg_out':<10} {'cost':<8} {'gate?'}")
    print("-" * 65)
    for vkey, v in all_variant_results.items():
        gate = "PASS" if v["meets_gate"] else "fail"
        print(f"  {vkey:<6} {v['label']:<22} "
              f"{v['parse_rate']:<8.1%} {v['copy_rate']:<8.1%} "
              f"{v['avg_tokens_out']:<10.0f} ${v['cost_usd']:<7.4f} {gate}")
    print()
    print(f"Total cost: ${shared_cost[0]:.4f}")
    print()
    print(f"Recommendation: {recommendation}")
    print()

    print("Per-task breakdown:")
    for vkey, v in all_variant_results.items():
        print(f"  Variant {vkey}:")
        for tt, pt in v["per_task"].items():
            print(f"    {tt:<22} parse={pt['parse_ok']}/{pt['done']} "
                  f"({pt['parse_rate']:.0%})  "
                  f"copy={pt['copy_ok']}/{pt['done']} "
                  f"({pt['copy_rate']:.0%})  "
                  f"avg_out={pt['avg_tokens_out']:.0f}tok")

    if any(v["fail_examples"] for v in all_variant_results.values()):
        print("\nFail examples (first 2 per variant):")
        for vkey, v in all_variant_results.items():
            if v["fail_examples"]:
                print(f"  Variant {vkey}:")
                for ex in v["fail_examples"][:2]:
                    print(f"    id={ex['id']} task={ex['task_type']} "
                          f"gold={ex['gold_answer']!r} pred={ex['model_answer']!r}")
                    print(f"    raw: {ex['raw_snippet'][:120]!r}")

    print(f"\nOutputs:")
    print(f"  {OUTPUT_REPORT}")
    print(f"  {SUBSET_CSV}")
    print(f"\nDo NOT run full generation until a variant passes the gate.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
