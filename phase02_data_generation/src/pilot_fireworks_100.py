#!/usr/bin/env python3
"""
Fireworks 100-row pilot — post_hoc_rationale_v8.

Mode: post_hoc_rationale
  Gold answer IS included in the prompt. The model is asked to explain
  a given correct answer, not to solve the problem independently.
  - 'answer_copy_ok' = answer-copy/format verification (not solving accuracy)
  - Do NOT interpret this metric as model intelligence on these tasks.

Hard stops:
  - Accumulated cost > $0.25 → stop and report
  - Max 100 rows (the full rejected_v8_pool)

Usage (from project root):
    python phase02_data_generation/src/pilot_fireworks_100.py --confirm-paid-api

Cost rates default to Fireworks serverless estimates. Override if you have
exact pricing:
    --cost-per-m-in 0.22 --cost-per-m-out 0.88
"""

import argparse
import json
import os
import sys
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL          = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL     = "accounts/fireworks/models/deepseek-v4-flash"
POOL_PATH         = "phase02_data_generation/data/v8/rejected_v8_pool.csv"
OUTPUT_DIR        = Path("phase02_data_generation/data/v8")
OUTPUT_JSONL      = OUTPUT_DIR / "fireworks_pilot_100_outputs.jsonl"
OUTPUT_REPORT     = OUTPUT_DIR / "fireworks_pilot_100_report.json"
MODE_LABEL        = "post_hoc_rationale_v8"
WORKERS           = 2
MAX_TOKENS        = 2048
TEMPERATURE       = 0.0
COST_HARD_STOP    = 0.25          # USD

# Default Fireworks serverless estimates for deepseek-v4-flash
# Verify at https://fireworks.ai/pricing before trusting these
DEFAULT_COST_IN   = 0.22 / 1_000_000   # $/token
DEFAULT_COST_OUT  = 0.88 / 1_000_000   # $/token

SYSTEM_PROMPT = """You are given a problem and its CORRECT answer.
Your ONLY job is to write a brief explanation of why that answer is right.

CRITICAL RULES — violating any of these will fail:
1. Do NOT re-solve the problem. Do NOT re-derive the answer. It is already given.
2. Do NOT list, analyze, or compare examples from the problem.
3. Respond in EXACTLY this two-line format and nothing else:

REASONING: <one or two sentences — state the rule/method used, confirm the answer fits>
ANSWER: <copy the given answer exactly, character for character>

For pattern/cipher/transformation tasks: state that the rule was identified from the
examples, then confirm this specific input maps to the given output under that rule.
Do not recompute. Do not verify. Maximum 40 words in REASONING.
"""

TASK_SHORT = {
    "bit_manipulation": "bit",
    "cipher_text":      "cipher",
    "gravity":          "grav",
    "roman":            "roman",
    "symbol_transform": "sym",
    "unit_conversion":  "unit",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_response(text):
    rm = re.search(r"REASONING:\s*(.*?)(?=ANSWER:|$)", text, re.DOTALL)
    am = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text)
    if rm and am:
        return rm.group(1).strip(), am.group(1).strip(), True
    return text.strip(), "PARSE_ERROR", False


def answer_copy_ok(pred, gold, task_type):
    """Checks whether the model correctly copied the given gold answer."""
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


def user_content(row):
    return (
        f"Problem:\n{row['prompt']}\n\n"
        f"Correct answer: {row['answer']}\n\n"
        f"Explain concisely why this is correct."
    )

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_row(row, client, model_id):
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_content(row)},
        ],
    )
    elapsed = round(time.time() - t0, 2)
    raw = response.choices[0].message.content
    tokens_in  = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    reasoning, answer, parse_ok = parse_response(raw)
    copy_ok = answer_copy_ok(answer, str(row["answer"]), row["task_type"])
    return {
        "id":              str(row["id"]),
        "task_type":       row["task_type"],
        "mode":            MODE_LABEL,
        "prompt":          row["prompt"],
        "gold_answer":     str(row["answer"]),
        "model":           model_id,
        "raw_response":    raw,
        "reasoning":       reasoning,
        "answer":          answer,
        "parse_success":   parse_ok,
        "answer_copy_ok":  copy_ok,       # format/copy check — NOT solving accuracy
        "v7_rejected":     bool(row.get("v7_rejected", False)),
        "tokens_in":       tokens_in,
        "tokens_out":      tokens_out,
        "gen_time":        elapsed,
    }

# ---------------------------------------------------------------------------
# Per-row processor with cost-stop awareness
# ---------------------------------------------------------------------------

def process_row(row, client, model_id, write_lock, counters, counter_lock):
    for attempt in range(3):
        try:
            result = call_row(row, client, model_id)
            break
        except Exception as e:
            err = str(e)
            if any(k in err.lower() for k in ("429", "rate", "quota", "overloaded")):
                wait = 60 if "quota" in err.lower() else 30
                print(f"  [{row['id']}] Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  [{row['id']}] Error attempt {attempt+1}: {e}", flush=True)
                time.sleep(5)
    else:
        result = {
            "id": str(row["id"]), "task_type": row["task_type"], "mode": MODE_LABEL,
            "prompt": row["prompt"], "gold_answer": str(row["answer"]),
            "model": model_id, "raw_response": "", "reasoning": "GENERATION_FAILED",
            "answer": "ERROR", "parse_success": False, "answer_copy_ok": False,
            "v7_rejected": bool(row.get("v7_rejected", False)),
            "tokens_in": 0, "tokens_out": 0, "gen_time": -1,
        }

    with write_lock:
        with open(OUTPUT_JSONL, "a") as f:
            row_data = {k: v for k, v in result.items() if k != "raw_response"}
            f.write(json.dumps(row_data) + "\n")

    with counter_lock:
        counters["done"] += 1
        counters["tokens_in"]  += result["tokens_in"]
        counters["tokens_out"] += result["tokens_out"]
        counters["cost"] += (result["tokens_in"] * counters["cost_in"]
                            + result["tokens_out"] * counters["cost_out"])
        counters["errors"] += (1 if result["reasoning"] == "GENERATION_FAILED" else 0)
        tt = result["task_type"]
        counters["by_task"][tt]["done"] += 1
        if result["parse_success"]:      counters["by_task"][tt]["parse_ok"] += 1
        if result["answer_copy_ok"]:     counters["by_task"][tt]["copy_ok"]  += 1
        counters["by_task"][tt]["tokens_in"]  += result["tokens_in"]
        counters["by_task"][tt]["tokens_out"] += result["tokens_out"]
        counters["gen_times"].append(result["gen_time"])
        _log_progress(counters)

    return result


def _log_progress(counters):
    done  = counters["done"]
    total = counters["total"]
    if done % 10 != 0:
        return
    recent  = counters["gen_times"][-20:]
    avg_t   = sum(recent) / len(recent)
    eta_s   = (total - done) * avg_t / max(WORKERS, 1)
    summary = "  ".join(
        f"{TASK_SHORT.get(tt,'?')}:{int(100*v['copy_ok']/v['done'])}%"
        for tt, v in sorted(counters["by_task"].items()) if v["done"] > 0
    )
    cost_str = f"${counters['cost']:.4f}" if counters["cost_in"] > 0 else "cost=n/a"
    print(f"  [{done:03d}/{total}] {summary} | "
          f"avg {avg_t:.1f}s | eta {eta_s/60:.0f}min | {cost_str}", flush=True)

# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(results, model_id, elapsed, cost_in_rate, cost_out_rate, pool_total):
    n          = len(results)
    parse_ok   = sum(r["parse_success"] for r in results)
    copy_ok    = sum(r["answer_copy_ok"] for r in results)
    t_in       = sum(r["tokens_in"] for r in results)
    t_out      = sum(r["tokens_out"] for r in results)
    total_cost = t_in * cost_in_rate + t_out * cost_out_rate
    errors     = sum(1 for r in results if r["reasoning"] == "GENERATION_FAILED")

    by_task = defaultdict(lambda: {
        "done": 0, "parse_ok": 0, "copy_ok": 0,
        "tokens_in": 0, "tokens_out": 0,
    })
    for r in results:
        tt = r["task_type"]
        by_task[tt]["done"]       += 1
        by_task[tt]["tokens_in"]  += r["tokens_in"]
        by_task[tt]["tokens_out"] += r["tokens_out"]
        if r["parse_success"]:    by_task[tt]["parse_ok"] += 1
        if r["answer_copy_ok"]:   by_task[tt]["copy_ok"]  += 1

    per_task_report = {}
    for tt, v in sorted(by_task.items()):
        d = v["done"]
        per_task_report[tt] = {
            "done":               d,
            "parse_ok":           v["parse_ok"],
            "parse_rate":         round(v["parse_ok"] / d, 4) if d else 0,
            "answer_copy_ok":     v["copy_ok"],
            "answer_copy_rate":   round(v["copy_ok"] / d, 4) if d else 0,
            "avg_tokens_in":      round(v["tokens_in"] / d, 1) if d else 0,
            "avg_tokens_out":     round(v["tokens_out"] / d, 1) if d else 0,
        }

    # Cost projection to full dataset (pool_total rows at current per-row cost)
    cost_per_row = total_cost / n if n > 0 else 0
    projected_total = cost_per_row * pool_total

    # Good examples: parse_ok AND answer_copy_ok
    good = [r for r in results if r["parse_success"] and r["answer_copy_ok"]]
    bad  = [r for r in results if not r["parse_success"] or not r["answer_copy_ok"]]

    def summarise_example(r):
        return {
            "id":           r["id"],
            "task_type":    r["task_type"],
            "parse_ok":     r["parse_success"],
            "copy_ok":      r["answer_copy_ok"],
            "gold_answer":  r["gold_answer"],
            "model_answer": r["answer"],
            "tokens_out":   r["tokens_out"],
            "reasoning":    r["reasoning"][:300] + ("..." if len(r["reasoning"]) > 300 else ""),
            "raw_response": r.get("raw_response", "")[:500],
        }

    cost_note = ("Rates are estimates — verify at https://fireworks.ai/pricing"
                 if cost_in_rate == DEFAULT_COST_IN else "User-supplied rates")

    return {
        "mode":                  MODE_LABEL,
        "mode_note":             ("Gold answer provided in prompt. answer_copy_ok measures "
                                  "format/copy fidelity, NOT model-solving accuracy."),
        "provider":              "fireworks",
        "model":                 model_id,
        "timestamp_utc":         datetime.now(timezone.utc).isoformat(),
        "rows_total":            n,
        "errors":                errors,
        "parse_ok":              parse_ok,
        "parse_rate":            round(parse_ok / n, 4) if n else 0,
        "answer_copy_ok":        copy_ok,
        "answer_copy_rate":      round(copy_ok / n, 4) if n else 0,
        "tokens_in_total":       t_in,
        "tokens_out_total":      t_out,
        "avg_tokens_in":         round(t_in / n, 1) if n else 0,
        "avg_tokens_out":        round(t_out / n, 1) if n else 0,
        "elapsed_seconds":       round(elapsed, 2),
        "cost_usd":              round(total_cost, 6),
        "cost_per_row_usd":      round(cost_per_row, 6),
        "cost_in_rate_per_m":    cost_in_rate * 1_000_000,
        "cost_out_rate_per_m":   cost_out_rate * 1_000_000,
        "cost_note":             cost_note,
        "projected_cost_9500_rows": round(cost_per_row * 9500, 4),
        "projected_cost_pool_total": round(projected_total, 4),
        "hard_stop_limit_usd":   COST_HARD_STOP,
        "per_task":              per_task_report,
        "good_examples":         [summarise_example(r) for r in good[:3]],
        "bad_examples":          [summarise_example(r) for r in bad[:5]],
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=f"Fireworks 100-row pilot ({MODE_LABEL})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--confirm-paid-api", action="store_true",
                        help="Required — acknowledges real API calls will incur cost.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cost-per-m-in",  type=float, default=DEFAULT_COST_IN  * 1_000_000,
                        help="Input cost per million tokens in USD (default: 0.22)")
    parser.add_argument("--cost-per-m-out", type=float, default=DEFAULT_COST_OUT * 1_000_000,
                        help="Output cost per million tokens in USD (default: 0.88)")
    args = parser.parse_args()

    cost_in  = args.cost_per_m_in  / 1_000_000
    cost_out = args.cost_per_m_out / 1_000_000

    print("\n" + "=" * 60)
    print(f"Fireworks 100-row pilot — {MODE_LABEL}")
    print("=" * 60)
    print(f"  Mode note: gold_answer IS in prompt — answer_copy_ok is format")
    print(f"             verification only, NOT model-solving accuracy.")
    print(f"  Model    : {args.model}")
    print(f"  Workers  : {WORKERS}")
    print(f"  max_tokens: {MAX_TOKENS}  temperature: {TEMPERATURE}")
    print(f"  Cost rates: ${args.cost_per_m_in:.2f}/M in  "
          f"${args.cost_per_m_out:.2f}/M out  (hard stop: ${COST_HARD_STOP})")
    print(f"  Cost note: verify rates at https://fireworks.ai/pricing")

    if not args.confirm_paid_api:
        print(f"\n  Add --confirm-paid-api to run. Exiting.\n")
        sys.exit(0)

    # Load API key
    load_dotenv()
    key = os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        raise SystemExit("\nERROR: FIREWORKS_API_KEY not set.\n")
    masked = key[:6] + "*" * max(4, len(key) - 10) + key[-4:]
    print(f"\n  API key  : present ({masked})")

    # Load pool
    pool = pd.read_csv(POOL_PATH)
    pool['id'] = pool['id'].astype(str)
    print(f"  Pool     : {len(pool)} rows from {POOL_PATH}")
    print(f"  Outputs  : {OUTPUT_JSONL}")

    # Resume: skip already-done IDs
    done_ids = set()
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(str(json.loads(line)["id"]))
                    except (json.JSONDecodeError, KeyError):
                        pass
        if done_ids:
            print(f"  Resuming : {len(done_ids)} already done, skipping.")

    to_process = pool[~pool['id'].isin(done_ids)].reset_index(drop=True)
    print(f"  To run   : {len(to_process)} rows\n")

    if len(to_process) == 0:
        print("Nothing to do — all rows already processed.")
        return

    # Setup
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=BASE_URL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task_types  = pool["task_type"].dropna().unique().tolist()
    write_lock  = threading.Lock()
    counter_lock = threading.Lock()
    counters = {
        "done":      0,
        "errors":    0,
        "total":     len(to_process),
        "tokens_in": 0, "tokens_out": 0,
        "cost":      0.0,
        "cost_in":   cost_in,
        "cost_out":  cost_out,
        "gen_times": [],
        "stop_flag": False,
        "by_task":   {tt: {"done": 0, "parse_ok": 0, "copy_ok": 0,
                            "tokens_in": 0, "tokens_out": 0}
                      for tt in task_types},
    }

    t_start  = time.time()
    results  = []
    stopped_early = False

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(
                process_row, row, client, args.model,
                write_lock, counters, counter_lock
            ): row["id"]
            for _, row in to_process.iterrows()
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"  Unhandled future error: {e}", flush=True)

            with counter_lock:
                current_cost = counters["cost"]

            if current_cost > COST_HARD_STOP:
                print(f"\n  HARD STOP: accumulated cost ${current_cost:.4f} "
                      f"exceeds limit ${COST_HARD_STOP}.", flush=True)
                stopped_early = True
                executor.shutdown(wait=False, cancel_futures=True)
                break

    elapsed = time.time() - t_start

    # Reload all results from JSONL (includes any from previous runs if resuming)
    all_results = []
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Enrich results in memory with raw_response for report examples
    raw_by_id = {r["id"]: r.get("raw_response", "") for r in results}
    for r in all_results:
        r["raw_response"] = raw_by_id.get(r["id"], "")

    # Write report
    report = build_report(all_results, args.model, elapsed, cost_in, cost_out, len(pool))
    report["stopped_early"] = stopped_early
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    n = report["rows_total"]
    print(f"\n{'='*60}")
    print(f"PILOT COMPLETE  ({MODE_LABEL})")
    print(f"{'='*60}")
    if stopped_early:
        print(f"  *** STOPPED EARLY — cost limit hit ***")
    print(f"  Mode note      : answer_copy_ok = format check, NOT solve accuracy")
    print(f"  Rows           : {n}")
    print(f"  Errors         : {report['errors']}")
    print(f"  Parse rate     : {report['parse_ok']}/{n}  ({report['parse_rate']:.1%})")
    print(f"  Answer copy    : {report['answer_copy_ok']}/{n}  ({report['answer_copy_rate']:.1%})")
    print(f"  Avg tokens in  : {report['avg_tokens_in']}")
    print(f"  Avg tokens out : {report['avg_tokens_out']}")
    print(f"  Total tokens   : {report['tokens_in_total']:,} in / "
          f"{report['tokens_out_total']:,} out")
    print(f"  Elapsed        : {elapsed:.1f}s")
    print(f"  Cost (pilot)   : ${report['cost_usd']:.4f}")
    print(f"  Cost/row       : ${report['cost_per_row_usd']:.5f}")
    print(f"  Projected 9500 rows : ${report['projected_cost_9500_rows']:.2f}")
    print(f"  {report['cost_note']}")

    print(f"\n  Per task (parse% / copy%):")
    for tt, v in sorted(report["per_task"].items()):
        print(f"    {tt:<20}  parse={v['parse_ok']}/{v['done']} ({v['parse_rate']:.0%})  "
              f"copy={v['answer_copy_ok']}/{v['done']} ({v['answer_copy_rate']:.0%})  "
              f"avg_out={v['avg_tokens_out']:.0f}tok")

    if report["bad_examples"]:
        print(f"\n  Bad examples ({len(report['bad_examples'])}):")
        for ex in report["bad_examples"][:3]:
            print(f"    id={ex['id']} task={ex['task_type']} "
                  f"parse={ex['parse_ok']} copy={ex['copy_ok']} "
                  f"gold={ex['gold_answer']!r} pred={ex['model_answer']!r}")

    print(f"\n  Outputs:")
    print(f"    {OUTPUT_JSONL}")
    print(f"    {OUTPUT_REPORT}")
    print(f"\n  STOP — do not run full generation until these results are approved.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
