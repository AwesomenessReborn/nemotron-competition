#!/usr/bin/env python3
"""
Fireworks provider smoke test.

Stages (run from project root):
  1. Verify FIREWORKS_API_KEY is present — never print the key
  2. Dry-run on row 0 — print sanitized payload, no API call
  3. Real call on row 0 — requires --confirm-paid-api
     If row 0 fails quality gate, write result and stop.
  4. If row 0 passes, run rows 1-4 sequentially (5 rows total)
  5. Write outputs to phase02_data_generation/data/v8/
  6. Print final report — STOP (do not run 100-row pilot)

Usage:
    # Env check + dry-run only (no API call):
    python phase02_data_generation/src/smoke_test_fireworks.py

    # Full smoke test (real API calls, will incur cost):
    python phase02_data_generation/src/smoke_test_fireworks.py --confirm-paid-api
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4-flash"
INPUT_PATH = "shared/data/raw/train_with_task_type.csv"
OUTPUT_DIR = Path("phase02_data_generation/data/v8")
OUTPUT_JSONL = OUTPUT_DIR / "fireworks_smoke_outputs.jsonl"
OUTPUT_REPORT = OUTPUT_DIR / "fireworks_smoke_report.json"

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

# ---------------------------------------------------------------------------
# Shared helpers (kept in sync with generate_llm.py)
# ---------------------------------------------------------------------------

def parse_response(text):
    reasoning_match = re.search(r"REASONING:\s*(.*?)(?=ANSWER:|$)", text, re.DOTALL)
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text)
    if reasoning_match and answer_match:
        return reasoning_match.group(1).strip(), answer_match.group(1).strip(), True
    return text.strip(), "PARSE_ERROR", False


def answer_correct(pred, gold, task_type):
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
# Stage 1: env check
# ---------------------------------------------------------------------------

def stage_check_env():
    load_dotenv()
    key = os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        raise SystemExit(
            "\nERROR: FIREWORKS_API_KEY is not set.\n"
            "  Add it to .env or: export FIREWORKS_API_KEY='your-key'\n"
        )
    visible_tail = key[-4:] if len(key) >= 10 else "****"
    masked = key[:6] + "*" * max(4, len(key) - 10) + visible_tail
    print(f"  FIREWORKS_API_KEY  present   length={len(key)}   {masked}")
    return key

# ---------------------------------------------------------------------------
# Stage 2: dry-run
# ---------------------------------------------------------------------------

def stage_dry_run(row, model_id):
    payload = {
        "model": model_id,
        "max_tokens": 2048,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_content(row)},
        ],
    }
    print(f"  Endpoint : POST {BASE_URL}/chat/completions")
    print(f"  Headers  : Authorization: Bearer [REDACTED]")
    print(f"             Content-Type: application/json")
    print(f"  Row      : id={row['id']}   task={row['task_type']}")
    print(f"  Payload  :")
    print(json.dumps(payload, indent=4))

# ---------------------------------------------------------------------------
# API call helper
# ---------------------------------------------------------------------------

def call_one_row(row, client, model_id):
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content(row)},
        ],
    )
    elapsed = round(time.time() - t0, 2)
    raw = response.choices[0].message.content
    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    reasoning, answer, parse_ok = parse_response(raw)
    correct = answer_correct(answer, str(row["answer"]), row["task_type"])
    return {
        "id": str(row["id"]),
        "task_type": row["task_type"],
        "prompt": row["prompt"],
        "gold_answer": str(row["answer"]),
        "model": model_id,
        "raw_response": raw,
        "reasoning": reasoning,
        "answer": answer,
        "parse_success": parse_ok,
        "answer_correct": correct,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "gen_time": elapsed,
    }


def print_row_result(result, label):
    reasoning_preview = result["reasoning"][:200].replace("\n", " ")
    if len(result["reasoning"]) > 200:
        reasoning_preview += "..."
    print(f"  [{label}]")
    print(f"    parse_ok       : {result['parse_success']}")
    print(f"    exact_match    : {result['answer_correct']}")
    print(f"    input_tokens   : {result['tokens_in']}")
    print(f"    output_tokens  : {result['tokens_out']}")
    print(f"    estimated_cost : N/A — check https://fireworks.ai/pricing")
    print(f"    gen_time       : {result['gen_time']}s")
    print(f"    gold_answer    : {result['gold_answer']!r}")
    print(f"    model_answer   : {result['answer']!r}")
    print(f"    reasoning      : {reasoning_preview}")

# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_jsonl(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w") as f:
        for r in results:
            row_data = {k: v for k, v in r.items() if k != "raw_response"}
            f.write(json.dumps(row_data) + "\n")


def build_report(results, model_id, elapsed_total):
    n = len(results)
    parse_ok = sum(r["parse_success"] for r in results)
    correct = sum(r.get("answer_correct", False) for r in results)
    tokens_in = sum(r["tokens_in"] for r in results)
    tokens_out = sum(r["tokens_out"] for r in results)

    by_task = {}
    for r in results:
        tt = r["task_type"]
        if tt not in by_task:
            by_task[tt] = {"done": 0, "parse_ok": 0, "correct": 0}
        by_task[tt]["done"] += 1
        if r["parse_success"]:
            by_task[tt]["parse_ok"] += 1
        if r.get("answer_correct", False):
            by_task[tt]["correct"] += 1

    gate_passed = (parse_ok / n >= 0.95 and correct / n >= 0.95) if n else False

    return {
        "provider": "fireworks",
        "model": model_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rows_total": n,
        "parse_ok": parse_ok,
        "parse_rate": round(parse_ok / n, 4) if n else 0,
        "exact_match": correct,
        "match_rate": round(correct / n, 4) if n else 0,
        "tokens_in_total": tokens_in,
        "tokens_out_total": tokens_out,
        "elapsed_seconds": round(elapsed_total, 2),
        "cost_note": "Pricing not hardcoded for Fireworks — check https://fireworks.ai/pricing",
        "pilot_gate_threshold": 0.95,
        "pilot_gate_passed": gate_passed,
        "per_task": by_task,
        "per_row": [
            {
                "row_index": i,
                "id": r["id"],
                "task_type": r["task_type"],
                "parse_ok": r["parse_success"],
                "exact_match": r.get("answer_correct", False),
                "input_tokens": r["tokens_in"],
                "output_tokens": r["tokens_out"],
                "gen_time": r["gen_time"],
                "gold_answer": r["gold_answer"],
                "model_answer": r["answer"],
                "raw_response": r.get("raw_response", ""),
                "reasoning": r["reasoning"],
            }
            for i, r in enumerate(results)
        ],
    }


def write_report(report):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fireworks smoke test — env check, dry-run, 1-row call, 5-row pilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--confirm-paid-api", action="store_true",
        help="Required to make real API calls. Acknowledges cost will be incurred.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model ID (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    t_start = time.time()

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1 — Environment check")
    print("=" * 60)
    api_key = stage_check_env()

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2 — Dry-run payload (1 row, no API call)")
    print("=" * 60)
    df = pd.read_csv(INPUT_PATH)
    print(f"  Dataset : {len(df)} rows  ({INPUT_PATH})\n")
    stage_dry_run(df.iloc[0], args.model)

    if not args.confirm_paid_api:
        print("\n" + "=" * 60)
        print("Stages 3-6 skipped — add --confirm-paid-api to run real API calls.")
        print("=" * 60 + "\n")
        return

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 3 — Real API call (row 0, workers=1)")
    print("=" * 60)
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    try:
        test_result = call_one_row(df.iloc[0], client, args.model)
    except Exception as e:
        print(f"\n  ERROR: API call failed — {e}")
        print("  Aborting. Check your API key and endpoint.")
        sys.exit(1)

    print_row_result(test_result, "row 0 / validation")

    if not (test_result["parse_success"] and test_result["answer_correct"]):
        print(f"\n  Row 0 did NOT pass quality gate "
              f"(parse={test_result['parse_success']}, match={test_result['answer_correct']}).")
        print("  Writing result and stopping — review before continuing.")
        write_jsonl([test_result])
        write_report(build_report([test_result], args.model, time.time() - t_start))
        print(f"  {OUTPUT_JSONL}")
        print(f"  {OUTPUT_REPORT}")
        sys.exit(1)

    print("\n  Row 0 PASSED — proceeding to 5-row mini pilot.\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print("STAGE 4 — 5-row mini pilot (rows 1-4, workers=1)")
    print("=" * 60)
    pilot_results = []
    for idx in range(1, 5):
        row = df.iloc[idx]
        print(f"\n  [row {idx}/4]  id={row['id']}  task={row['task_type']}")
        try:
            result = call_one_row(row, client, args.model)
            pilot_results.append(result)
            print(f"    parse_ok={result['parse_success']}  "
                  f"exact_match={result['answer_correct']}  "
                  f"in={result['tokens_in']} out={result['tokens_out']}  "
                  f"time={result['gen_time']}s  "
                  f"gold={result['gold_answer']!r}  pred={result['answer']!r}")
        except Exception as e:
            print(f"    ERROR: {e}")
            pilot_results.append({
                "id": str(row["id"]), "task_type": row["task_type"],
                "prompt": row["prompt"], "gold_answer": str(row["answer"]),
                "model": args.model, "raw_response": "", "reasoning": "ERROR",
                "answer": "ERROR", "parse_success": False, "answer_correct": False,
                "tokens_in": 0, "tokens_out": 0, "gen_time": -1,
            })

    # ------------------------------------------------------------------
    all_results = [test_result] + pilot_results
    elapsed = time.time() - t_start

    print("\n" + "=" * 60)
    print("STAGE 5 — Writing outputs")
    print("=" * 60)
    write_jsonl(all_results)
    report = build_report(all_results, args.model, elapsed)
    write_report(report)
    print(f"  JSONL   : {OUTPUT_JSONL}  ({len(all_results)} rows)")
    print(f"  Report  : {OUTPUT_REPORT}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 6 — Final smoke-test report")
    print("=" * 60)
    n = report["rows_total"]
    gate = "PASSED" if report["pilot_gate_passed"] else "FAILED"
    print(f"  Provider       : {report['provider']}")
    print(f"  Model          : {report['model']}")
    print(f"  Rows           : {n}")
    print(f"  Parse rate     : {report['parse_ok']}/{n}  ({report['parse_rate']:.1%})")
    print(f"  Match rate     : {report['exact_match']}/{n}  ({report['match_rate']:.1%})")
    print(f"  Tokens in      : {report['tokens_in_total']:,}")
    print(f"  Tokens out     : {report['tokens_out_total']:,}")
    print(f"  Elapsed        : {elapsed:.1f}s")
    print(f"  {report['cost_note']}")
    print(f"\n  Per task:")
    for tt, v in sorted(report["per_task"].items()):
        print(f"    {tt:<20}  parse={v['parse_ok']}/{v['done']}  match={v['correct']}/{v['done']}")
    print(f"\n  Pilot gate (>=95% parse & match): {gate}")
    print(f"\n  STOP — do not run the 100-row pilot until you approve these results.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
