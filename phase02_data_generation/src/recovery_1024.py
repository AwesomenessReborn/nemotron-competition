#!/usr/bin/env python3
"""
Targeted max_tokens=1024 recovery test for gate-failed rows.

Reads failed row IDs from the existing gate JSONL, re-runs those rows
with max_tokens=1024, and prints a recovery report.

Does NOT modify the main gate JSONL.
Output: phase02_data_generation/data/v8/recovery_1024_results.jsonl

Run from project root:
  python phase02_data_generation/src/recovery_1024.py
  python phase02_data_generation/src/recovery_1024.py --workers 2
"""

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants — must match generate_llm.py exactly
# ---------------------------------------------------------------------------

MAX_TOKENS = 1024
MAX_PARSE_ATTEMPTS = 3
PROVIDER = "fireworks"
FIREWORKS_API_KEY_ENV = "FIREWORKS_API_KEY"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL_ID = "accounts/fireworks/models/deepseek-v4-flash"

INPUT_PATH = "shared/data/raw/train_with_task_type.csv"
GATE_JSONL_DEFAULT = "phase02_data_generation/data/train_reasoning_v7_fireworks.jsonl"
OUTPUT_DEFAULT = "phase02_data_generation/data/v8/recovery_1024_results.jsonl"

# Prompt variant A++ — identical to generate_llm.py SYSTEM_PROMPT
SYSTEM_PROMPT = r"""You will receive a problem and its CORRECT_ANSWER.
Your ONLY job is to output a JSON object — nothing else.

Rules (violating any fails the task):
1. Do NOT re-solve, re-derive, re-verify, or re-compute anything.
2. "answer" MUST be the CORRECT_ANSWER value copied exactly, character-for-character.
   If CORRECT_ANSWER contains a double-quote ("), escape it as \" in the JSON string.
   If CORRECT_ANSWER contains a backslash (\), escape it as \\ in the JSON string.
   Example: CORRECT_ANSWER is %">  →  "answer": "%\">"
   Example: CORRECT_ANSWER is "|%<  →  "answer": "\"|%<"
   Example: CORRECT_ANSWER is \([#  →  "answer": "\\([#"
3. "reasoning" MUST be ≤60 words. State the rule/method used; confirm the answer fits. No math.

Output exactly this JSON (no markdown fences, no extra text):
{"reasoning": "The rule [state it briefly]. Applying it to the input yields the given answer.", "answer": "CORRECT_ANSWER_HERE"}
"""

# ---------------------------------------------------------------------------
# Helpers — inline copies of generate_llm.py functions
# ---------------------------------------------------------------------------

def user_content(row):
    return f"CORRECT_ANSWER: {row['answer']}\n\n{row['prompt']}"


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


def parse_response(text, gold, task_type):
    raw = text
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    stripped = stripped.strip()

    reasoning = ""
    answer = "PARSE_ERROR"
    parsed_ok = False

    try:
        obj = json.loads(stripped)
        answer = str(obj.get("answer", "")).strip()
        reasoning = str(obj.get("reasoning", "")).strip()
        if answer:
            parsed_ok = True
    except json.JSONDecodeError:
        pass

    if not parsed_ok:
        m = re.search(r'"answer"\s*:\s*"(.*)"', stripped, re.DOTALL)
        if m:
            answer = m.group(1)
            parsed_ok = True

    matches_gold = answer_correct(answer, gold, task_type) if parsed_ok else False

    return {
        "parsed_ok": parsed_ok,
        "answer": answer,
        "answer_matches_gold": matches_gold,
        "reasoning": reasoning,
        "raw": raw,
    }

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_failed_ids(gate_jsonl):
    rows = []
    with open(gate_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("parse_success") or not r.get("answer_correct"):
                rows.append(r)
    return rows


def make_call_fn(client):
    def call(row):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content(row)},
            ],
        )
        return (
            resp.choices[0].message.content,
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            round(time.time() - t0, 2),
        )
    return call


def process_row(row, call_fn, output_path, write_lock, results_list, results_lock):
    total_tokens_in = 0
    total_tokens_out = 0
    last_result = None

    for attempt in range(MAX_PARSE_ATTEMPTS):
        if attempt > 0:
            print(
                f"  [RETRY {attempt}/{MAX_PARSE_ATTEMPTS-1}] "
                f"row_id={row['id']} task={row['task_type']}",
                flush=True,
            )
        try:
            raw, tokens_in, tokens_out, gen_time = call_fn(row)
        except Exception as e:
            err = str(e)
            wait = 60 if "quota" in err.lower() else (30 if any(k in err.lower() for k in ("429", "rate", "overloaded")) else 5)
            print(f"  [{row['id']}] API error attempt {attempt+1}: {e}", flush=True)
            time.sleep(wait)
            continue

        total_tokens_in += tokens_in
        total_tokens_out += tokens_out
        parsed = parse_response(raw, str(row["answer"]), row["task_type"])

        last_result = {
            "id":              row["id"],
            "task_type":       row["task_type"],
            "gold_answer":     str(row["answer"]),
            "parse_success":   parsed["parsed_ok"],
            "answer_correct":  parsed["answer_matches_gold"],
            "answer":          parsed["answer"],
            "reasoning":       parsed["reasoning"],
            "tokens_in":       total_tokens_in,
            "tokens_out":      total_tokens_out,
            "gen_time":        gen_time,
            "model":           MODEL_ID,
            "max_tokens_used": MAX_TOKENS,
            "attempts":        attempt + 1,
        }

        if parsed["parsed_ok"]:
            break

    if last_result is None:
        last_result = {
            "id":              row["id"],
            "task_type":       row["task_type"],
            "gold_answer":     str(row["answer"]),
            "parse_success":   False,
            "answer_correct":  False,
            "answer":          "ERROR",
            "reasoning":       "GENERATION_FAILED",
            "tokens_in":       total_tokens_in,
            "tokens_out":      total_tokens_out,
            "gen_time":        -1,
            "model":           MODEL_ID,
            "max_tokens_used": MAX_TOKENS,
            "attempts":        MAX_PARSE_ATTEMPTS,
        }

    status = "GOOD" if (last_result["parse_success"] and last_result["answer_correct"]) else "FAIL"
    print(
        f"  [{row['id']}] task={row['task_type']:<20} "
        f"parse={last_result['parse_success']} correct={last_result['answer_correct']} "
        f"tok_out={last_result['tokens_out']} attempts={last_result['attempts']} -> {status}",
        flush=True,
    )

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(last_result) + "\n")

    with results_lock:
        results_list.append(last_result)

    return last_result


def print_recovery_report(results, elapsed):
    n = len(results)
    good = [r for r in results if r["parse_success"] and r["answer_correct"]]
    parse_ok = [r for r in results if r["parse_success"]]
    avg_tok_out = sum(r["tokens_out"] for r in results) / n if n else 0
    total_tok_in = sum(r["tokens_in"] for r in results)
    total_tok_out = sum(r["tokens_out"] for r in results)
    est_cost = total_tok_in * 0.22 / 1_000_000 + total_tok_out * 0.88 / 1_000_000

    print(f"\n{'='*60}")
    print(f"RECOVERY REPORT (max_tokens={MAX_TOKENS})")
    print(f"  Rows attempted:   {n}")
    print(f"  Recovered (good): {len(good)}/{n}")
    print(f"  Parse OK:         {len(parse_ok)}/{n}")
    print(f"  Answer correct:   {len(good)}/{n}")
    print(f"  Avg tok_out:      {avg_tok_out:.0f}")
    print(f"  Total tokens:     {total_tok_in:,} in / {total_tok_out:,} out")
    print(f"  Est. cost:        ${est_cost:.4f}  (Fireworks $0.22/$0.88 per M)")
    print(f"  Elapsed:          {elapsed:.1f}s")
    print()

    by_task = defaultdict(lambda: {"n": 0, "good": 0})
    for r in results:
        tt = r["task_type"]
        by_task[tt]["n"] += 1
        if r["parse_success"] and r["answer_correct"]:
            by_task[tt]["good"] += 1
    print(f"  Per-task:")
    for tt, v in sorted(by_task.items()):
        print(f"    {tt:<22} recovered={v['good']}/{v['n']}")

    failures = [r for r in results if not r["parse_success"] or not r["answer_correct"]]
    if failures:
        print(f"\n  Still failing ({len(failures)}):")
        for r in failures:
            fail_type = "parse_fail" if not r["parse_success"] else "copy_fail"
            print(
                f"    id={r['id']} task={r['task_type']} "
                f"type={fail_type} tok_out={r['tokens_out']} "
                f"gold={r['gold_answer']!r} pred={r['answer']!r}"
            )
    else:
        print("\n  All rows recovered.")

    print("=" * 60)
    return len(good)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=f"Recovery test: re-run failed rows at max_tokens={MAX_TOKENS}"
    )
    parser.add_argument("--gate-jsonl", default=GATE_JSONL_DEFAULT,
                        help=f"Gate JSONL to read failures from (default: {GATE_JSONL_DEFAULT})")
    parser.add_argument("--output", default=OUTPUT_DEFAULT,
                        help=f"Output JSONL for recovery results (default: {OUTPUT_DEFAULT})")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent workers (default: 2)")
    args = parser.parse_args()

    api_key = os.environ.get(FIREWORKS_API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"ERROR: {FIREWORKS_API_KEY_ENV} not set — add to .env or export in shell")

    failed_rows_meta = load_failed_ids(args.gate_jsonl)
    failed_ids = {r["id"] for r in failed_rows_meta}
    print(f"Found {len(failed_ids)} failed rows in {args.gate_jsonl}")

    df = pd.read_csv(INPUT_PATH)
    subset = df[df["id"].astype(str).isin(failed_ids)].reset_index(drop=True)
    print(f"Loaded {len(subset)} matching rows from {INPUT_PATH}")

    if len(subset) != len(failed_ids):
        missing = failed_ids - set(subset["id"].astype(str))
        print(f"WARNING: {len(missing)} IDs not found in CSV: {missing}")

    if os.path.exists(args.output):
        os.remove(args.output)
        print(f"Cleared existing {args.output}")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)
    call_fn = make_call_fn(client)
    write_lock = threading.Lock()
    results_lock = threading.Lock()
    results_list = []

    print(f"\nRunning {len(subset)} rows @ {args.workers} workers, max_tokens={MAX_TOKENS}...\n")
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_row, row, call_fn, args.output, write_lock, results_list, results_lock
            ): row["id"]
            for _, row in subset.iterrows()
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  Unhandled error: {e}", flush=True)

    elapsed = time.time() - t_start
    recovered = print_recovery_report(results_list, elapsed)

    print(f"\nOutput written to: {args.output}")
    if recovered >= 12:
        print(
            f"\nRECOVERED {recovered}/13. "
            f"Confirm with user before re-running full 100-row gate at max_tokens=1024."
        )
    else:
        print(f"\nOnly {recovered}/13 recovered. Review failures before proceeding.")


if __name__ == "__main__":
    main()
