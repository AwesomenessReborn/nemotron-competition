import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

INPUT_PATH = "shared/data/raw/train_with_task_type.csv"
OUTPUT_PATH = "phase02_data_generation/data/train_reasoning_v7_haiku.jsonl"
MODEL_ID = "claude-haiku-4-5"
MAX_WORKERS = 15
PILOT_N = 100
PILOT_GATE = 0.95  # both parse and match must be >= this to continue

# claude-haiku-4-5 pricing (per token)
COST_PER_INPUT_TOKEN = 0.80 / 1_000_000
COST_PER_OUTPUT_TOKEN = 4.00 / 1_000_000

TASK_SHORT = {
    "bit_manipulation": "bit",
    "cipher_text": "cipher",
    "gravity": "grav",
    "roman": "roman",
    "symbol_transform": "sym",
    "unit_conversion": "unit",
}

SYSTEM_PROMPT = """You are given a problem and its correct answer.
Write a concise explanation of WHY the answer is correct.

You MUST respond in this exact format:

REASONING: <concise explanation, 2-5 sentences max>
ANSWER: <copy the given answer exactly>

Rules:
- Keep reasoning SHORT and direct — explain the pattern/rule used, not every step
- Do not re-derive or verify the answer — just explain it
- Copy the answer field exactly as given, no changes
"""


def parse_response(text):
    reasoning_match = re.search(r"REASONING:\s*(.*?)(?=ANSWER:|$)", text, re.DOTALL)
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text)
    if reasoning_match and answer_match:
        return reasoning_match.group(1).strip(), answer_match.group(1).strip(), True
    return text.strip(), "PARSE_ERROR", False


def answer_correct(pred, gold, task_type):
    pred = str(pred).strip()
    gold = str(gold).strip()
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


def process_row(row, client, write_lock, counters, counter_lock):
    for attempt in range(3):
        try:
            t0 = time.time()
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content(row)}],
            )
            raw = response.content[0].text
            gen_time = round(time.time() - t0, 2)
            reasoning, answer, parse_ok = parse_response(raw)
            correct = answer_correct(answer, str(row["answer"]), row["task_type"])

            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens

            result = {
                "id": row["id"],
                "prompt": row["prompt"],
                "reasoning": reasoning,
                "answer": answer,
                "task_type": row["task_type"],
                "gold_answer": str(row["answer"]),
                "model": MODEL_ID,
                "gen_time": gen_time,
                "parse_success": parse_ok,
                "answer_correct": correct,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }

            with write_lock:
                with open(OUTPUT_PATH, "a") as f:
                    f.write(json.dumps(result) + "\n")

            with counter_lock:
                counters["done"] += 1
                counters["tokens_in"] += tokens_in
                counters["tokens_out"] += tokens_out
                tt = row["task_type"]
                counters["by_task"][tt]["done"] += 1
                if parse_ok:
                    counters["by_task"][tt]["parse_ok"] += 1
                if correct:
                    counters["by_task"][tt]["correct"] += 1
                counters["gen_times"].append(gen_time)

                done = counters["done"]
                total = counters["total"]

                if done % 100 == 0:
                    recent = counters["gen_times"][-100:]
                    avg_t = sum(recent) / len(recent)
                    remaining_s = (total - done) * avg_t / MAX_WORKERS
                    task_summary = " ".join(
                        f"{TASK_SHORT.get(tt, tt)}:{int(100*v['parse_ok']/v['done'])}%"
                        for tt, v in sorted(counters["by_task"].items())
                        if v["done"] > 0
                    )
                    cost = (counters["tokens_in"] * COST_PER_INPUT_TOKEN
                            + counters["tokens_out"] * COST_PER_OUTPUT_TOKEN)
                    print(
                        f"[{done:04d}/{total}] | {task_summary} | "
                        f"avg {avg_t:.1f}s | est {remaining_s/60:.0f}min remaining | "
                        f"cost ${cost:.2f}",
                        flush=True,
                    )

            return result

        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "overloaded" in str(e).lower():
                print(f"  [{row['id']}] Rate limited, waiting 30s...", flush=True)
                time.sleep(30)
            else:
                print(f"  [{row['id']}] Error attempt {attempt + 1}: {e}", flush=True)
                time.sleep(5)

    # All retries failed
    error_result = {
        "id": row["id"],
        "prompt": row["prompt"],
        "reasoning": "GENERATION_FAILED",
        "answer": "ERROR",
        "task_type": row["task_type"],
        "gold_answer": str(row["answer"]),
        "model": MODEL_ID,
        "gen_time": -1,
        "parse_success": False,
        "answer_correct": False,
        "tokens_in": 0,
        "tokens_out": 0,
    }
    with write_lock:
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(error_result) + "\n")
    with counter_lock:
        counters["done"] += 1
        counters["errors"] += 1
    return error_result


def run_batch(rows, client, write_lock, counters, counter_lock):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, row, client, write_lock, counters, counter_lock): row["id"]
            for _, row in rows.iterrows()
        }
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  Unhandled future error: {e}", flush=True)
        return results


def check_gate(results, label):
    n = len(results)
    parse_ok = sum(r["parse_success"] for r in results)
    correct = sum(r["answer_correct"] for r in results)
    parse_rate = parse_ok / n
    match_rate = correct / n

    print(f"\n{'='*55}")
    print(f"PILOT GATE CHECK ({label})")
    print(f"  Samples:     {n}")
    print(f"  Parse rate:  {parse_ok}/{n}  {parse_rate:.1%}")
    print(f"  Match rate:  {correct}/{n}  {match_rate:.1%}")

    # Per-task breakdown
    by_task = defaultdict(lambda: [0, 0, 0])  # done, parse_ok, correct
    for r in results:
        tt = r["task_type"]
        by_task[tt][0] += 1
        if r["parse_success"]: by_task[tt][1] += 1
        if r["answer_correct"]: by_task[tt][2] += 1
    for tt, (d, p, c) in sorted(by_task.items()):
        print(f"  {tt:<20} parse={p}/{d} match={c}/{d}")

    if parse_rate < PILOT_GATE or match_rate < PILOT_GATE:
        print(f"\nGATE FAILED — threshold is {PILOT_GATE:.0%}. Stopping.")
        failures = [r for r in results if not r["parse_success"] or not r["answer_correct"]]
        print(f"\nFirst 3 failures:")
        for r in failures[:3]:
            print(f"  id={r['id']} task={r['task_type']} parse={r['parse_success']} "
                  f"gold={r['gold_answer']!r} pred={r['answer']!r}")
        return False

    print(f"GATE PASSED — continuing full run.")
    print(f"{'='*55}\n")
    return True


def main():
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not set in environment / .env")

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    df = pd.read_csv(INPUT_PATH)
    all_ids = set(df["id"].astype(str))

    # Resume: load already-completed IDs
    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        completed_ids.add(str(json.loads(line)["id"]))
                    except (json.JSONDecodeError, KeyError):
                        pass
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} already done, skipping.")

    to_process = df[~df["id"].astype(str).isin(completed_ids)].reset_index(drop=True)
    total_remaining = len(to_process)
    already_done = len(completed_ids)

    print(f"Model:   {MODEL_ID}")
    print(f"Input:   {INPUT_PATH}  ({len(df)} total rows)")
    print(f"Output:  {OUTPUT_PATH}")
    print(f"To run:  {total_remaining} rows  (skipping {already_done} already done)\n")

    write_lock = threading.Lock()
    counter_lock = threading.Lock()
    task_types = df["task_type"].dropna().unique().tolist()

    # Shared counter (total includes already-done for display)
    counters = {
        "done": 0,
        "errors": 0,
        "total": total_remaining,
        "tokens_in": 0,
        "tokens_out": 0,
        "gen_times": [],
        "by_task": {tt: {"done": 0, "parse_ok": 0, "correct": 0} for tt in task_types},
    }

    t_start = time.time()

    # --- Pilot run ---
    skip_pilot = already_done >= PILOT_N
    if skip_pilot:
        print(f"Skipping pilot — {already_done} rows already completed.\n")
        pilot_results = None
    else:
        pilot_rows = to_process.head(PILOT_N)
        print(f"Running pilot: {len(pilot_rows)} rows @ {MAX_WORKERS} workers...\n")
        pilot_results = run_batch(pilot_rows, client, write_lock, counters, counter_lock)

        if not check_gate(pilot_results, f"first {len(pilot_rows)} rows"):
            sys.exit(1)

    # --- Full run (remaining rows after pilot) ---
    remainder = to_process.iloc[PILOT_N:] if not skip_pilot else to_process
    if len(remainder) == 0:
        print("All rows already processed.")
    else:
        print(f"Running remaining {len(remainder)} rows @ {MAX_WORKERS} workers...\n")
        run_batch(remainder, client, write_lock, counters, counter_lock)

    # --- Final summary ---
    elapsed = time.time() - t_start
    total_done = counters["done"]
    total_cost = (counters["tokens_in"] * COST_PER_INPUT_TOKEN
                  + counters["tokens_out"] * COST_PER_OUTPUT_TOKEN)

    print(f"\n{'='*55}")
    print(f"COMPLETE")
    print(f"  Rows processed this run: {total_done}")
    print(f"  Errors:                  {counters['errors']}")
    print(f"  Time:                    {elapsed/60:.1f} min")
    print(f"  Tokens in/out:           {counters['tokens_in']:,} / {counters['tokens_out']:,}")
    print(f"  Estimated cost:          ${total_cost:.4f}")
    print(f"\nParse / match per task type:")
    for tt, v in sorted(counters["by_task"].items()):
        if v["done"] > 0:
            print(f"  {tt:<22} parse={v['parse_ok']}/{v['done']}  "
                  f"match={v['correct']}/{v['done']}")


if __name__ == "__main__":
    main()
