# generate_full_gemini.py
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

INPUT_PATH = "shared/data/raw/train_with_task_type.csv"
OUTPUT_PATH = "phase02_data_generation/data/train_reasoning_v7_gemini.jsonl"
MODEL_ID = "gemini-2.5-flash"
MAX_WORKERS = 20

SYSTEM_PROMPT = """You are a precise reasoning assistant.
Think through the problem step by step, then give your final answer.

You MUST respond in this exact format and nothing else:

REASONING: <your step by step working here>
ANSWER: <final answer only>

Format rules:
- Roman numerals: uppercase only e.g. LXXXII
- Numbers: digits only, no units e.g. 91.84
- Cipher/symbol results: just the result e.g. 2655
- No caveats, no extra explanation outside the format
"""

# Gemini 2.5 Flash pricing (as of 2025, thinking_budget=0 = non-thinking tier)
# Input: $0.075 / 1M tokens   Output: $0.30 / 1M tokens
COST_PER_INPUT_TOKEN = 0.075 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.30 / 1_000_000

import re

def parse_response(text):
    reasoning_match = re.search(r"REASONING:\s*(.*?)(?=ANSWER:|$)", text, re.DOTALL)
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text)
    if reasoning_match and answer_match:
        return reasoning_match.group(1).strip(), answer_match.group(1).strip(), True
    return text.strip(), "PARSE_ERROR", False


def process_row(row, gemini_client, gemini_cfg, write_lock, counters, counter_lock):
    from google.api_core.exceptions import ResourceExhausted

    for attempt in range(3):
        try:
            t0 = time.time()
            response = gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=row["prompt"],
                config=gemini_cfg,
            )
            raw = response.text
            gen_time = round(time.time() - t0, 2)
            reasoning, answer, parse_ok = parse_response(raw)

            tokens_in = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            tokens_out = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

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
                counters["by_task"][row["task_type"]]["done"] += 1
                if parse_ok:
                    counters["by_task"][row["task_type"]]["parse_ok"] += 1
                counters["gen_times"].append(gen_time)

                done = counters["done"]
                total = counters["total"]

                if done % 100 == 0:
                    avg_t = sum(counters["gen_times"][-100:]) / min(100, len(counters["gen_times"]))
                    remaining = (total - done) * avg_t / MAX_WORKERS
                    task_summary = " ".join(
                        f"{tt}:{int(100*v['parse_ok']/v['done'])}%"
                        for tt, v in sorted(counters["by_task"].items())
                        if v["done"] > 0
                    )
                    print(f"[{done:04d}/{total}] | {task_summary} | avg {avg_t:.1f}s | est {remaining/60:.0f}min remaining")

                if done % 500 == 0:
                    cost = (counters["tokens_in"] * COST_PER_INPUT_TOKEN
                            + counters["tokens_out"] * COST_PER_OUTPUT_TOKEN)
                    print(f"  Cost estimate so far: ${cost:.4f} "
                          f"({counters['tokens_in']:,} in / {counters['tokens_out']:,} out tokens)")

            return result

        except ResourceExhausted:
            print(f"  [{row['id']}] Rate limited, waiting 60s...")
            time.sleep(60)
        except Exception as e:
            print(f"  [{row['id']}] Error attempt {attempt + 1}: {e}")
            time.sleep(10)

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


def main():
    load_dotenv()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("ERROR: GEMINI_API_KEY not set in environment / .env")

    from google import genai
    from google.genai import types

    g_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    gemini_cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=2048,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    df = pd.read_csv(INPUT_PATH)

    # Resume: skip already-completed IDs
    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        completed_ids.add(str(r["id"]))
                    except json.JSONDecodeError:
                        pass
        print(f"Resuming: {len(completed_ids)} already done, skipping.")

    to_process = df[~df["id"].astype(str).isin(completed_ids)]
    total = len(to_process)
    print(f"Processing {total} rows with {MODEL_ID} (max_workers={MAX_WORKERS})")
    print(f"Output: {OUTPUT_PATH}\n")

    task_types = df["task_type"].unique().tolist()
    write_lock = threading.Lock()
    counter_lock = threading.Lock()
    counters = {
        "done": 0,
        "errors": 0,
        "total": total,
        "tokens_in": 0,
        "tokens_out": 0,
        "gen_times": [],
        "by_task": {tt: {"done": 0, "parse_ok": 0} for tt in task_types},
    }

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_row, row, g_client, gemini_cfg, write_lock, counters, counter_lock
            ): row["id"]
            for _, row in to_process.iterrows()
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  Unhandled future error: {e}")

    elapsed = time.time() - t_start

    # Final summary
    print(f"\n{'='*60}")
    print(f"COMPLETE: {counters['done']} rows in {elapsed/60:.1f} min")
    print(f"Errors: {counters['errors']}")
    print(f"\nParse success by task type:")
    for tt, v in sorted(counters["by_task"].items()):
        if v["done"] > 0:
            pct = 100 * v["parse_ok"] / v["done"]
            print(f"  {tt:<20} {v['parse_ok']}/{v['done']}  {pct:.0f}%")
    total_cost = (counters["tokens_in"] * COST_PER_INPUT_TOKEN
                  + counters["tokens_out"] * COST_PER_OUTPUT_TOKEN)
    print(f"\nTokens: {counters['tokens_in']:,} in / {counters['tokens_out']:,} out")
    print(f"Estimated cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
