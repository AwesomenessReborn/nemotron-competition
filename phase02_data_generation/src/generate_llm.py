#!/usr/bin/env python3
"""
Phase 2 LLM reasoning-trace generation — unified multi-provider script.

Providers: deepseek, gemini, openrouter, mock

Workflow:
  # 1. Inspect payloads (no API calls)
  python -m src.generate_llm --provider deepseek --dry-run

  # 2. Pilot: 100 rows, quality gate, then stop
  python -m src.generate_llm --provider deepseek --max-rows 100

  # 3. Full run (after reviewing pilot output)
  python -m src.generate_llm --provider deepseek

Keys read from .env or shell environment — never hardcoded, never committed.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

INPUT_PATH = "shared/data/raw/train_with_task_type.csv"
PILOT_N = 100
PILOT_GATE = 0.95

PROVIDERS = {
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "cost_in": 0.14 / 1_000_000,   # $/token, deepseek-chat non-reasoner
        "cost_out": 0.28 / 1_000_000,
        "default_workers": 8,
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "base_url": None,
        "default_model": "gemini-2.5-flash",
        "cost_in": 0.075 / 1_000_000,  # $/token, non-thinking tier
        "cost_out": 0.300 / 1_000_000,
        "default_workers": 20,
    },
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "deepseek/deepseek-chat",
        "cost_in": None,                # varies by routed model
        "cost_out": None,
        "default_workers": 10,
    },
    "mock": {
        "env_key": None,
        "base_url": None,
        "default_model": "mock-v1",
        "cost_in": 0.0,
        "cost_out": 0.0,
        "default_workers": 50,
    },
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

TASK_SHORT = {
    "bit_manipulation": "bit",
    "cipher_text": "cipher",
    "gravity": "grav",
    "roman": "roman",
    "symbol_transform": "sym",
    "unit_conversion": "unit",
}

# ---------------------------------------------------------------------------
# Shared helpers
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
# Provider setup
# ---------------------------------------------------------------------------

def check_api_key(provider):
    cfg = PROVIDERS[provider]
    env_key = cfg["env_key"]
    if env_key and not os.environ.get(env_key):
        raise SystemExit(
            f"\nERROR: {env_key} is not set.\n"
            f"  Add it to .env or export it in your shell.\n"
            f"  Do NOT hardcode keys in source files or commit .env to git.\n"
        )


def build_client(provider, model_id):
    """Return a client_state dict appropriate for the provider."""
    cfg = PROVIDERS[provider]

    if provider in ("deepseek", "openrouter"):
        from openai import OpenAI
        kwargs = {
            "api_key": os.environ[cfg["env_key"]],
            "base_url": cfg["base_url"],
        }
        if provider == "openrouter":
            kwargs["default_headers"] = {
                "HTTP-Referer": "https://github.com/nemotron-competition"
            }
        return {"client": OpenAI(**kwargs)}

    if provider == "gemini":
        from google import genai
        from google.genai import types
        g_client = genai.Client(api_key=os.environ[cfg["env_key"]])
        g_cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=600,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        return {"client": g_client, "config": g_cfg, "model": model_id}

    if provider == "mock":
        return {}

    raise ValueError(f"Unknown provider: {provider}")

# ---------------------------------------------------------------------------
# Call functions — return (raw_text, tokens_in, tokens_out, gen_time)
# ---------------------------------------------------------------------------

def make_caller(provider, model_id, client_state):
    if provider in ("deepseek", "openrouter"):
        client = client_state["client"]

        def call(row):
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=600,
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

    elif provider == "gemini":
        g_client = client_state["client"]
        g_cfg = client_state["config"]

        def call(row):
            t0 = time.time()
            resp = g_client.models.generate_content(
                model=model_id,
                contents=user_content(row),
                config=g_cfg,
            )
            t_in = getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
            t_out = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
            return resp.text, t_in, t_out, round(time.time() - t0, 2)

    elif provider == "mock":
        def call(row):
            time.sleep(0.005)
            answer = str(row["answer"])
            reasoning = f"The {row['task_type']} problem yields {answer} by the standard rule."
            raw = f"REASONING: {reasoning}\nANSWER: {answer}"
            return raw, 120, 40, 0.005

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return call

# ---------------------------------------------------------------------------
# Dry-run payload printer
# ---------------------------------------------------------------------------

def dry_run(df, provider, model_id, output_path):
    rows = df.head(3)
    print(f"DRY RUN — provider={provider}  model={model_id}")
    print(f"Would process {len(df)} total rows → {output_path}\n")

    for i, (_, row) in enumerate(rows.iterrows(), 1):
        print(f"--- Example {i}  id={row['id']}  task={row['task_type']} ---")

        if provider in ("deepseek", "openrouter"):
            payload = {
                "model": model_id,
                "max_tokens": 600,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": user_content(row)},
                ],
            }
        elif provider == "gemini":
            payload = {
                "model": model_id,
                "contents": user_content(row),
                "system_instruction": SYSTEM_PROMPT.strip(),
                "temperature": 0.0,
                "max_output_tokens": 600,
                "thinking_budget": 0,
            }
        elif provider == "mock":
            payload = {"model": "mock-v1", "user_content": user_content(row)}

        print(json.dumps(payload, indent=2))
        print()

    print("(No API calls made — remove --dry-run to execute.)")

# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------

def process_row(row, call_fn, model_id, output_path, write_lock, counters, counter_lock):
    for attempt in range(3):
        try:
            raw, tokens_in, tokens_out, gen_time = call_fn(row)
            reasoning, answer, parse_ok = parse_response(raw)
            correct = answer_correct(answer, str(row["answer"]), row["task_type"])

            result = {
                "id": row["id"],
                "prompt": row["prompt"],
                "reasoning": reasoning,
                "answer": answer,
                "task_type": row["task_type"],
                "gold_answer": str(row["answer"]),
                "model": model_id,
                "gen_time": gen_time,
                "parse_success": parse_ok,
                "answer_correct": correct,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }

            with write_lock:
                with open(output_path, "a") as f:
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
                _maybe_log(counters)

            return result

        except Exception as e:
            err = str(e)
            if any(k in err.lower() for k in ("429", "rate", "quota", "overloaded")):
                wait = 60 if "quota" in err.lower() else 30
                print(f"  [{row['id']}] Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  [{row['id']}] Error attempt {attempt+1}: {e}", flush=True)
                time.sleep(5)

    # All retries failed
    error_result = {
        "id": row["id"],
        "prompt": row["prompt"],
        "reasoning": "GENERATION_FAILED",
        "answer": "ERROR",
        "task_type": row["task_type"],
        "gold_answer": str(row["answer"]),
        "model": model_id,
        "gen_time": -1,
        "parse_success": False,
        "answer_correct": False,
        "tokens_in": 0,
        "tokens_out": 0,
    }
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(error_result) + "\n")
    with counter_lock:
        counters["done"] += 1
        counters["errors"] += 1
    return error_result


def _maybe_log(counters):
    done = counters["done"]
    if done % 100 != 0:
        return
    total = counters["total"]
    recent = counters["gen_times"][-100:]
    avg_t = sum(recent) / len(recent)
    eta_s = (total - done) * avg_t / max(counters["max_workers"], 1)
    task_summary = " ".join(
        f"{TASK_SHORT.get(tt, tt)}:{int(100*v['parse_ok']/v['done'])}%"
        for tt, v in sorted(counters["by_task"].items())
        if v["done"] > 0
    )
    c_in = counters["tokens_in"] * counters["cost_in"]
    c_out = counters["tokens_out"] * counters["cost_out"]
    cost_str = f"${c_in+c_out:.4f}" if counters["cost_known"] else "cost=n/a"
    print(
        f"[{done:04d}/{total}] {task_summary} | avg {avg_t:.1f}s | "
        f"est {eta_s/60:.0f}min | {cost_str}",
        flush=True,
    )


def run_batch(rows, call_fn, model_id, output_path, max_workers, counters, write_lock, counter_lock):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_row, row, call_fn, model_id, output_path, write_lock, counters, counter_lock
            ): row["id"]
            for _, row in rows.iterrows()
        }
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  Unhandled future error: {e}", flush=True)
        return results

# ---------------------------------------------------------------------------
# Pilot gate report
# ---------------------------------------------------------------------------

def print_gate_report(results, label):
    n = len(results)
    if n == 0:
        print("(No results to evaluate)")
        return True

    parse_ok = sum(r["parse_success"] for r in results)
    correct = sum(r.get("answer_correct", False) for r in results)
    parse_rate = parse_ok / n
    match_rate = correct / n

    print(f"\n{'='*60}")
    print(f"PILOT GATE REPORT ({label})")
    print(f"  Samples:    {n}")
    print(f"  Parse rate: {parse_ok}/{n}  {parse_rate:.1%}")
    print(f"  Match rate: {correct}/{n}  {match_rate:.1%}")

    by_task = defaultdict(lambda: [0, 0, 0])
    for r in results:
        tt = r["task_type"]
        by_task[tt][0] += 1
        if r["parse_success"]:
            by_task[tt][1] += 1
        if r.get("answer_correct", False):
            by_task[tt][2] += 1
    for tt, (d, p, c) in sorted(by_task.items()):
        print(f"  {tt:<20} parse={p}/{d}  match={c}/{d}")

    passed = parse_rate >= PILOT_GATE and match_rate >= PILOT_GATE
    if not passed:
        print(f"\nGATE: FAILED  (threshold {PILOT_GATE:.0%})")
        failures = [r for r in results if not r["parse_success"] or not r.get("answer_correct")]
        print(f"First 3 failures:")
        for r in failures[:3]:
            print(f"  id={r['id']} task={r['task_type']} "
                  f"parse={r['parse_success']} gold={r['gold_answer']!r} pred={r['answer']!r}")
    else:
        print(f"\nGATE: PASSED  (threshold {PILOT_GATE:.0%})")
    print("=" * 60)
    return passed


def print_final_summary(counters, elapsed, cfg):
    c_in = counters["tokens_in"] * counters["cost_in"]
    c_out = counters["tokens_out"] * counters["cost_out"]
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"  Rows this run:  {counters['done']}")
    print(f"  Errors:         {counters['errors']}")
    print(f"  Time:           {elapsed/60:.1f} min")
    print(f"  Tokens in/out:  {counters['tokens_in']:,} / {counters['tokens_out']:,}")
    if counters["cost_known"]:
        print(f"  Est. cost:      ${c_in+c_out:.4f}")
    else:
        print(f"  Est. cost:      n/a (pricing varies by OpenRouter model)")
    print(f"\nParse / match per task type:")
    for tt, v in sorted(counters["by_task"].items()):
        if v["done"] > 0:
            print(f"  {tt:<22} parse={v['parse_ok']}/{v['done']}  match={v['correct']}/{v['done']}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Phase 2 LLM reasoning-trace generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Inspect payloads without API calls
  python -m src.generate_llm --provider deepseek --dry-run

  # Pilot: 100 rows (required before bulk)
  python -m src.generate_llm --provider deepseek --max-rows 100

  # Full run after reviewing pilot
  python -m src.generate_llm --provider deepseek

  # Mock provider (no API key needed)
  python -m src.generate_llm --provider mock --max-rows 20
""",
    )
    parser.add_argument(
        "--provider", required=True, choices=list(PROVIDERS),
        help="LLM provider",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Max rows to process this session. Use 100 for pilot. Omit for full run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print request payloads for 3 examples; no API calls made.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override default model for the provider.",
    )
    parser.add_argument(
        "--input", default=INPUT_PATH,
        help=f"Input CSV (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL path (default: data/train_reasoning_v7_<provider>.jsonl)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override concurrent workers.",
    )
    parser.add_argument(
        "--skip-pilot-gate", action="store_true",
        help="Bypass the 'run pilot first' enforcement (for scripted bulk after manual review).",
    )
    args = parser.parse_args()

    provider = args.provider
    cfg = PROVIDERS[provider]

    if args.model is None:
        args.model = cfg["default_model"]
    if args.output is None:
        args.output = f"phase02_data_generation/data/train_reasoning_v7_{provider}.jsonl"
    max_workers = args.workers or cfg["default_workers"]

    # Validate max_rows
    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("ERROR: --max-rows must be >= 1")

    df = pd.read_csv(args.input)

    # Dry run: show payloads and exit (no key check needed)
    if args.dry_run:
        dry_run(df, provider, args.model, args.output)
        return

    # Fail loudly if API key is missing
    check_api_key(provider)

    # Load already-completed IDs for resume
    completed_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed_ids.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} already done, skipping.")

    already_done = len(completed_ids)
    to_process = df[~df["id"].astype(str).isin(completed_ids)].reset_index(drop=True)

    # Enforce pilot-first: block unlimited bulk if no prior pilot
    if not args.skip_pilot_gate and already_done < PILOT_N and args.max_rows is None:
        raise SystemExit(
            f"\nERROR: Cannot start bulk generation before pilot review.\n"
            f"\n  Step 1 — run pilot ({PILOT_N} rows):\n"
            f"    python -m src.generate_llm --provider {provider} --max-rows {PILOT_N}\n"
            f"\n  Step 2 — review output:\n"
            f"    {args.output}\n"
            f"\n  Step 3 — full run:\n"
            f"    python -m src.generate_llm --provider {provider}\n"
            f"\n  (Use --skip-pilot-gate to bypass this check.)\n"
        )

    # Apply session cap
    if args.max_rows is not None:
        to_process = to_process.head(args.max_rows)

    total_remaining = len(to_process)

    print(f"Provider:  {provider}  ({args.model})")
    print(f"Input:     {args.input}  ({len(df)} total rows)")
    print(f"Output:    {args.output}")
    print(f"To run:    {total_remaining} rows  ({already_done} already done)")
    if args.max_rows is not None:
        print(f"Session cap: --max-rows {args.max_rows}")
    print()

    if total_remaining == 0:
        print("Nothing to do — all rows already processed.")
        return

    client_state = build_client(provider, args.model)
    call_fn = make_caller(provider, args.model, client_state)
    write_lock = threading.Lock()
    counter_lock = threading.Lock()
    task_types = df["task_type"].dropna().unique().tolist()

    counters = {
        "done": 0,
        "errors": 0,
        "total": total_remaining,
        "tokens_in": 0,
        "tokens_out": 0,
        "gen_times": [],
        "max_workers": max_workers,
        "cost_in": cfg["cost_in"] or 0.0,
        "cost_out": cfg["cost_out"] or 0.0,
        "cost_known": cfg["cost_in"] is not None,
        "by_task": {tt: {"done": 0, "parse_ok": 0, "correct": 0} for tt in task_types},
    }

    t_start = time.time()
    is_capped_run = args.max_rows is not None

    print(f"Starting {'capped' if is_capped_run else 'full'} run "
          f"({total_remaining} rows @ {max_workers} workers)...\n")

    results = run_batch(
        to_process, call_fn, args.model, args.output,
        max_workers, counters, write_lock, counter_lock,
    )

    elapsed = time.time() - t_start
    print_final_summary(counters, elapsed, cfg)

    if is_capped_run:
        gate_ok = print_gate_report(results, f"{len(results)} rows")
        total_done_so_far = already_done + counters["done"]
        if gate_ok:
            print(f"\nPilot complete ({total_done_so_far} rows done total).")
            print(f"Review {args.output}, then run without --max-rows for bulk generation.")
        else:
            print(f"\nPilot quality gate FAILED. Fix prompt/provider before bulk generation.")
            sys.exit(1)


if __name__ == "__main__":
    main()
