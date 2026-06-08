#!/usr/bin/env python3
"""
10-row hard pilot for local Gemma 4 12B via llama.cpp OpenAI-compatible server.

Uses the 10 Fireworks-gate failed rows as the test set (these are the hardest rows
for the post_hoc_rationale_v8 task). Runs single-threaded with max_tokens=256.

Outputs:
  phase02_data_generation/data/v8/local_gemma_12b_hard10_outputs.jsonl
  phase02_data_generation/data/v8/local_gemma_12b_hard10_report.json

Run from project root:
  python phase02_data_generation/src/local_gemma_pilot.py [--dry-run] [--single]
"""

import argparse
import json
import os
import re
import subprocess
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERVER_BASE_URL = os.environ.get("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL_ID = os.environ.get("LOCAL_OPENAI_MODEL", "unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL")
MAX_TOKENS_INITIAL = 256
MAX_TOKENS_FALLBACK = 512
MAX_API_ERROR_RETRIES = 1  # retry only on server/transient errors, not parse fails

INPUT_CSV = "shared/data/raw/train_with_task_type.csv"
GATE_JSONL = "phase02_data_generation/data/train_reasoning_v7_fireworks.jsonl"
OUTPUT_JSONL = "phase02_data_generation/data/v8/local_gemma_12b_hard10_outputs.jsonl"
OUTPUT_REPORT = "phase02_data_generation/data/v8/local_gemma_12b_hard10_report.json"

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
# Helpers — inline copies matching generate_llm.py
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
# Server check
# ---------------------------------------------------------------------------

def check_server():
    import urllib.request
    try:
        with urllib.request.urlopen(f"{SERVER_BASE_URL.rstrip('/v1').rstrip('/')}/health", timeout=5) as r:
            body = json.loads(r.read())
            if body.get("status") != "ok":
                raise SystemExit(f"Server health check returned: {body}")
    except Exception as e:
        raise SystemExit(f"Cannot reach llama.cpp server at {SERVER_BASE_URL}: {e}")

    try:
        with urllib.request.urlopen(f"{SERVER_BASE_URL.rstrip('/v1').rstrip('/')}/props", timeout=5) as r:
            props = json.loads(r.read())
        n_ctx = props.get("default_generation_settings", {}).get("n_ctx", "?")
        total_slots = props.get("total_slots", "?")
        print(f"Server OK — model={props.get('model_alias','?')}  n_ctx={n_ctx}  slots={total_slots}")
        if n_ctx != "?" and int(n_ctx) > 8192:
            raise SystemExit(
                f"SAFETY STOP: server n_ctx={n_ctx} is too large for VRAM constraints. "
                f"Restart with --ctx-size 4096 or smaller."
            )
        if total_slots != "?" and int(total_slots) > 1:
            raise SystemExit(
                f"SAFETY STOP: server total_slots={total_slots} (n_parallel>1). "
                f"Restart with --parallel 1."
            )
        return props
    except SystemExit:
        raise
    except Exception as e:
        print(f"Warning: could not read /props: {e}")
        return {}


def vram_snapshot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True
        ).strip()
        used, total = [int(x.strip()) for x in out.split(",")]
        return {"used_mb": used, "total_mb": total}
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Load row IDs for pilot
# ---------------------------------------------------------------------------

def load_hard_row_ids(gate_jsonl):
    """Return IDs of rows that failed the Fireworks 1024 gate."""
    ids = []
    with open(gate_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("parse_success") or not r.get("answer_correct"):
                ids.append(r["id"])
    return ids

# ---------------------------------------------------------------------------
# Call with max_tokens=256; retry with 512 if response is truncated JSON
# ---------------------------------------------------------------------------

def call_row(client, row, dry_run=False):
    """Call the local server. Returns (raw_text, tokens_in, tokens_out, gen_time, max_tokens_used)."""
    max_tokens = MAX_TOKENS_INITIAL

    for api_attempt in range(MAX_API_ERROR_RETRIES + 1):
        t0 = time.time()
        try:
            if dry_run:
                payload = {
                    "model": MODEL_ID,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT.strip()},
                        {"role": "user", "content": user_content(row)},
                    ],
                }
                print(f"\nDRY RUN payload for id={row['id']} task={row['task_type']}:")
                print(json.dumps(payload, indent=2))
                return None, 0, 0, 0.0, max_tokens

            resp = client.chat.completions.create(
                model=MODEL_ID,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content(row)},
                ],
            )
            raw = resp.choices[0].message.content
            tok_in = resp.usage.prompt_tokens
            tok_out = resp.usage.completion_tokens
            elapsed = round(time.time() - t0, 2)

            # If output hit the token ceiling and looks like truncated JSON, retry at 512
            if tok_out >= max_tokens and max_tokens < MAX_TOKENS_FALLBACK:
                stripped = raw.strip()
                if not stripped.endswith("}") or stripped.count("{") != stripped.count("}"):
                    print(
                        f"  [{row['id']}] Output truncated at {tok_out} tokens "
                        f"— retrying at max_tokens={MAX_TOKENS_FALLBACK}",
                        flush=True,
                    )
                    max_tokens = MAX_TOKENS_FALLBACK
                    continue

            return raw, tok_in, tok_out, elapsed, max_tokens

        except Exception as e:
            err = str(e)
            wait = 60 if "quota" in err.lower() else (30 if any(k in err.lower() for k in ("429", "rate", "overloaded")) else 10)
            if api_attempt < MAX_API_ERROR_RETRIES:
                print(f"  [{row['id']}] API error (attempt {api_attempt+1}): {e} — retrying in {wait}s", flush=True)
                time.sleep(wait)
            else:
                print(f"  [{row['id']}] API error (final): {e}", flush=True)
                return None, 0, 0, round(time.time() - t0, 2), max_tokens

    # Exhausted retries (only max_tokens upgrade can loop here)
    return None, 0, 0, 0.0, max_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="10-row hard pilot: local Gemma 4 12B")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print one payload; no API call made.")
    parser.add_argument("--single", action="store_true",
                        help="Run only the first row as a real single-call test.")
    args = parser.parse_args()

    # ---- Step 0: server safety check ----------------------------------------
    if not args.dry_run:
        server_props = check_server()
    else:
        server_props = {}
        print("DRY RUN — skipping server check")

    # ---- Step 1: load rows ---------------------------------------------------
    hard_ids = load_hard_row_ids(GATE_JSONL)
    print(f"\nHard row IDs from Fireworks gate ({len(hard_ids)}): {hard_ids}")

    df = pd.read_csv(INPUT_CSV)
    subset = df[df["id"].astype(str).isin(set(hard_ids))].reset_index(drop=True)

    # Preserve order of hard_ids (hardest-first for visibility in output)
    id_order = {rid: i for i, rid in enumerate(hard_ids)}
    subset = subset.sort_values("id", key=lambda s: s.map(lambda x: id_order.get(str(x), 99)))
    subset = subset.reset_index(drop=True)

    print(f"Loaded {len(subset)} rows from CSV")

    if args.dry_run:
        client = None
    else:
        client = OpenAI(api_key="dummy", base_url=SERVER_BASE_URL)

    # ---- Step 2: dry-run payload print ---------------------------------------
    if args.dry_run:
        first_row = subset.iloc[0]
        call_row(client, first_row, dry_run=True)
        print("\n(No API calls made — remove --dry-run to execute.)")
        return

    # ---- Step 3: optional single-call smoke test ----------------------------
    if args.single:
        print(f"\n--- SINGLE CALL SMOKE TEST ---")
        first_row = subset.iloc[0]
        vram_before = vram_snapshot()
        raw, tok_in, tok_out, elapsed, mt_used = call_row(client, first_row)
        vram_after = vram_snapshot()
        if raw is None:
            print("FAILED — API call returned None")
            return
        parsed = parse_response(raw, str(first_row["answer"]), first_row["task_type"])
        print(f"\nRow: {first_row['id']}  task={first_row['task_type']}")
        print(f"max_tokens_used: {mt_used}  tok_in={tok_in}  tok_out={tok_out}  elapsed={elapsed}s")
        print(f"parse_ok={parsed['parsed_ok']}  answer_correct={parsed['answer_matches_gold']}")
        print(f"gold={first_row['answer']!r}")
        print(f"pred={parsed['answer']!r}")
        print(f"raw response:\n{raw!r}")
        if vram_before and vram_after:
            print(f"VRAM: {vram_before['used_mb']} MB → {vram_after['used_mb']} MB / {vram_after['total_mb']} MB")
        if parsed["parsed_ok"] and parsed["answer_matches_gold"]:
            print("\nSingle call: GOOD — proceed with 10-row pilot? (re-run without --single)")
        else:
            print("\nSingle call: FAILED — review output before running full pilot")
        return

    # ---- Step 4: 10-row hard pilot -------------------------------------------
    print(f"\nStarting 10-row hard pilot (max_tokens={MAX_TOKENS_INITIAL}, workers=1)...\n")
    vram_start = vram_snapshot()

    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)

    results = []
    for _, row in subset.iterrows():
        raw, tok_in, tok_out, elapsed, mt_used = call_row(client, row)
        vram_mid = vram_snapshot()

        if raw is None:
            result = {
                "id":              row["id"],
                "task_type":       row["task_type"],
                "gold_answer":     str(row["answer"]),
                "parse_success":   False,
                "answer_correct":  False,
                "answer":          "ERROR",
                "reasoning":       "GENERATION_FAILED",
                "tokens_in":       tok_in,
                "tokens_out":      tok_out,
                "gen_time":        elapsed,
                "model":           MODEL_ID,
                "max_tokens_used": mt_used,
                "vram_used_mb":    vram_mid["used_mb"] if vram_mid else None,
            }
        else:
            parsed = parse_response(raw, str(row["answer"]), row["task_type"])
            result = {
                "id":              row["id"],
                "task_type":       row["task_type"],
                "gold_answer":     str(row["answer"]),
                "parse_success":   parsed["parsed_ok"],
                "answer_correct":  parsed["answer_matches_gold"],
                "answer":          parsed["answer"],
                "reasoning":       parsed["reasoning"],
                "raw":             raw,
                "tokens_in":       tok_in,
                "tokens_out":      tok_out,
                "gen_time":        elapsed,
                "model":           MODEL_ID,
                "max_tokens_used": mt_used,
                "vram_used_mb":    vram_mid["used_mb"] if vram_mid else None,
            }

        results.append(result)

        status = "GOOD" if (result["parse_success"] and result["answer_correct"]) else "FAIL"
        print(
            f"  [{row['id']}] task={row['task_type']:<20} "
            f"parse={result['parse_success']} correct={result['answer_correct']} "
            f"tok_out={result['tokens_out']} mt={result['max_tokens_used']} "
            f"t={result['gen_time']}s -> {status}",
            flush=True,
        )

        with open(OUTPUT_JSONL, "a") as f:
            f.write(json.dumps(result) + "\n")

    vram_end = vram_snapshot()

    # ---- Step 5: report -------------------------------------------------------
    n = len(results)
    good = [r for r in results if r["parse_success"] and r["answer_correct"]]
    parse_ok = [r for r in results if r["parse_success"]]
    avg_tok_out = sum(r["tokens_out"] for r in results) / n
    avg_latency = sum(r["gen_time"] for r in results) / n
    max_vram = max((r["vram_used_mb"] for r in results if r.get("vram_used_mb")), default=None)
    mt_used_counts = {}
    for r in results:
        k = r["max_tokens_used"]
        mt_used_counts[k] = mt_used_counts.get(k, 0) + 1

    by_task = {}
    for r in results:
        tt = r["task_type"]
        if tt not in by_task:
            by_task[tt] = {"n": 0, "parse_ok": 0, "correct": 0}
        by_task[tt]["n"] += 1
        if r["parse_success"]:
            by_task[tt]["parse_ok"] += 1
        if r["answer_correct"]:
            by_task[tt]["correct"] += 1

    failures = [r for r in results if not r["parse_success"] or not r["answer_correct"]]
    fail_examples = []
    for r in failures:
        fail_type = "parse_fail" if not r["parse_success"] else "copy_fail"
        fail_examples.append({
            "id":        r["id"],
            "task_type": r["task_type"],
            "fail_type": fail_type,
            "tok_out":   r["tokens_out"],
            "gold":      r["gold_answer"],
            "pred":      r["answer"],
        })

    report = {
        "mode":            "post_hoc_rationale_v8",
        "mode_note":       "Gold answer provided in prompt. answer_correct measures copy fidelity, NOT model-solving accuracy.",
        "model":           MODEL_ID,
        "server_base_url": SERVER_BASE_URL,
        "timestamp_utc":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows_total":      n,
        "parse_ok":        len(parse_ok),
        "parse_rate":      len(parse_ok) / n,
        "answer_correct":  len(good),
        "answer_copy_rate": len(good) / n,
        "avg_tokens_out":  round(avg_tok_out, 1),
        "avg_latency_s":   round(avg_latency, 2),
        "max_tokens_per_call": mt_used_counts,
        "vram_start_mb":   vram_start["used_mb"] if vram_start else None,
        "vram_max_mb":     max_vram,
        "vram_end_mb":     vram_end["used_mb"] if vram_end else None,
        "vram_total_mb":   vram_end["total_mb"] if vram_end else None,
        "oom_observed":    False,
        "per_task":        by_task,
        "fail_examples":   fail_examples,
        "pilot_gate_threshold": 0.95,
        "pilot_gate_passed": len(good) / n >= 0.95,
    }

    print(f"\n{'='*60}")
    print(f"LOCAL GEMMA 4 12B — 10-ROW HARD PILOT REPORT")
    print(f"  Model:          {MODEL_ID}")
    print(f"  Rows:           {n}")
    print(f"  Parse OK:       {len(parse_ok)}/{n}  ({report['parse_rate']:.0%})")
    print(f"  Answer copy OK: {len(good)}/{n}  ({report['answer_copy_rate']:.0%})")
    print(f"  Avg tok_out:    {report['avg_tokens_out']}")
    print(f"  Avg latency:    {report['avg_latency_s']}s/row")
    print(f"  max_tokens used: {mt_used_counts}")
    if vram_end:
        print(f"  VRAM: start={report['vram_start_mb']} MB  max={max_vram} MB  "
              f"end={report['vram_end_mb']} MB / {report['vram_total_mb']} MB")
    print()
    print(f"  Per-task:")
    for tt, v in sorted(by_task.items()):
        print(f"    {tt:<22} parse={v['parse_ok']}/{v['n']}  correct={v['correct']}/{v['n']}")
    if fail_examples:
        print(f"\n  Failures ({len(fail_examples)}):")
        for fe in fail_examples:
            print(f"    id={fe['id']} task={fe['task_type']} type={fe['fail_type']} "
                  f"tok_out={fe['tok_out']} gold={fe['gold']!r} pred={fe['pred']!r}")
    print(f"\n  Gate ({report['pilot_gate_threshold']:.0%} threshold): "
          f"{'PASSED' if report['pilot_gate_passed'] else 'FAILED'}")
    print("=" * 60)

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSONL: {OUTPUT_JSONL}")
    print(f"Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
