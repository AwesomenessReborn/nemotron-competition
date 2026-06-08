#!/usr/bin/env python3
"""
Local Gemma 4 12B pilot via llama.cpp raw /completion endpoint.

Uses /completion (not /v1/chat/completions) to bypass the thinking-channel
stripping that occurs when reasoning_format=none + Gemma 4 chat template are
combined. Raw endpoint returns the full token stream including channel markers,
which we strip ourselves.

Prompt: minimal post-hoc rationale format (simpler than A++, no escape examples).
Schema: {"answer":"...", "reasoning":"..."}

Hard 10-row test: the 10 rows that failed the Fireworks 1024 gate.

Outputs:
  phase02_data_generation/data/v8/local_gemma_12b_hard10_outputs.jsonl
  phase02_data_generation/data/v8/local_gemma_12b_hard10_report.json

Run from project root:
  python phase02_data_generation/src/local_gemma_completion_pilot.py
  python phase02_data_generation/src/local_gemma_completion_pilot.py --probe   # single row first
  python phase02_data_generation/src/local_gemma_completion_pilot.py --dry-run # print prompt only
"""

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMPLETION_URL = os.environ.get("LOCAL_COMPLETION_URL", "http://127.0.0.1:8080/completion")
MODEL_ALIAS    = os.environ.get("LOCAL_OPENAI_MODEL", "unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL")

N_PREDICT   = 512
TEMPERATURE = 0.0
# Stop sequences to prevent over-generation after the JSON closing brace.
STOP_SEQS = ["<turn|>", "<|turn>", "<eos>", "</s>", "}\n\n", "}\n\nWait", "} \n\n"]

INPUT_CSV     = "shared/data/raw/train_with_task_type.csv"
GATE_JSONL    = "phase02_data_generation/data/train_reasoning_v7_fireworks.jsonl"

# Default output paths (hard-10 mode); gate mode uses its own paths
OUTPUT_JSONL  = "phase02_data_generation/data/v8/local_gemma_12b_hard10_outputs.jsonl"
OUTPUT_REPORT = "phase02_data_generation/data/v8/local_gemma_12b_hard10_report.json"

GATE_OUTPUT_JSONL  = "phase02_data_generation/data/v8/local_gemma_12b_gate100_outputs.jsonl"
GATE_OUTPUT_REPORT = "phase02_data_generation/data/v8/local_gemma_12b_gate100_report.json"

MODEL_LABEL = "local_gemma4_12b_completion_posthoc_v8"

SYM_REPAIR_OUTPUT_JSONL  = "phase02_data_generation/data/v8/local_gemma_12b_symrepair_outputs.jsonl"
SYM_REPAIR_OUTPUT_REPORT = "phase02_data_generation/data/v8/local_gemma_12b_symrepair_report.json"
SYM_N_PREDICT = 256

# Minimal post-hoc rationale prompt — no escape examples, explicit "do not solve"
SYSTEM_PROMPT = """You output exactly one JSON object.
The answer is already solved.
Copy CORRECT_ANSWER exactly.
Do not solve.
Do not include hidden reasoning.
Do not explain outside JSON.
Schema: {"answer":"...", "reasoning":"..."}
Reasoning must be one short sentence."""

# Strict copy prompt for symbol_transform — sentinel-bounded answer, fixed reasoning
SYMBOL_SYSTEM_PROMPT = """You output exactly one JSON object.
The answer is already solved.
Your only job is to copy the exact answer between <<ANSWER>> and <</ANSWER>>.
Copy every character exactly, including backslashes, brackets, parentheses, quotes, apostrophes, spaces, and punctuation.
Do not infer a rule.
Do not transform the symbols.
Do not repair or normalize the answer.
Do not explain the puzzle rule.
Use this JSON schema exactly:
{"answer":"...", "reasoning":"..."}

The reasoning field must be exactly:
"The correct answer was provided, so I copied it exactly." """

# ---------------------------------------------------------------------------
# Prompt formatting — Gemma 4 turn format (manual, not via chat template)
# ---------------------------------------------------------------------------

def format_prompt(gold_answer, puzzle_prompt):
    """Format a Gemma 4 turn-based prompt for the raw /completion endpoint."""
    user_content = f"CORRECT_ANSWER: {gold_answer}\n\nPUZZLE:\n{puzzle_prompt}"
    return (
        f"<bos><|turn>system\n{SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n{user_content}<turn|>\n"
        f"<|turn>model\n"
    )


def format_symbol_prompt(gold_answer, puzzle_prompt):
    """Strict copy prompt for symbol_transform rows — uses <<ANSWER>> sentinels.

    Uses double-angle-bracket delimiters (not XML tags) to avoid the model
    confusing </ANSWER> with JSON content.
    """
    user_content = (
        f"CORRECT_ANSWER_BETWEEN_TAGS: <<ANSWER>>{gold_answer}<</ANSWER>>\n\n"
        f"PUZZLE:\n{puzzle_prompt}"
    )
    return (
        f"<bos><|turn>system\n{SYMBOL_SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n{user_content}<turn|>\n"
        f"<|turn>model\n"
    )

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def strip_thinking(text):
    """Remove Gemma 4 thinking-channel markers and their content."""
    return re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL).strip()


def parse_response(raw, gold, task_type):
    """
    Strip thinking tokens, extract JSON, compare answer to gold.

    Uses raw_decode to parse the FIRST valid JSON object in the text, ignoring
    any trailing content the model generates after the closing brace.

    Returns dict: parsed_ok, answer, answer_matches_gold, reasoning, cleaned.
    """
    cleaned = strip_thinking(raw)
    # Remove markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    answer = "PARSE_ERROR"
    reasoning = ""
    parsed_ok = False
    decoder = json.JSONDecoder()

    # Primary: raw_decode from the first '{' — handles trailing text gracefully
    brace_idx = cleaned.find("{")
    if brace_idx >= 0:
        try:
            obj, _ = decoder.raw_decode(cleaned, brace_idx)
            answer    = str(obj.get("answer", "")).strip()
            reasoning = str(obj.get("reasoning", "")).strip()
            if answer:
                parsed_ok = True
        except json.JSONDecodeError:
            pass

    if not parsed_ok:
        # Greedy fallback: recover answer when JSON is entirely malformed
        m = re.search(r'"answer"\s*:\s*"(.*?)"', cleaned, re.DOTALL)
        if m:
            answer = m.group(1)
            parsed_ok = True

    matches_gold = _answer_correct(answer, gold, task_type) if parsed_ok else False
    return {
        "parsed_ok":           parsed_ok,
        "answer":              answer,
        "answer_matches_gold": matches_gold,
        "reasoning":           reasoning,
        "cleaned":             cleaned,
    }


def _answer_correct(pred, gold, task_type):
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
# Server / VRAM helpers
# ---------------------------------------------------------------------------

def check_server():
    base = COMPLETION_URL.replace("/completion", "")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            body = json.loads(r.read())
        assert body.get("status") == "ok", f"health={body}"
    except Exception as e:
        raise SystemExit(f"llama.cpp server not reachable at {base}: {e}")

    try:
        with urllib.request.urlopen(f"{base}/props", timeout=5) as r:
            props = json.loads(r.read())
        n_ctx       = props.get("default_generation_settings", {}).get("n_ctx", "?")
        total_slots = props.get("total_slots", "?")
        alias       = props.get("model_alias", "?")
        print(f"Server OK — model={alias}  n_ctx={n_ctx}  slots={total_slots}")
        if n_ctx != "?" and int(n_ctx) > 8192:
            raise SystemExit(f"SAFETY STOP: n_ctx={n_ctx} exceeds VRAM budget. Restart with --ctx-size ≤4096.")
        if total_slots != "?" and int(total_slots) > 1:
            raise SystemExit(f"SAFETY STOP: slots={total_slots} (n_parallel>1). Restart with --parallel 1.")
    except SystemExit:
        raise
    except Exception as e:
        print(f"Warning: /props unavailable: {e}")


def vram_snapshot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        ).strip()
        used, total = [int(x.strip()) for x in out.split(",")]
        return {"used_mb": used, "total_mb": total}
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Single row call
# ---------------------------------------------------------------------------

def call_row(row, n_predict=N_PREDICT, task_type=None):
    """
    POST to /completion, return (raw_text, tokens_predicted, elapsed, n_predict_used).
    Returns (None, 0, elapsed, n_predict) on failure.
    Retries once at 1024 if output is truncated at the requested n_predict ceiling.
    Uses the symbol-specific strict-copy prompt when task_type=='symbol_transform'.
    """
    tt = task_type or row.get("task_type", "")
    if tt == "symbol_transform":
        prompt = format_symbol_prompt(str(row["answer"]), row["prompt"])
    else:
        prompt = format_prompt(str(row["answer"]), row["prompt"])
    fallback = max(1024, n_predict * 2)

    for attempt_n_predict in (n_predict, fallback):
        payload = json.dumps({
            "prompt":      prompt,
            "n_predict":   attempt_n_predict,
            "temperature": TEMPERATURE,
            "stop":        STOP_SEQS,
            "cache_prompt": False,
        }).encode()

        t0 = time.time()
        try:
            req = urllib.request.Request(
                COMPLETION_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            print(f"  [{row['id']}] HTTP {e.code}: {body[:200]}", flush=True)
            return None, 0, round(time.time() - t0, 2), attempt_n_predict
        except Exception as e:
            print(f"  [{row['id']}] Request error: {e}", flush=True)
            return None, 0, round(time.time() - t0, 2), attempt_n_predict

        elapsed = round(time.time() - t0, 2)
        raw     = resp.get("content", "")
        tok_out = resp.get("tokens_predicted", 0)
        stopped = resp.get("stopped_limit", False)   # True when n_predict ceiling hit

        # If we hit the ceiling and output looks truncated, retry at the fallback limit
        if stopped and attempt_n_predict < fallback:
            cleaned = strip_thinking(raw)
            looks_truncated = not cleaned.rstrip().endswith("}")
            if looks_truncated:
                print(
                    f"  [{row['id']}] Output truncated at {tok_out} tokens "
                    f"— retrying at n_predict=512",
                    flush=True,
                )
                continue  # retry with 512

        return raw, tok_out, elapsed, attempt_n_predict

    return None, 0, 0.0, 512


# ---------------------------------------------------------------------------
# Load failed row IDs from Fireworks gate
# ---------------------------------------------------------------------------

def load_hard_row_ids(gate_jsonl):
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
# Report
# ---------------------------------------------------------------------------

def print_and_save_report(
    results, elapsed_total,
    probe_mode=False,
    gate_mode=False,
    out_jsonl=OUTPUT_JSONL,
    out_report=OUTPUT_REPORT,
    n_predict=N_PREDICT,
):
    n         = len(results)
    good      = [r for r in results if r["parse_success"] and r["answer_correct"]]
    parse_ok  = [r for r in results if r["parse_success"]]
    avg_tok   = sum(r["tokens_out"] for r in results) / n if n else 0
    avg_lat   = sum(r["gen_time"] for r in results) / n if n else 0
    max_vram  = max((r["vram_used_mb"] for r in results if r.get("vram_used_mb")), default=None)
    vram_snap = vram_snapshot()

    by_task: dict = {}
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
    fail_examples = [
        {
            "id":        r["id"],
            "task_type": r["task_type"],
            "fail_type": "parse_fail" if not r["parse_success"] else "copy_fail",
            "tok_out":   r["tokens_out"],
            "gold":      r["gold_answer"],
            "pred":      r["answer"],
            "cleaned":   r.get("cleaned", ""),
        }
        for r in failures
    ]

    # Raw examples (first 3 successes)
    raw_examples = [
        {
            "id":       r["id"],
            "task_type": r["task_type"],
            "gold":     r["gold_answer"],
            "pred":     r["answer"],
            "reasoning": r["reasoning"],
            "tok_out":  r["tokens_out"],
            "raw_snippet": r.get("raw", "")[:300],
        }
        for r in results
        if r["parse_success"] and r["answer_correct"]
    ][:3]

    gate_ok = (len(good) / n >= 0.95) if n else False
    proj_9500_h = round(avg_lat * 9500 / 3600, 1) if avg_lat else None

    if gate_mode:
        if gate_ok:
            recommendation = "PASS — consider as primary or parallel path for full 9,500-row run"
        elif len(good) / n >= 0.90:
            recommendation = "NEAR-PASS — review failures; may serve as fallback for Fireworks failures"
        elif len(good) / n >= 0.70:
            recommendation = "PARTIAL — useful as fallback on specific task types only"
        else:
            recommendation = "REJECTED — too many failures for production use"
    else:
        recommendation = (
            "YES — consider for 100-row gate"
            if gate_ok
            else (
                "PARTIAL — good on some tasks; review failures before gate"
                if len(good) / n >= 0.7
                else "NO — too many failures; investigate prompt or model"
            )
        ) if n else "N/A"

    report = {
        "mode":             "post_hoc_rationale_v8",
        "mode_note":        "Gold answer provided in prompt. answer_correct = copy fidelity, NOT model-solving.",
        "model_label":      MODEL_LABEL,
        "endpoint":         COMPLETION_URL,
        "model_alias":      MODEL_ALIAS,
        "prompt_format":    "gemma4_manual_turn",
        "system_prompt":    SYSTEM_PROMPT,
        "n_predict":        n_predict,
        "temperature":      TEMPERATURE,
        "timestamp_utc":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows_total":       n,
        "parse_ok":         len(parse_ok),
        "parse_rate":       round(len(parse_ok) / n, 4) if n else 0,
        "answer_correct":   len(good),
        "answer_copy_rate": round(len(good) / n, 4) if n else 0,
        "avg_tokens_out":   round(avg_tok, 1),
        "avg_latency_s":    round(avg_lat, 2),
        "total_elapsed_s":  round(elapsed_total, 1),
        "projected_9500_rows_h": proj_9500_h,
        "vram_max_mb":      max_vram,
        "vram_end_mb":      vram_snap["used_mb"] if vram_snap else None,
        "vram_total_mb":    vram_snap["total_mb"] if vram_snap else None,
        "per_task":         by_task,
        "fail_examples":    fail_examples,
        "raw_examples":     raw_examples,
        "pilot_gate_threshold": 0.95,
        "pilot_gate_passed":    gate_ok,
        "recommendation":   recommendation,
    }

    if probe_mode:
        label = "PROBE"
    elif gate_mode:
        label = "100-ROW GATE"
    else:
        label = "10-ROW HARD PILOT"

    print(f"\n{'='*60}")
    print(f"LOCAL GEMMA 4 12B — {label} REPORT")
    print(f"  Model label:    {MODEL_LABEL}")
    print(f"  Endpoint:       {COMPLETION_URL}")
    print(f"  n_predict:      {n_predict}")
    print(f"  Rows:           {n}")
    print(f"  Parse OK:       {len(parse_ok)}/{n}  ({report['parse_rate']:.0%})")
    print(f"  Answer copy OK: {len(good)}/{n}  ({report['answer_copy_rate']:.0%})")
    print(f"  Avg tok_out:    {report['avg_tokens_out']}")
    print(f"  Avg latency:    {report['avg_latency_s']}s/row")
    if proj_9500_h:
        print(f"  Projected 9,500-row time: ~{proj_9500_h}h  (workers=1)")
    if max_vram:
        print(f"  VRAM max:       {max_vram} MB / {report['vram_total_mb']} MB")
    print()
    print("  Per-task:")
    for tt, v in sorted(by_task.items()):
        bad_pct = (v["n"] - v["correct"]) / v["n"] if v["n"] else 0
        flag = "  <-- >10% bad" if bad_pct > 0.10 else ""
        print(f"    {tt:<22}  parse={v['parse_ok']}/{v['n']}  copy={v['correct']}/{v['n']}{flag}")
    if fail_examples:
        print(f"\n  Failures ({len(fail_examples)}):")
        for fe in fail_examples:
            print(
                f"    id={fe['id']} task={fe['task_type']} type={fe['fail_type']} "
                f"tok={fe['tok_out']}  gold={fe['gold']!r}  pred={fe['pred']!r}"
            )
    if raw_examples:
        print(f"\n  Good examples (first {len(raw_examples)}):")
        for ex in raw_examples:
            print(f"    id={ex['id']} task={ex['task_type']} tok={ex['tok_out']}")
            print(f"      reasoning: {ex['reasoning']!r}")
            print(f"      gold={ex['gold']!r}  pred={ex['pred']!r}")
    print(f"\n  Gate (95% threshold): {'PASSED' if gate_ok else 'FAILED'}")
    print(f"  Recommendation: {recommendation}")
    print("=" * 60)

    if not probe_mode:
        with open(out_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSONL:  {out_jsonl}")
        print(f"Report: {out_report}")

    return report

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_rows(subset, n_predict, out_jsonl, out_report, label, gate_mode=False):
    """Execute the generation loop, write JSONL, print+save report."""
    import sys

    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    results = []
    t_start = time.time()

    for _, row in subset.iterrows():
        vram_post = None
        raw, tok_out, elapsed, n_used = call_row(row, n_predict=n_predict)
        vram_snap_row = vram_snapshot()
        vram_used = vram_snap_row["used_mb"] if vram_snap_row else None

        if raw is None:
            result = {
                "id":             row["id"],
                "task_type":      row["task_type"],
                "gold_answer":    str(row["answer"]),
                "parse_success":  False,
                "answer_correct": False,
                "answer":         "ERROR",
                "reasoning":      "GENERATION_FAILED",
                "cleaned":        "",
                "raw":            "",
                "tokens_out":     tok_out,
                "gen_time":       elapsed,
                "n_predict_used": n_used,
                "model":          MODEL_ALIAS,
                "vram_used_mb":   vram_used,
            }
        else:
            parsed = parse_response(raw, str(row["answer"]), row["task_type"])
            result = {
                "id":             row["id"],
                "task_type":      row["task_type"],
                "gold_answer":    str(row["answer"]),
                "parse_success":  parsed["parsed_ok"],
                "answer_correct": parsed["answer_matches_gold"],
                "answer":         parsed["answer"],
                "reasoning":      parsed["reasoning"],
                "cleaned":        parsed["cleaned"],
                "raw":            raw,
                "tokens_out":     tok_out,
                "gen_time":       elapsed,
                "n_predict_used": n_used,
                "model":          MODEL_ALIAS,
                "vram_used_mb":   vram_used,
            }

        results.append(result)

        status = "GOOD" if (result["parse_success"] and result["answer_correct"]) else "FAIL"
        done = len(results)
        total = len(subset)
        print(
            f"  [{done:3}/{total}] [{row['id']}] task={row['task_type']:<20} "
            f"parse={result['parse_success']} copy={result['answer_correct']} "
            f"tok={result['tokens_out']} t={result['gen_time']}s -> {status}",
            flush=True,
        )

        with open(out_jsonl, "a") as f:
            f.write(json.dumps(result) + "\n")

    elapsed_total = time.time() - t_start
    report = print_and_save_report(
        results, elapsed_total,
        gate_mode=gate_mode,
        out_jsonl=out_jsonl,
        out_report=out_report,
        n_predict=n_predict,
    )

    if gate_mode and not report["pilot_gate_passed"]:
        sys.exit(1)

    return report


def _build_sym_repair_subset(df):
    """
    Build the symbol_transform repair pilot set:
      - All 12 sym rows from the 100-row gate (first 100 CSV rows)
      - Up to 20 additional random sym rows not in the gate set
    Returns a shuffled-but-stable DataFrame (failed rows first).
    """
    gate_ids = set(df.head(100)["id"].astype(str))
    gate_sym = df.head(100)[df.head(100)["task_type"] == "symbol_transform"].copy()

    FAILED_IDS = {"0133bcec", "01ef1e3e", "022c4d73", "02664ad5"}
    extra_pool = df[
        (df["task_type"] == "symbol_transform") &
        (~df["id"].astype(str).isin(gate_ids))
    ].copy()
    extra = extra_pool.sample(n=min(20, len(extra_pool)), random_state=42)

    failed_mask = gate_sym["id"].astype(str).isin(FAILED_IDS)
    ordered = pd.concat([
        gate_sym[failed_mask],    # 4 known failures first
        gate_sym[~failed_mask],   # remaining 8 gate sym rows
        extra,                    # 20 extra
    ]).reset_index(drop=True)
    return ordered


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Local Gemma 4 12B pilot/gate via /completion endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  (default)    10-row hard pilot on Fireworks gate failures
  --gate       100-row gate on same rows as the Fireworks 1024 gate
  --sym-repair symbol_transform repair pilot (12 gate rows + 20 extra)
  --probe      Single-row call check (first row of selected set)
  --dry-run    Print prompt only, no API call
""",
    )
    parser.add_argument("--gate", action="store_true",
                        help="Run 100-row gate on same row set as Fireworks gate.")
    parser.add_argument("--sym-repair", action="store_true",
                        help="Run symbol_transform repair pilot with strict copy prompt.")
    parser.add_argument("--probe", action="store_true",
                        help="Single-row call check (first row of selected set).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print formatted prompt for first row; no API call.")
    parser.add_argument("--n-predict", type=int, default=None,
                        help=f"Max tokens to generate per row (default: {N_PREDICT} general, "
                             f"{SYM_N_PREDICT} sym-repair).")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows processed (gate mode only).")
    args = parser.parse_args()

    # ---- Mode-specific defaults ---------------------------------------------
    if args.n_predict is not None:
        n_predict = args.n_predict
    elif args.sym_repair:
        n_predict = SYM_N_PREDICT
    else:
        n_predict = N_PREDICT

    # ---- Server check -------------------------------------------------------
    if not args.dry_run:
        check_server()

    df = pd.read_csv(INPUT_CSV)

    # ---- Row selection -------------------------------------------------------
    if args.sym_repair:
        subset     = _build_sym_repair_subset(df)
        out_jsonl  = SYM_REPAIR_OUTPUT_JSONL
        out_report = SYM_REPAIR_OUTPUT_REPORT
        print(f"\nSYM-REPAIR MODE: {len(subset)} symbol_transform rows "
              f"(4 known failures + {len(subset)-4} others), n_predict={n_predict}")
    elif args.gate:
        max_rows   = args.max_rows or 100
        subset     = df.head(max_rows).reset_index(drop=True)
        out_jsonl  = GATE_OUTPUT_JSONL
        out_report = GATE_OUTPUT_REPORT
        print(f"\nGATE MODE: first {len(subset)} rows from CSV (matches Fireworks gate set)")
    else:
        hard_ids  = load_hard_row_ids(GATE_JSONL)
        id_to_idx = {rid: i for i, rid in enumerate(hard_ids)}
        subset = (
            df[df["id"].astype(str).isin(set(hard_ids))]
            .copy()
            .sort_values("id", key=lambda s: s.map(lambda x: id_to_idx.get(str(x), 99)))
            .reset_index(drop=True)
        )
        if args.max_rows:
            subset = subset.head(args.max_rows)
        out_jsonl  = OUTPUT_JSONL
        out_report = OUTPUT_REPORT
        print(f"\nHARD-10 MODE: {len(subset)} Fireworks-failed rows")

    print(f"n_predict={n_predict}  temperature={TEMPERATURE}  workers=1\n")

    # ---- Dry run -------------------------------------------------------------
    if args.dry_run:
        row = subset.iloc[0]
        tt = row.get("task_type", "")
        if tt == "symbol_transform":
            prompt = format_symbol_prompt(str(row["answer"]), row["prompt"])
            prompt_variant = "SYMBOL (strict-copy sentinel)"
        else:
            prompt = format_prompt(str(row["answer"]), row["prompt"])
            prompt_variant = "GENERAL (minimal post-hoc)"
        print(f"--- Prompt for id={row['id']} task={tt} [{prompt_variant}] ---")
        print(prompt)
        print(f"\n(n_predict={n_predict}, stop={STOP_SEQS})")
        print("(No API call made.)")
        return

    # ---- Single probe -------------------------------------------------------
    if args.probe:
        row = subset.iloc[0]
        print(f"--- PROBE: id={row['id']} task={row['task_type']} gold={row['answer']!r} ---\n")
        vram_before = vram_snapshot()
        raw, tok_out, elapsed, n_used = call_row(row, n_predict=n_predict)
        vram_after  = vram_snapshot()

        if raw is None:
            print("PROBE FAILED — no response from server")
            return

        parsed = parse_response(raw, str(row["answer"]), row["task_type"])
        print(f"n_predict_used : {n_used}")
        print(f"tokens_out     : {tok_out}")
        print(f"latency        : {elapsed}s")
        print(f"parse_ok       : {parsed['parsed_ok']}")
        print(f"answer_copy_ok : {parsed['answer_matches_gold']}")
        print(f"gold           : {row['answer']!r}")
        print(f"pred           : {parsed['answer']!r}")
        print(f"reasoning      : {parsed['reasoning']!r}")
        print(f"raw (full)     :\n{raw!r}")
        print(f"cleaned        :\n{parsed['cleaned']!r}")
        if vram_before and vram_after:
            print(f"VRAM           : {vram_before['used_mb']} → {vram_after['used_mb']} MB "
                  f"/ {vram_after['total_mb']} MB")

        if parsed["parsed_ok"] and parsed["answer_matches_gold"]:
            print("\nPROBE PASSED")
        else:
            print("\nPROBE FAILED — review output above")
        return

    # ---- Main run -----------------------------------------------------------
    gate_mode = args.gate
    if args.sym_repair:
        label = "SYM-REPAIR PILOT"
    elif gate_mode:
        label = "100-ROW GATE"
    else:
        label = "10-ROW HARD PILOT"

    print(f"Starting {label}  ({len(subset)} rows, n_predict={n_predict}, workers=1)...\n")
    report = run_rows(subset, n_predict, out_jsonl, out_report, label, gate_mode=False)

    if args.sym_repair:
        sym_rate = report["answer_copy_rate"]
        if sym_rate >= 0.95:
            print(f"\nSYM-REPAIR PASSED ({sym_rate:.0%}) — symbol_transform prompt ready for full run.")
        else:
            print(f"\nSYM-REPAIR FAILED ({sym_rate:.0%} < 95%) — review failures before full run.")


if __name__ == "__main__":
    main()
