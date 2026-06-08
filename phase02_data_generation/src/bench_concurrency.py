#!/usr/bin/env python3
"""
Concurrency benchmark for the local Gemma 4 12B /completion endpoint.

Runs N rows with configurable workers, reports per-row latency, throughput,
wall-clock time, VRAM peak, and output quality (parse + copy rates).

Run from project root:
  python phase02_data_generation/src/bench_concurrency.py
  python phase02_data_generation/src/bench_concurrency.py --workers 4
  python phase02_data_generation/src/bench_concurrency.py --workers 1 --rows 20
"""

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMPLETION_URL = os.environ.get("LOCAL_COMPLETION_URL", "http://127.0.0.1:8080/completion")
INPUT_CSV      = "shared/data/raw/train_with_task_type.csv"

BENCH_OUTPUT_JSONL  = "phase02_data_generation/data/v8/bench_concurrency_results.jsonl"
BENCH_OUTPUT_REPORT = "phase02_data_generation/data/v8/bench_concurrency_report.json"

N_PREDICT   = 512
TEMPERATURE = 0.0
STOP_SEQS   = ["<turn|>", "<|turn>", "<eos>", "</s>", "}\n\n", "}\n\nWait", "} \n\n"]

SYSTEM_PROMPT = """You output exactly one JSON object.
The answer is already solved.
Copy CORRECT_ANSWER exactly.
Do not solve.
Do not include hidden reasoning.
Do not explain outside JSON.
Schema: {"answer":"...", "reasoning":"..."}
Reasoning must be one short sentence."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_prompt(gold_answer, puzzle_prompt):
    user_content = f"CORRECT_ANSWER: {gold_answer}\n\nPUZZLE:\n{puzzle_prompt}"
    return (
        f"<bos><|turn>system\n{SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n{user_content}<turn|>\n"
        f"<|turn>model\n"
    )


def strip_thinking(text):
    return re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL).strip()


def parse_response(raw, gold, task_type):
    cleaned = strip_thinking(raw)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    answer, reasoning, parsed_ok = "PARSE_ERROR", "", False
    decoder = json.JSONDecoder()
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
        m = re.search(r'"answer"\s*:\s*"(.*?)"', cleaned, re.DOTALL)
        if m:
            answer    = m.group(1)
            parsed_ok = True

    matches_gold = _answer_correct(answer, gold, task_type) if parsed_ok else False
    return {"parsed_ok": parsed_ok, "answer": answer,
            "answer_matches_gold": matches_gold, "reasoning": reasoning}


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


def check_server():
    base = COMPLETION_URL.replace("/completion", "")
    with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
        body = json.loads(r.read())
    assert body.get("status") == "ok", f"health={body}"

    with urllib.request.urlopen(f"{base}/props", timeout=5) as r:
        props = json.loads(r.read())
    total_slots = props.get("total_slots", "?")
    alias       = props.get("model_alias", "?")
    print(f"Server OK — model={alias}  total_slots={total_slots}")

    with urllib.request.urlopen(f"{base}/slots", timeout=5) as r:
        slots = json.loads(r.read())
    for s in slots:
        print(f"  slot id={s['id']}  n_ctx={s.get('n_ctx')}  state={s.get('state')}")
    return int(total_slots) if total_slots != "?" else 1


def call_row(row, n_predict=N_PREDICT):
    prompt  = format_prompt(str(row["answer"]), row["prompt"])
    payload = json.dumps({
        "prompt":      prompt,
        "n_predict":   n_predict,
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
        elapsed = round(time.time() - t0, 3)
        print(f"  [{row['id']}] HTTP {e.code}: {body[:200]}", flush=True)
        return None, 0, elapsed, False
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        print(f"  [{row['id']}] Error: {e}", flush=True)
        return None, 0, elapsed, False

    elapsed  = round(time.time() - t0, 3)
    raw      = resp.get("content", "")
    tok_out  = resp.get("tokens_predicted", 0)
    truncated = resp.get("stopped_limit", False)
    return raw, tok_out, elapsed, truncated


def process_row(row, n_predict, seq_num, total):
    t_submit = time.time()
    raw, tok_out, elapsed, truncated = call_row(row, n_predict=n_predict)
    t_done = time.time()

    if raw is None:
        result = {
            "id": row["id"], "task_type": row["task_type"],
            "gold_answer": str(row["answer"]),
            "parse_success": False, "answer_correct": False,
            "answer": "ERROR", "reasoning": "CALL_FAILED",
            "tokens_out": tok_out, "latency_s": elapsed,
            "truncated": truncated, "seq_num": seq_num,
        }
    else:
        parsed = parse_response(raw, str(row["answer"]), row["task_type"])
        result = {
            "id": row["id"], "task_type": row["task_type"],
            "gold_answer": str(row["answer"]),
            "parse_success": parsed["parsed_ok"],
            "answer_correct": parsed["answer_matches_gold"],
            "answer": parsed["answer"], "reasoning": parsed["reasoning"],
            "tokens_out": tok_out, "latency_s": elapsed,
            "truncated": truncated, "seq_num": seq_num,
        }

    status = "GOOD" if (result["parse_success"] and result["answer_correct"]) else "FAIL"
    print(
        f"  [{seq_num:2}/{total}] [{row['id']}] task={row['task_type']:<20} "
        f"parse={result['parse_success']} copy={result['answer_correct']} "
        f"tok={result['tokens_out']} lat={elapsed}s -> {status}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Concurrency benchmark for /completion endpoint")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent workers (default: 4)")
    parser.add_argument("--rows", type=int, default=20,
                        help="Number of rows to benchmark (default: 20, from head of CSV)")
    parser.add_argument("--n-predict", type=int, default=N_PREDICT,
                        help=f"Tokens to generate per row (default: {N_PREDICT})")
    parser.add_argument("--offset", type=int, default=0,
                        help="Row offset into CSV (default: 0)")
    args = parser.parse_args()

    # ---- Server check -------------------------------------------------------
    total_slots = check_server()
    if args.workers > total_slots:
        print(f"WARNING: requested workers={args.workers} > server slots={total_slots}. "
              f"Excess requests will queue; this measures queuing overhead too.")

    # ---- Load rows ----------------------------------------------------------
    df = pd.read_csv(INPUT_CSV)
    subset = df.iloc[args.offset : args.offset + args.rows].reset_index(drop=True)
    print(f"\nBenchmark: {len(subset)} rows  workers={args.workers}  "
          f"n_predict={args.n_predict}  temperature={TEMPERATURE}")
    print(f"Task mix: {dict(subset['task_type'].value_counts())}\n")

    # ---- VRAM baseline ------------------------------------------------------
    vram_start = vram_snapshot()
    if vram_start:
        print(f"VRAM before: {vram_start['used_mb']} MB / {vram_start['total_mb']} MB\n")

    # ---- Clear output -------------------------------------------------------
    if os.path.exists(BENCH_OUTPUT_JSONL):
        os.remove(BENCH_OUTPUT_JSONL)

    # ---- Run ----------------------------------------------------------------
    results   = [None] * len(subset)
    seq_counter = [0]  # mutable for closure
    vram_peak = vram_start["used_mb"] if vram_start else 0

    t_wall_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(process_row, row, args.n_predict, i + 1, len(subset)): i
            for i, (_, row) in enumerate(subset.iterrows())
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as e:
                row = subset.iloc[idx]
                print(f"  [{row['id']}] Unhandled error: {e}", flush=True)
                result = {
                    "id": row["id"], "task_type": row["task_type"],
                    "gold_answer": str(row["answer"]),
                    "parse_success": False, "answer_correct": False,
                    "answer": "ERROR", "reasoning": str(e),
                    "tokens_out": 0, "latency_s": 0,
                    "truncated": False, "seq_num": idx + 1,
                }
            results[idx] = result

            snap = vram_snapshot()
            if snap and snap["used_mb"] > vram_peak:
                vram_peak = snap["used_mb"]

            with open(BENCH_OUTPUT_JSONL, "a") as f:
                f.write(json.dumps(result) + "\n")

    t_wall_end  = time.time()
    wall_clock  = round(t_wall_end - t_wall_start, 2)
    vram_end    = vram_snapshot()

    # ---- Compute metrics ----------------------------------------------------
    n           = len(results)
    good        = [r for r in results if r["parse_success"] and r["answer_correct"]]
    parse_ok    = [r for r in results if r["parse_success"]]
    latencies   = [r["latency_s"] for r in results]
    toks        = [r["tokens_out"] for r in results]
    truncated   = [r for r in results if r.get("truncated")]

    avg_lat     = sum(latencies) / n if n else 0
    p50_lat     = sorted(latencies)[n // 2] if latencies else 0
    p95_lat     = sorted(latencies)[int(n * 0.95)] if latencies else 0
    max_lat     = max(latencies) if latencies else 0
    avg_tok     = sum(toks) / n if n else 0
    throughput  = round(n / wall_clock, 3) if wall_clock else 0

    by_task = {}
    for r in results:
        tt = r["task_type"]
        if tt not in by_task:
            by_task[tt] = {"n": 0, "good": 0}
        by_task[tt]["n"] += 1
        if r["parse_success"] and r["answer_correct"]:
            by_task[tt]["good"] += 1

    # ---- Print report -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"CONCURRENCY BENCHMARK REPORT")
    print(f"  Workers:        {args.workers}")
    print(f"  Rows:           {n}")
    print(f"  n_predict:      {args.n_predict}")
    print(f"  Server slots:   {total_slots}")
    print()
    print(f"  --- Quality ---")
    print(f"  Parse OK:       {len(parse_ok)}/{n}  ({len(parse_ok)/n:.0%})")
    print(f"  Answer copy OK: {len(good)}/{n}  ({len(good)/n:.0%})")
    print(f"  Truncated rows: {len(truncated)}")
    print()
    print(f"  --- Latency (per-row, includes queue wait) ---")
    print(f"  Avg:            {avg_lat:.3f}s")
    print(f"  p50:            {p50_lat:.3f}s")
    print(f"  p95:            {p95_lat:.3f}s")
    print(f"  Max:            {max_lat:.3f}s")
    print()
    print(f"  --- Throughput ---")
    print(f"  Wall clock:     {wall_clock}s  ({wall_clock/60:.1f} min)")
    print(f"  Throughput:     {throughput} rows/sec")
    print(f"  Proj 9,500 rows: ~{round(9500 / throughput / 3600, 1)}h  (workers={args.workers})")
    print()
    print(f"  --- Tokens ---")
    print(f"  Avg tok_out:    {avg_tok:.1f}")
    print()
    print(f"  --- VRAM ---")
    if vram_start:
        print(f"  Before:         {vram_start['used_mb']} MB / {vram_start['total_mb']} MB")
    if vram_peak:
        print(f"  Peak (sampled): {vram_peak} MB")
    if vram_end:
        print(f"  After:          {vram_end['used_mb']} MB / {vram_end['total_mb']} MB")
    print()
    print(f"  --- Per-task ---")
    for tt, v in sorted(by_task.items()):
        rate = v["good"] / v["n"] if v["n"] else 0
        print(f"  {tt:<22}  good={v['good']}/{v['n']}  ({rate:.0%})")

    failures = [r for r in results if not r["parse_success"] or not r["answer_correct"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            ft = "parse_fail" if not r["parse_success"] else "copy_fail"
            print(f"    [{r['id']}] {r['task_type']:<20} {ft}  "
                  f"gold={r['gold_answer']!r}  pred={r['answer']!r}")

    speedup_vs_w1 = round(1.25 / avg_lat, 2) if avg_lat else "?"  # vs known w=1 baseline
    proj_w1_h = round(9500 * 1.25 / 3600, 1)
    proj_wN_h = round(9500 / throughput / 3600, 1) if throughput else "?"

    print(f"\n  --- vs workers=1 baseline (1.25s/row) ---")
    print(f"  Per-row speedup:        {speedup_vs_w1}x  "
          f"({'faster' if avg_lat < 1.25 else 'same/slower'})")
    print(f"  Proj 9,500-row @ w=1:  ~{proj_w1_h}h")
    print(f"  Proj 9,500-row @ w={args.workers}: ~{proj_wN_h}h")

    # ---- Recommendation -----------------------------------------------------
    stable = (len(parse_ok) == n and len(failures) == 0) or (len(good) / n >= 0.95)
    no_errors = all(r["answer"] != "ERROR" for r in results)
    if stable and no_errors and args.workers > 1:
        rec = f"RECOMMEND workers={args.workers} for full generation"
    elif not stable:
        rec = f"NOT RECOMMENDED — quality degraded at workers={args.workers}"
    else:
        rec = "workers=1 already stable; no concurrency gain needed"
    print(f"\n  Recommendation: {rec}")
    print("=" * 60)

    # ---- Save report --------------------------------------------------------
    report = {
        "workers": args.workers, "rows": n, "n_predict": args.n_predict,
        "server_slots": total_slots,
        "parse_rate": round(len(parse_ok) / n, 4) if n else 0,
        "copy_rate": round(len(good) / n, 4) if n else 0,
        "avg_latency_s": round(avg_lat, 4),
        "p50_latency_s": p50_lat, "p95_latency_s": p95_lat, "max_latency_s": max_lat,
        "wall_clock_s": wall_clock,
        "throughput_rows_per_sec": throughput,
        "projected_9500_h": proj_wN_h,
        "avg_tok_out": round(avg_tok, 1),
        "truncated_rows": len(truncated),
        "vram_start_mb": vram_start["used_mb"] if vram_start else None,
        "vram_peak_mb": vram_peak if vram_peak else None,
        "vram_end_mb": vram_end["used_mb"] if vram_end else None,
        "vram_total_mb": vram_end["total_mb"] if vram_end else None,
        "per_task": by_task,
        "failures": [
            {"id": r["id"], "task_type": r["task_type"],
             "fail_type": "parse_fail" if not r["parse_success"] else "copy_fail",
             "gold": r["gold_answer"], "pred": r["answer"]}
            for r in failures
        ],
        "recommendation": rec,
    }
    with open(BENCH_OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSONL:  {BENCH_OUTPUT_JSONL}")
    print(f"Report: {BENCH_OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
