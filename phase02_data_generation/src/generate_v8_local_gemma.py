#!/usr/bin/env python3
"""
Full V8 post-hoc rationale generation — local Gemma 4 12B.

Provider  : local llama.cpp /completion (NOT /v1/chat/completions)
Model     : unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL
Workers   : 4  (server running with --parallel 4)
n_predict : 512
Mode      : post_hoc_rationale_v8  (gold answer given; answer_correct = copy fidelity)

Acceptance:
  Model-generated : parse_success=True AND answer_correct=True
  Symbol repair   : symbol_transform, parse_success=True, answer_correct=False
                    → deterministic row with gold answer + fixed reasoning string
  Failure         : everything else (written to failures log, excluded from dataset)

Outputs:
  phase02_data_generation/data/v8/train_reasoning_v8_local_gemma.jsonl
  phase02_data_generation/data/v8/val_reasoning_v8_local_gemma.jsonl
  phase02_data_generation/data/v8/local_gemma_full_report.json
  phase02_data_generation/data/v8/local_gemma_symbol_repairs.jsonl
  phase02_data_generation/data/v8/local_gemma_full_failures.jsonl

Run from project root:
  python phase02_data_generation/src/generate_v8_local_gemma.py
  python phase02_data_generation/src/generate_v8_local_gemma.py --dry-run
  python phase02_data_generation/src/generate_v8_local_gemma.py --workers 4 --val-frac 0.05
"""

import argparse
import json
import os
import re
import subprocess
import threading
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

DATA_DIR = "phase02_data_generation/data/v8"
STAGING_JSONL   = f"{DATA_DIR}/local_gemma_staging.jsonl"        # resume checkpoint
TRAIN_JSONL     = f"{DATA_DIR}/train_reasoning_v8_local_gemma.jsonl"
VAL_JSONL       = f"{DATA_DIR}/val_reasoning_v8_local_gemma.jsonl"
REPORT_JSON     = f"{DATA_DIR}/local_gemma_full_report.json"
REPAIRS_JSONL   = f"{DATA_DIR}/local_gemma_symbol_repairs.jsonl"
FAILURES_JSONL  = f"{DATA_DIR}/local_gemma_full_failures.jsonl"

N_PREDICT    = 512
TEMPERATURE  = 0.0
WORKERS      = 4
VAL_FRAC     = 0.05   # 5 % → ~475 val rows
RANDOM_SEED  = 42
PROGRESS_EVERY = 250  # print progress every N completions

STOP_SEQS = ["<turn|>", "<|turn>", "<eos>", "</s>", "}\n\n", "}\n\nWait", "} \n\n"]

MODEL_LABEL  = "local_gemma4_12b_completion_posthoc_v8"
REPAIR_LABEL = "deterministic_symbol_copy_repair_v8"
REPAIR_REASONING = (
    "The correct symbol sequence is provided for this post-hoc training trace, "
    "so I copy it exactly."
)

SYSTEM_PROMPT = """You output exactly one JSON object.
The answer is already solved.
Copy CORRECT_ANSWER exactly.
Do not solve.
Do not include hidden reasoning.
Do not explain outside JSON.
Schema: {"answer":"...", "reasoning":"..."}
Reasoning must be one short sentence."""

# ---------------------------------------------------------------------------
# Prompt / response helpers
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
# Server helpers
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
        total_slots = props.get("total_slots", "?")
        alias       = props.get("model_alias", "?")
        with urllib.request.urlopen(f"{base}/slots", timeout=5) as r:
            slots = json.loads(r.read())
        n_ctx_per_slot = slots[0].get("n_ctx", "?") if slots else "?"
        print(f"Server OK — model={alias}  slots={total_slots}  n_ctx/slot={n_ctx_per_slot}")
        return int(total_slots) if total_slots != "?" else 1
    except Exception as e:
        print(f"Warning: /props unavailable: {e}")
        return 1


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
# HTTP call (with retry on connection error only)
# ---------------------------------------------------------------------------

def call_completion(row_id, prompt, n_predict, max_http_retries=3):
    """POST to /completion. Retries on connection errors; not on content failures."""
    payload = json.dumps({
        "prompt":       prompt,
        "n_predict":    n_predict,
        "temperature":  TEMPERATURE,
        "stop":         STOP_SEQS,
        "cache_prompt": False,
    }).encode()

    for attempt in range(max_http_retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                COMPLETION_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.load(r)
            elapsed  = round(time.time() - t0, 3)
            raw      = resp.get("content", "")
            tok_out  = resp.get("tokens_predicted", 0)
            truncated = resp.get("stopped_limit", False)
            return raw, tok_out, elapsed, truncated
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            elapsed = round(time.time() - t0, 3)
            print(f"  [{row_id}] HTTP {e.code}: {body[:200]}", flush=True)
            return None, 0, elapsed, False
        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            if attempt < max_http_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  [{row_id}] Connection error (attempt {attempt+1}): {e} — retry in {wait}s",
                      flush=True)
                time.sleep(wait)
            else:
                print(f"  [{row_id}] Failed after {max_http_retries} attempts: {e}", flush=True)
                return None, 0, elapsed, False

    return None, 0, 0.0, False

# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def process_row(row, n_predict):
    """
    Generate, parse, classify.
    Returns (accepted_record, repair_record, failure_record).
    Exactly one of the three is non-None.
    """
    row_id    = str(row["id"])
    task_type = str(row["task_type"])
    gold      = str(row["answer"])
    prompt    = format_prompt(gold, row["prompt"])

    raw, tok_out, elapsed, truncated = call_completion(row_id, prompt, n_predict)

    base = {
        "id":         row_id,
        "task_type":  task_type,
        "prompt":     row["prompt"],
        "gold_answer": gold,
        "tokens_out": tok_out,
        "gen_time_s": elapsed,
        "truncated":  truncated,
    }

    # ---- HTTP failure -------------------------------------------------------
    if raw is None:
        failure = {**base,
                   "source":         "http_error",
                   "parse_success":  False,
                   "answer_correct": False,
                   "answer":         "HTTP_ERROR",
                   "reasoning":      "",
                   "fail_reason":    "http_error"}
        return None, None, failure

    parsed = parse_response(raw, gold, task_type)

    # ---- Accepted (model-generated) -----------------------------------------
    if parsed["parsed_ok"] and parsed["answer_matches_gold"]:
        record = {**base,
                  "source":         MODEL_LABEL,
                  "parse_success":  True,
                  "answer_correct": True,
                  "answer":         parsed["answer"],
                  "reasoning":      parsed["reasoning"]}
        return record, None, None

    # ---- Symbol repair ------------------------------------------------------
    if task_type == "symbol_transform" and parsed["parsed_ok"]:
        repair = {**base,
                  "source":         REPAIR_LABEL,
                  "parse_success":  True,
                  "answer_correct": True,   # gold is authoritative
                  "answer":         gold,
                  "reasoning":      REPAIR_REASONING,
                  "orig_pred":      parsed["answer"]}
        return repair, repair, None   # accepted AND logged as repair

    # ---- Failure (non-symbol, or symbol parse failure) ----------------------
    fail_reason = "parse_fail" if not parsed["parsed_ok"] else "copy_fail"
    failure = {**base,
               "source":         MODEL_LABEL,
               "parse_success":  parsed["parsed_ok"],
               "answer_correct": False,
               "answer":         parsed["answer"],
               "reasoning":      parsed["reasoning"],
               "fail_reason":    fail_reason}
    return None, None, failure

# ---------------------------------------------------------------------------
# Resume: load already-processed IDs from staging + failures
# ---------------------------------------------------------------------------

def load_done_ids():
    done = set()
    for path in (STAGING_JSONL, FAILURES_JSONL):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                            done.add(str(r["id"]))
                        except Exception:
                            pass
    return done

# ---------------------------------------------------------------------------
# Progress printer (thread-safe)
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, total):
        self._lock  = threading.Lock()
        self.total  = total
        self.done   = 0
        self.good   = 0     # model-accepted
        self.repair = 0     # symbol repairs
        self.fail   = 0     # failures
        self.tok_sum = 0
        self.lat_sum = 0.0
        self.t_start = time.time()

    def update(self, accepted, is_repair, is_failure, tok_out, lat):
        with self._lock:
            self.done  += 1
            self.tok_sum += tok_out
            self.lat_sum += lat
            if accepted and not is_repair:
                self.good += 1
            elif is_repair:
                self.repair += 1
            elif is_failure:
                self.fail += 1
            should_print = (self.done % PROGRESS_EVERY == 0) or (self.done == self.total)
        if should_print:
            self._print()

    def _print(self):
        elapsed  = time.time() - self.t_start
        rate     = self.done / elapsed if elapsed > 0 else 0
        eta_s    = (self.total - self.done) / rate if rate > 0 else 0
        avg_tok  = self.tok_sum / self.done if self.done else 0
        avg_lat  = self.lat_sum / self.done if self.done else 0
        snap     = vram_snapshot()
        vram_str = f"  VRAM={snap['used_mb']}MB" if snap else ""
        pct      = self.done / self.total * 100
        print(
            f"\n[PROGRESS {self.done}/{self.total} ({pct:.1f}%)]"
            f"  model={self.good}  repair={self.repair}  fail={self.fail}"
            f"  rate={rate:.2f}r/s  eta={eta_s/60:.1f}min"
            f"  avg_tok={avg_tok:.0f}  avg_lat={avg_lat:.2f}s{vram_str}",
            flush=True,
        )

# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def make_train_val_split(staging_path, train_path, val_path, val_frac, seed):
    rows = []
    with open(staging_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    # Stratified by task_type
    val_rows = (
        df.groupby("task_type", group_keys=False)
        .apply(lambda g: g.sample(frac=val_frac, random_state=seed))
    )
    val_ids  = set(val_rows["id"].astype(str))
    train_rows = df[~df["id"].astype(str).isin(val_ids)]

    with open(train_path, "w") as f:
        for _, r in train_rows.iterrows():
            f.write(json.dumps(r.to_dict()) + "\n")
    with open(val_path, "w") as f:
        for _, r in val_rows.iterrows():
            f.write(json.dumps(r.to_dict()) + "\n")

    return len(train_rows), len(val_rows)

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def write_report(tracker, wall_clock, n_total, n_train, n_val,
                 vram_peak, vram_start, n_predict, workers):
    snap_end = vram_snapshot()

    # Per-task breakdown from failures and repairs
    fail_path = FAILURES_JSONL
    repair_path = REPAIRS_JSONL
    by_task_fail: dict = {}
    by_task_repair: dict = {}

    if os.path.exists(fail_path):
        with open(fail_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                tt = r.get("task_type", "?")
                by_task_fail[tt] = by_task_fail.get(tt, 0) + 1

    if os.path.exists(repair_path):
        with open(repair_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                tt = r.get("task_type", "?")
                by_task_repair[tt] = by_task_repair.get(tt, 0) + 1

    total_accepted  = tracker.good + tracker.repair
    parse_ok_count  = total_accepted + sum(
        1 for v in by_task_fail.values() for _ in range(v)
        # approximate: failures include both parse_fail and copy_fail
    )

    avg_tok = tracker.tok_sum / tracker.done if tracker.done else 0
    avg_lat = tracker.lat_sum / tracker.done if tracker.done else 0
    rate    = tracker.done / wall_clock if wall_clock else 0

    report = {
        "mode":              "post_hoc_rationale_v8",
        "model_label":       MODEL_LABEL,
        "endpoint":          COMPLETION_URL,
        "n_predict":         n_predict,
        "temperature":       TEMPERATURE,
        "workers":           workers,
        "timestamp_utc":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_rows_in_csv": n_total,
        "rows_processed":    tracker.done,
        "model_accepted":    tracker.good,
        "symbol_repairs":    tracker.repair,
        "failures":          tracker.fail,
        "total_usable":      total_accepted,
        "train_rows":        n_train,
        "val_rows":          n_val,
        "val_frac":          VAL_FRAC,
        "avg_tok_out":       round(avg_tok, 1),
        "avg_latency_s":     round(avg_lat, 3),
        "throughput_rows_per_sec": round(rate, 3),
        "wall_clock_s":      round(wall_clock, 1),
        "wall_clock_min":    round(wall_clock / 60, 1),
        "failures_by_task":  by_task_fail,
        "repairs_by_task":   by_task_repair,
        "vram_start_mb":     vram_start["used_mb"] if vram_start else None,
        "vram_peak_mb":      vram_peak,
        "vram_end_mb":       snap_end["used_mb"] if snap_end else None,
        "vram_total_mb":     snap_end["total_mb"] if snap_end else None,
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print(f"\n{'='*64}")
    print("V8 GENERATION COMPLETE")
    print(f"  Input rows:          {n_total}")
    print(f"  Rows processed:      {tracker.done}")
    print(f"  Model accepted:      {tracker.good}")
    print(f"  Symbol repairs:      {tracker.repair}")
    print(f"  Failures:            {tracker.fail}")
    print(f"  Total usable:        {total_accepted}")
    print(f"  Train / Val:         {n_train} / {n_val}  ({VAL_FRAC:.0%} val)")
    print(f"  Avg tok_out:         {avg_tok:.1f}")
    print(f"  Avg latency/row:     {avg_lat:.3f}s")
    print(f"  Throughput:          {rate:.3f} rows/sec")
    print(f"  Wall clock:          {wall_clock/60:.1f} min  ({wall_clock/3600:.2f}h)")
    if vram_peak:
        print(f"  VRAM peak:           {vram_peak} MB")
    if by_task_fail:
        print(f"\n  Failures by task:")
        for tt, n in sorted(by_task_fail.items()):
            print(f"    {tt:<22}  {n}")
    if by_task_repair:
        print(f"\n  Repairs by task:")
        for tt, n in sorted(by_task_repair.items()):
            print(f"    {tt:<22}  {n}")
    print(f"\n  Output files:")
    print(f"    {TRAIN_JSONL}")
    print(f"    {VAL_JSONL}")
    print(f"    {REPORT_JSON}")
    print(f"    {REPAIRS_JSONL}")
    print(f"    {FAILURES_JSONL}")
    print("=" * 64)

    return report

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="V8 full post-hoc rationale generation — local Gemma 4 12B"
    )
    parser.add_argument("--workers",   type=int,   default=WORKERS,
                        help=f"Concurrent workers (default: {WORKERS})")
    parser.add_argument("--n-predict", type=int,   default=N_PREDICT,
                        help=f"Tokens per row (default: {N_PREDICT})")
    parser.add_argument("--val-frac",  type=float, default=VAL_FRAC,
                        help=f"Validation fraction (default: {VAL_FRAC})")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print first prompt and exit; no API calls.")
    parser.add_argument("--max-rows",  type=int, default=None,
                        help="Cap rows (for testing, not for production).")
    args = parser.parse_args()

    # ---- Server check -------------------------------------------------------
    if not args.dry_run:
        total_slots = check_server()
        if args.workers > total_slots:
            print(f"WARNING: workers={args.workers} > server slots={total_slots}. "
                  f"Requests will queue.")

    # ---- Load CSV -----------------------------------------------------------
    df = pd.read_csv(INPUT_CSV)
    if args.max_rows:
        df = df.head(args.max_rows)
    n_total = len(df)

    # ---- Resume: skip already-done IDs -------------------------------------
    done_ids = load_done_ids()
    if done_ids:
        n_skip = len(df[df["id"].astype(str).isin(done_ids)])
        print(f"Resume: skipping {n_skip} already-processed rows "
              f"({len(done_ids)} IDs found in staging/failures).")
        df = df[~df["id"].astype(str).isin(done_ids)].reset_index(drop=True)

    pending = len(df)
    print(f"\nRows to process: {pending} / {n_total}  "
          f"(workers={args.workers}, n_predict={args.n_predict})\n")

    # ---- Dry run ------------------------------------------------------------
    if args.dry_run:
        row = df.iloc[0]
        prompt = format_prompt(str(row["answer"]), row["prompt"])
        print(f"--- Prompt: id={row['id']} task={row['task_type']} ---")
        print(prompt)
        print(f"\n(n_predict={args.n_predict}, stop={STOP_SEQS})")
        return

    if pending == 0:
        print("All rows already processed. Running split + report from staging.")
        n_train, n_val = make_train_val_split(
            STAGING_JSONL, TRAIN_JSONL, VAL_JSONL, args.val_frac, RANDOM_SEED)
        print(f"Train: {n_train}  Val: {n_val}")
        return

    # ---- Init outputs -------------------------------------------------------
    for path in (STAGING_JSONL, REPAIRS_JSONL, FAILURES_JSONL):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    staging_lock  = threading.Lock()
    repair_lock   = threading.Lock()
    failure_lock  = threading.Lock()

    vram_start = vram_snapshot()
    vram_peak  = vram_start["used_mb"] if vram_start else 0
    vram_lock  = threading.Lock()

    tracker   = ProgressTracker(total=n_total)
    tracker.done  = len(done_ids)   # account for already-done rows in counters
    # Re-read staging for repair/good counts to get accurate resumption state
    if done_ids and os.path.exists(STAGING_JSONL):
        with open(STAGING_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("source") == REPAIR_LABEL:
                    tracker.repair += 1
                else:
                    tracker.good += 1
    if done_ids and os.path.exists(FAILURES_JSONL):
        with open(FAILURES_JSONL) as f:
            for line in f:
                if line.strip():
                    tracker.fail += 1

    t_wall_start = time.time()
    print(f"Starting generation  "
          f"[{pending} rows  workers={args.workers}  n_predict={args.n_predict}]\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_row = {
            executor.submit(process_row, row, args.n_predict): row
            for _, row in df.iterrows()
        }

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            row_id    = str(row["id"])
            task_type = str(row["task_type"])

            try:
                accepted, repair_rec, failure = future.result()
            except Exception as e:
                print(f"  [{row_id}] Unhandled exception: {e}", flush=True)
                failure  = {
                    "id": row_id, "task_type": task_type,
                    "prompt": row["prompt"], "gold_answer": str(row["answer"]),
                    "source": "exception", "parse_success": False,
                    "answer_correct": False, "answer": "EXCEPTION",
                    "reasoning": str(e), "tokens_out": 0, "gen_time_s": 0,
                    "truncated": False, "fail_reason": "exception",
                }
                accepted = repair_rec = None

            is_repair   = repair_rec is not None
            is_failure  = failure is not None
            tok_out     = (accepted or failure or {}).get("tokens_out", 0)
            lat         = (accepted or failure or {}).get("gen_time_s", 0.0)

            # ---- Write outputs -----------------------------------------------
            if accepted is not None:
                with staging_lock:
                    with open(STAGING_JSONL, "a") as f:
                        f.write(json.dumps(accepted) + "\n")
            if repair_rec is not None:
                with repair_lock:
                    with open(REPAIRS_JSONL, "a") as f:
                        f.write(json.dumps(repair_rec) + "\n")
            if failure is not None:
                with failure_lock:
                    with open(FAILURES_JSONL, "a") as f:
                        f.write(json.dumps(failure) + "\n")

            # ---- VRAM peak tracking ------------------------------------------
            snap = vram_snapshot()
            if snap:
                with vram_lock:
                    if snap["used_mb"] > vram_peak:
                        vram_peak = snap["used_mb"]

            # ---- Row-level console log (terse) -------------------------------
            if accepted is not None:
                status = "REPAIR" if is_repair else "GOOD"
            else:
                status = "FAIL"
            print(
                f"  [{row_id}] {task_type:<20} tok={tok_out} "
                f"lat={lat}s -> {status}",
                flush=True,
            )

            tracker.update(
                accepted   = accepted is not None,
                is_repair  = is_repair,
                is_failure = is_failure,
                tok_out    = tok_out,
                lat        = lat,
            )

    wall_clock = time.time() - t_wall_start

    # ---- Train / val split --------------------------------------------------
    print("\nBuilding train/val split...", flush=True)
    n_train, n_val = make_train_val_split(
        STAGING_JSONL, TRAIN_JSONL, VAL_JSONL, args.val_frac, RANDOM_SEED)

    # ---- Final report -------------------------------------------------------
    write_report(tracker, wall_clock, n_total, n_train, n_val,
                 vram_peak, vram_start, args.n_predict, args.workers)

    print("\nGeneration complete. Review report before proceeding.")
    print("Do NOT train LoRA until user approves the dataset.")


if __name__ == "__main__":
    main()
