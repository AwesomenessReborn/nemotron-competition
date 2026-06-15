#!/usr/bin/env python3
"""
V8.1 symbol repair regeneration.

Replaces 355+30 training rows with source=deterministic_symbol_copy_repair_v8
(from V8 clean train/val files) with fresh model-generated reasoning that
explains the actual character-level symbol transformation rule.

The original V8 repair rows all share a single generic reasoning string that
leaked into model inference. This script regenerates them with task-specific
reasoning and validates that no banned generic-copy phrases are present.

Inputs:
  phase02_data_generation/data/v8/train_reasoning_v8_local_gemma_clean.jsonl
  phase02_data_generation/data/v8/val_reasoning_v8_local_gemma_clean.jsonl

Outputs:
  phase02_data_generation/data/v8/train_reasoning_v8_1_local_gemma_clean.jsonl
  phase02_data_generation/data/v8/val_reasoning_v8_1_local_gemma_clean.jsonl
  phase02_data_generation/data/v8/v8_1_symbol_regen_failures.jsonl
  phase02_data_generation/data/v8/v8_1_local_gemma_clean_dataset_audit.json
  phase02_data_generation/data/v8/v8_1_local_gemma_clean_dataset_audit.md

Run from project root:
  python phase02_data_generation/src/regen_v8_1_symbol_repairs.py
  python phase02_data_generation/src/regen_v8_1_symbol_repairs.py --dry-run
  python phase02_data_generation/src/regen_v8_1_symbol_repairs.py --workers 8
"""

import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMPLETION_URL = os.environ.get("LOCAL_COMPLETION_URL", "http://127.0.0.1:8080/completion")
DATA_DIR = "phase02_data_generation/data/v8"

V8_TRAIN  = f"{DATA_DIR}/train_reasoning_v8_local_gemma_clean.jsonl"
V8_VAL    = f"{DATA_DIR}/val_reasoning_v8_local_gemma_clean.jsonl"

V81_TRAIN    = f"{DATA_DIR}/train_reasoning_v8_1_local_gemma_clean.jsonl"
V81_VAL      = f"{DATA_DIR}/val_reasoning_v8_1_local_gemma_clean.jsonl"
FAILURES_OUT = f"{DATA_DIR}/v8_1_symbol_regen_failures.jsonl"
AUDIT_JSON   = f"{DATA_DIR}/v8_1_local_gemma_clean_dataset_audit.json"
AUDIT_MD     = f"{DATA_DIR}/v8_1_local_gemma_clean_dataset_audit.md"

REPAIR_SOURCE   = "deterministic_symbol_copy_repair_v8"
REGEN_SOURCE    = "local_gemma4_12b_symbol_regen_v8_1"

N_PREDICT   = 512
TEMPERATURE = 0.0
WORKERS     = 8

STOP_SEQS = ["<turn|>", "<|turn>", "<eos>", "</s>", "}\n\n", "}\n\nWait", "} \n\n"]

# Phrases that must NOT appear in accepted reasoning.
# These are specific to the copy-it-exactly repair leak and should not appear
# in legitimate reasoning. "provided correct answer" and "given answer" are
# intentionally excluded — they appear naturally in gravity/unit_conversion
# reasoning ("the provided correct answer is 55.28") and are not a leak signal.
BANNED_PHRASES = [
    "copy it exactly",
    "the correct symbol sequence is provided",
    "post-hoc training trace",
    "i copy it",
    "so i copy",
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are analyzing a character-level symbol transformation puzzle.
The correct answer is already given to you as CORRECT_ANSWER.

Your task:
1. Study the example input→output pairs in the puzzle
2. Identify the specific character-by-character mapping rule
3. Verify that applying this mapping to the query input produces CORRECT_ANSWER
4. Write a short, specific reasoning that names the actual symbol mappings observed

Output exactly one JSON object on a single line:
{"reasoning": "...", "answer": "..."}

Rules:
- "answer" must be copied exactly from CORRECT_ANSWER
- "reasoning" must describe the actual symbol mapping (name the specific characters)
- Do NOT write "copy it exactly", "provided correct answer", "given answer", \
or any phrase suggesting you are merely copying
- Reasoning must be specific to the symbols in this puzzle, not generic\
"""


def format_prompt(gold_answer: str, puzzle_prompt: str) -> str:
    user_content = (
        f"CORRECT_ANSWER: {gold_answer}\n\n"
        f"PUZZLE:\n{puzzle_prompt}\n\n"
        "Identify the character-level mapping rule from the examples and explain "
        "why CORRECT_ANSWER is the correct output for the query."
    )
    return (
        f"<bos><|turn>system\n{SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n{user_content}<turn|>\n"
        f"<|turn>model\n"
    )

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    return re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL).strip()


def parse_response(raw: str, gold: str) -> dict:
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

    answer_correct = parsed_ok and (answer == gold)
    return {
        "parsed_ok":      parsed_ok,
        "answer":         answer,
        "answer_correct": answer_correct,
        "reasoning":      reasoning,
        "cleaned":        cleaned,
    }


def has_banned_phrase(text: str) -> str | None:
    lower = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lower:
            return phrase
    return None

# ---------------------------------------------------------------------------
# Server
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
        slots = props.get("total_slots", "?")
        alias = props.get("model_alias", "?")
        print(f"Server OK — model={alias}  slots={slots}")
        return int(slots) if slots != "?" else 1
    except Exception as e:
        print(f"Warning: /props unavailable: {e}")
        return 1

# ---------------------------------------------------------------------------
# HTTP call
# ---------------------------------------------------------------------------

def call_completion(row_id: str, prompt: str, max_retries: int = 3):
    payload = json.dumps({
        "prompt":       prompt,
        "n_predict":    N_PREDICT,
        "temperature":  TEMPERATURE,
        "stop":         STOP_SEQS,
        "cache_prompt": False,
    }).encode()

    for attempt in range(max_retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                COMPLETION_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.load(r)
            elapsed   = round(time.time() - t0, 3)
            raw       = resp.get("content", "")
            tok_out   = resp.get("tokens_predicted", 0)
            truncated = resp.get("stopped_limit", False)
            return raw, tok_out, elapsed, truncated
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            elapsed = round(time.time() - t0, 3)
            print(f"  [{row_id}] HTTP {e.code}: {body[:200]}", flush=True)
            return None, 0, elapsed, False
        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  [{row_id}] Connection error (attempt {attempt+1}): {e} — retry in {wait}s",
                      flush=True)
                time.sleep(wait)
            else:
                print(f"  [{row_id}] Failed after {max_retries} attempts: {e}", flush=True)
                return None, 0, elapsed, False

    return None, 0, 0.0, False

# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def process_row(row: dict) -> tuple[dict | None, dict | None]:
    """
    Returns (accepted_record, failure_record). Exactly one is non-None.
    """
    row_id = str(row["id"])
    gold   = str(row["gold_answer"])

    prompt = format_prompt(gold, row["prompt"])
    raw, tok_out, elapsed, truncated = call_completion(row_id, prompt)

    base = {
        "id":         row_id,
        "task_type":  "symbol_transform",
        "prompt":     row["prompt"],
        "gold_answer": gold,
        "tokens_out": tok_out,
        "gen_time_s": elapsed,
        "truncated":  truncated,
    }

    if raw is None:
        failure = {**base,
                   "source":        "http_error",
                   "parse_success": False,
                   "answer_correct": False,
                   "answer":        "HTTP_ERROR",
                   "reasoning":     "",
                   "fail_reason":   "http_error"}
        return None, failure

    parsed = parse_response(raw, gold)

    if not parsed["parsed_ok"]:
        failure = {**base,
                   "source":        REGEN_SOURCE,
                   "parse_success": False,
                   "answer_correct": False,
                   "answer":        parsed["answer"],
                   "reasoning":     parsed["reasoning"],
                   "fail_reason":   "parse_fail",
                   "raw_snippet":   raw[:300]}
        return None, failure

    if not parsed["answer_correct"]:
        failure = {**base,
                   "source":        REGEN_SOURCE,
                   "parse_success": True,
                   "answer_correct": False,
                   "answer":        parsed["answer"],
                   "reasoning":     parsed["reasoning"],
                   "fail_reason":   "answer_mismatch",
                   "raw_snippet":   raw[:300]}
        return None, failure

    if not parsed["reasoning"]:
        failure = {**base,
                   "source":        REGEN_SOURCE,
                   "parse_success": True,
                   "answer_correct": True,
                   "answer":        parsed["answer"],
                   "reasoning":     "",
                   "fail_reason":   "empty_reasoning",
                   "raw_snippet":   raw[:300]}
        return None, failure

    banned = has_banned_phrase(parsed["reasoning"])
    if banned:
        failure = {**base,
                   "source":        REGEN_SOURCE,
                   "parse_success": True,
                   "answer_correct": True,
                   "answer":        parsed["answer"],
                   "reasoning":     parsed["reasoning"],
                   "fail_reason":   f"banned_phrase:{banned}",
                   "raw_snippet":   raw[:300]}
        return None, failure

    accepted = {**base,
                "source":        REGEN_SOURCE,
                "parse_success": True,
                "answer_correct": True,
                "answer":        parsed["answer"],
                "reasoning":     parsed["reasoning"]}
    return accepted, None

# ---------------------------------------------------------------------------
# Load V8 clean files
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

# ---------------------------------------------------------------------------
# QA audit
# ---------------------------------------------------------------------------

def run_audit(train_rows: list[dict], val_rows: list[dict],
              v8_train_len: int, v8_val_len: int,
              regen_accepted: int, regen_failed: int) -> dict:

    all_rows = train_rows + val_rows
    all_ids  = [r["id"] for r in all_rows]

    # Schema check
    required_keys = {"id", "task_type", "prompt", "gold_answer",
                     "source", "parse_success", "answer_correct",
                     "answer", "reasoning"}
    schema_issues = []
    for r in all_rows:
        missing = required_keys - set(r.keys())
        if missing:
            schema_issues.append({"id": r["id"], "missing": list(missing)})

    # Duplicate IDs
    seen, dup_ids = set(), []
    for rid in all_ids:
        if rid in seen:
            dup_ids.append(rid)
        seen.add(rid)

    # Train/val overlap
    train_ids = {r["id"] for r in train_rows}
    val_ids   = {r["id"] for r in val_rows}
    overlap   = list(train_ids & val_ids)

    # Empty reasoning
    empty_reasoning = [r["id"] for r in all_rows if not r.get("reasoning", "").strip()]

    # answer != gold_answer (float tolerance for gravity/unit_conversion)
    def answers_match(r):
        a, g, tt = str(r.get("answer", "")), str(r.get("gold_answer", "")), r.get("task_type", "")
        if a == g:
            return True
        if tt in ("gravity", "unit_conversion"):
            try:
                pf, gf = float(a), float(g)
                return abs(pf - gf) / max(abs(gf), 1) < 0.01
            except ValueError:
                pass
        return False

    answer_mismatch = [r["id"] for r in all_rows if not answers_match(r)]

    # Banned phrase scan
    banned_hits = []
    for r in all_rows:
        phrase = has_banned_phrase(r.get("reasoning", ""))
        if phrase:
            banned_hits.append({"id": r["id"], "phrase": phrase})

    # Source distribution
    from collections import Counter
    source_dist_train = Counter(r["source"] for r in train_rows)
    source_dist_val   = Counter(r["source"] for r in val_rows)

    # Task distribution
    task_dist_train = Counter(r["task_type"] for r in train_rows)
    task_dist_val   = Counter(r["task_type"] for r in val_rows)

    # Source breakdown for symbol_transform
    sym_train = [r for r in train_rows if r["task_type"] == "symbol_transform"]
    sym_val   = [r for r in val_rows   if r["task_type"] == "symbol_transform"]
    sym_src_train = Counter(r["source"] for r in sym_train)
    sym_src_val   = Counter(r["source"] for r in sym_val)

    # Random samples of regen rows
    regen_rows = [r for r in all_rows if r["source"] == REGEN_SOURCE]
    sample_regen = random.sample(regen_rows, min(5, len(regen_rows)))

    # Row count comparison
    row_count_comparison = {
        "v8_train":  v8_train_len,
        "v8_val":    v8_val_len,
        "v8_total":  v8_train_len + v8_val_len,
        "v8_1_train": len(train_rows),
        "v8_1_val":   len(val_rows),
        "v8_1_total": len(all_rows),
        "rows_removed": (v8_train_len + v8_val_len) - len(all_rows),
        "regen_accepted": regen_accepted,
        "regen_failed":   regen_failed,
    }

    passed = (
        not schema_issues
        and not dup_ids
        and not overlap
        and not empty_reasoning
        and not answer_mismatch
        and not banned_hits
    )

    return {
        "timestamp_utc":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed":          passed,
        "row_count_comparison": row_count_comparison,
        "schema_issues":   schema_issues,
        "duplicate_ids":   dup_ids,
        "train_val_overlap": overlap,
        "empty_reasoning_ids": empty_reasoning,
        "answer_mismatch_ids": answer_mismatch,
        "banned_phrase_hits":  banned_hits,
        "source_distribution": {
            "train": dict(source_dist_train),
            "val":   dict(source_dist_val),
        },
        "task_distribution": {
            "train": dict(task_dist_train),
            "val":   dict(task_dist_val),
        },
        "symbol_transform_source": {
            "train": dict(sym_src_train),
            "val":   dict(sym_src_val),
        },
        "regen_sample": [
            {"id": r["id"],
             "gold_answer": r["gold_answer"],
             "answer": r["answer"],
             "reasoning": r["reasoning"][:300]}
            for r in sample_regen
        ],
    }


def write_audit_md(audit: dict, path: str):
    rc  = audit["row_count_comparison"]
    sym = audit["symbol_transform_source"]

    lines = [
        "# V8.1 Dataset Audit",
        "",
        f"**Generated:** {audit['timestamp_utc']}",
        f"**Status:** {'PASS' if audit['passed'] else 'FAIL'}",
        "",
        "## Row Count Comparison",
        "",
        f"| Dataset | Train | Val | Total |",
        f"|---------|-------|-----|-------|",
        f"| V8 clean | {rc['v8_train']} | {rc['v8_val']} | {rc['v8_total']} |",
        f"| V8.1 clean | {rc['v8_1_train']} | {rc['v8_1_val']} | {rc['v8_1_total']} |",
        f"| Delta | {rc['v8_1_train']-rc['v8_train']} | {rc['v8_1_val']-rc['v8_val']} | {-rc['rows_removed']} |",
        "",
        f"Regen accepted: **{rc['regen_accepted']}**  |  Regen failed (excluded): **{rc['regen_failed']}**",
        "",
        "## QA Checks",
        "",
        "| Check | Result |",
        "|-------|--------|",
        "| Schema | " + ("OK" if not audit["schema_issues"] else f"FAIL ({len(audit['schema_issues'])} rows)") + " |",
        "| Duplicate IDs | " + ("OK" if not audit["duplicate_ids"] else f"FAIL ({len(audit['duplicate_ids'])})") + " |",
        "| Train/val overlap | " + ("OK" if not audit["train_val_overlap"] else f"FAIL ({len(audit['train_val_overlap'])})") + " |",
        "| Empty reasoning | " + ("OK" if not audit["empty_reasoning_ids"] else f"FAIL ({len(audit['empty_reasoning_ids'])})") + " |",
        "| answer == gold_answer | " + ("OK" if not audit["answer_mismatch_ids"] else f"FAIL ({len(audit['answer_mismatch_ids'])})") + " |",
        "| Banned phrases | " + ("OK" if not audit["banned_phrase_hits"] else f"FAIL ({len(audit['banned_phrase_hits'])} hits)") + " |",
        "",
        "## Source Distribution",
        "",
        "**Train:**",
    ]
    for src, n in sorted(audit["source_distribution"]["train"].items()):
        lines.append(f"- `{src}`: {n}")
    lines += ["", "**Val:**"]
    for src, n in sorted(audit["source_distribution"]["val"].items()):
        lines.append(f"- `{src}`: {n}")

    lines += [
        "",
        "## Task Distribution",
        "",
        "**Train:**",
    ]
    for task, n in sorted(audit["task_distribution"]["train"].items()):
        lines.append(f"- `{task}`: {n}")
    lines += ["", "**Val:**"]
    for task, n in sorted(audit["task_distribution"]["val"].items()):
        lines.append(f"- `{task}`: {n}")

    lines += [
        "",
        "## symbol_transform Source Breakdown",
        "",
        "**Train:**",
    ]
    for src, n in sorted(sym["train"].items()):
        lines.append(f"- `{src}`: {n}")
    lines += ["", "**Val:**"]
    for src, n in sorted(sym["val"].items()):
        lines.append(f"- `{src}`: {n}")

    lines += ["", "## Random Regen Samples", ""]
    for s in audit["regen_sample"]:
        lines += [
            f"**ID:** `{s['id']}`  **Gold:** `{s['gold_answer']}`  **Answer:** `{s['answer']}`",
            f"> {s['reasoning'][:250]}",
            "",
        ]

    if audit["schema_issues"]:
        lines += ["## Schema Issues", ""]
        for iss in audit["schema_issues"][:10]:
            lines.append(f"- `{iss['id']}`: missing {iss['missing']}")
        lines.append("")

    if audit["banned_phrase_hits"]:
        lines += ["## Banned Phrase Hits", ""]
        for h in audit["banned_phrase_hits"][:10]:
            lines.append(f"- `{h['id']}`: `{h['phrase']}`")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V8.1 symbol repair row regeneration"
    )
    parser.add_argument("--workers",    type=int, default=WORKERS)
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print first prompt and exit.")
    parser.add_argument("--audit-only", action="store_true",
                        help="Skip generation; re-run QA audit on existing V8.1 files.")
    args = parser.parse_args()

    # ---- Audit-only shortcut ------------------------------------------------
    if args.audit_only:
        print("Audit-only mode: loading existing V8.1 files...", flush=True)
        v8_train = load_jsonl(V8_TRAIN)
        v8_val   = load_jsonl(V8_VAL)
        v8_1_train = load_jsonl(V81_TRAIN)
        v8_1_val   = load_jsonl(V81_VAL)
        failures_list = load_jsonl(FAILURES_OUT) if os.path.exists(FAILURES_OUT) else []
        regen_accepted = sum(1 for r in v8_1_train + v8_1_val
                             if r.get("source") == REGEN_SOURCE)
        regen_failed = len(failures_list)
        audit = run_audit(v8_1_train, v8_1_val,
                          len(v8_train), len(v8_val),
                          regen_accepted, regen_failed)
        with open(AUDIT_JSON, "w") as f:
            json.dump(audit, f, indent=2)
        write_audit_md(audit, AUDIT_MD)
        status = "PASS" if audit["passed"] else "FAIL"
        print(f"Audit: {status}")
        if not audit["passed"]:
            for key in ("schema_issues", "duplicate_ids", "train_val_overlap",
                        "empty_reasoning_ids", "answer_mismatch_ids", "banned_phrase_hits"):
                if audit[key]:
                    print(f"  {key}: {len(audit[key])} issues")
        return

    if not args.dry_run:
        total_slots = check_server()
        if args.workers > total_slots:
            print(f"WARNING: workers={args.workers} > server slots={total_slots}")

    # ---- Load V8 clean files ------------------------------------------------
    print("Loading V8 clean files...", flush=True)
    v8_train = load_jsonl(V8_TRAIN)
    v8_val   = load_jsonl(V8_VAL)
    print(f"  V8 train: {len(v8_train)} rows  V8 val: {len(v8_val)} rows")

    # Split into keep / repair for each split
    train_keep   = [r for r in v8_train if r.get("source") != REPAIR_SOURCE]
    train_repair = [r for r in v8_train if r.get("source") == REPAIR_SOURCE]
    val_keep     = [r for r in v8_val   if r.get("source") != REPAIR_SOURCE]
    val_repair   = [r for r in v8_val   if r.get("source") == REPAIR_SOURCE]

    print(f"  Train repair rows: {len(train_repair)}  Val repair rows: {len(val_repair)}")
    print(f"  Train keep rows:   {len(train_keep)}    Val keep rows:   {len(val_keep)}")

    all_repair = [("train", r) for r in train_repair] + [("val", r) for r in val_repair]
    total = len(all_repair)

    # ---- Dry run ------------------------------------------------------------
    if args.dry_run:
        split, row = all_repair[0]
        prompt = format_prompt(str(row["gold_answer"]), row["prompt"])
        print(f"\n--- Dry run: id={row['id']} split={split} ---")
        print(prompt)
        print(f"\n(n_predict={N_PREDICT}, stop={STOP_SEQS})")
        return

    # ---- Generate -----------------------------------------------------------
    print(f"\nRegenerating {total} repair rows (workers={args.workers})...\n", flush=True)

    accepted_train: dict[str, dict] = {}
    accepted_val:   dict[str, dict] = {}
    failures:       list[dict]      = []
    lock = threading.Lock()
    done_count = [0]
    t_start = time.time()

    def process_and_track(split_row):
        split, row = split_row
        accepted, failure = process_row(row)
        with lock:
            done_count[0] += 1
            n = done_count[0]
            elapsed = time.time() - t_start
            rate = n / elapsed if elapsed > 0 else 0
            eta = (total - n) / rate if rate > 0 else 0
            if accepted:
                status = "OK"
                if split == "train":
                    accepted_train[row["id"]] = accepted
                else:
                    accepted_val[row["id"]] = accepted
            else:
                status = f"FAIL:{failure.get('fail_reason','?')}"
                failures.append(failure)
            print(
                f"  [{n}/{total}] {row['id']} -> {status}"
                f"  rate={rate:.2f}r/s  eta={eta/60:.1f}min",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_and_track, sr) for sr in all_repair]
        for f in as_completed(futures):
            pass  # progress printed inside process_and_track

    wall = time.time() - t_start
    regen_accepted = len(accepted_train) + len(accepted_val)
    regen_failed   = len(failures)

    print(f"\nGeneration done in {wall/60:.1f}min")
    print(f"  Accepted: {regen_accepted}  Failed/excluded: {regen_failed}")

    # ---- Write failures -----------------------------------------------------
    if failures:
        write_jsonl(FAILURES_OUT, failures)
        print(f"  Failures written: {FAILURES_OUT}")

    # ---- Build V8.1 files (preserve original order, swap repair rows) -------
    v8_1_train = []
    for r in v8_train:
        if r.get("source") != REPAIR_SOURCE:
            v8_1_train.append(r)
        elif r["id"] in accepted_train:
            v8_1_train.append(accepted_train[r["id"]])
        # else: failed regen → row excluded

    v8_1_val = []
    for r in v8_val:
        if r.get("source") != REPAIR_SOURCE:
            v8_1_val.append(r)
        elif r["id"] in accepted_val:
            v8_1_val.append(accepted_val[r["id"]])

    write_jsonl(V81_TRAIN, v8_1_train)
    write_jsonl(V81_VAL,   v8_1_val)
    print(f"\nV8.1 train: {len(v8_1_train)} rows  (V8: {len(v8_train)})")
    print(f"V8.1 val:   {len(v8_1_val)} rows    (V8: {len(v8_val)})")

    # ---- Audit --------------------------------------------------------------
    print("\nRunning QA audit...", flush=True)
    audit = run_audit(v8_1_train, v8_1_val,
                      len(v8_train), len(v8_val),
                      regen_accepted, regen_failed)

    with open(AUDIT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    write_audit_md(audit, AUDIT_MD)

    status = "PASS" if audit["passed"] else "FAIL"
    print(f"\nAudit: {status}")
    if not audit["passed"]:
        for key in ("schema_issues", "duplicate_ids", "train_val_overlap",
                    "empty_reasoning_ids", "answer_mismatch_ids", "banned_phrase_hits"):
            if audit[key]:
                print(f"  {key}: {len(audit[key])} issues")

    print(f"\nOutputs:")
    print(f"  {V81_TRAIN}")
    print(f"  {V81_VAL}")
    if failures:
        print(f"  {FAILURES_OUT}")
    print(f"  {AUDIT_JSON}")
    print(f"  {AUDIT_MD}")
    print("\nDone. Do NOT train until user reviews audit.")


if __name__ == "__main__":
    main()
