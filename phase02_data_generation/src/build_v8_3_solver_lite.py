#!/usr/bin/env python3
"""
Build V8.3-lite: V8.1 base with gravity and unit_conversion rows replaced by
compact single-paragraph deterministic solver traces (avoids V8.2's verbose
multi-line traces that hurt parse rate and diluted other tasks).

Inputs:  V8.1 train/val
Outputs:
  phase02_data_generation/data/v8/train_reasoning_v8_3_solver_lite_clean.jsonl
  phase02_data_generation/data/v8/val_reasoning_v8_3_solver_lite_clean.jsonl
  phase02_data_generation/data/v8/v8_3_solver_lite_failures.jsonl
  phase02_data_generation/data/v8/v8_3_solver_lite_dataset_audit.json
  phase02_data_generation/data/v8/v8_3_solver_lite_dataset_audit.md

Run from project root:
  python phase02_data_generation/src/build_v8_3_solver_lite.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR   = Path("phase02_data_generation/data/v8")
TRAIN_IN   = DATA_DIR / "train_reasoning_v8_1_local_gemma_clean.jsonl"
VAL_IN     = DATA_DIR / "val_reasoning_v8_1_local_gemma_clean.jsonl"
TRAIN_OUT  = DATA_DIR / "train_reasoning_v8_3_solver_lite_clean.jsonl"
VAL_OUT    = DATA_DIR / "val_reasoning_v8_3_solver_lite_clean.jsonl"
FAIL_OUT   = DATA_DIR / "v8_3_solver_lite_failures.jsonl"
AUDIT_JSON = DATA_DIR / "v8_3_solver_lite_dataset_audit.json"
AUDIT_MD   = DATA_DIR / "v8_3_solver_lite_dataset_audit.md"

GRAVITY_SOURCE = "deterministic_gravity_solver_lite_v8_3"
UC_SOURCE      = "deterministic_unit_conversion_solver_lite_v8_3"

BANNED_PHRASES = [
    "copy it exactly",
    "the correct symbol sequence is provided",
    "post-hoc training trace",
    "i copy it",
    "so i copy",
]

# ---------------------------------------------------------------------------
# Gravity solver (same logic as V8.2)
# ---------------------------------------------------------------------------

def parse_gravity(prompt):
    pairs = re.findall(
        r't\s*=\s*([\d.]+)s.*?distance\s*=\s*([\d.]+)\s*m', prompt, re.DOTALL
    )
    query = re.search(r't\s*=\s*([\d.]+)s\s+given', prompt)
    if not pairs or not query:
        return None, None
    return pairs, query.group(1)


def solve_gravity(pairs, qt, gold):
    """Return (g_used, pred_str) or (None, None)."""
    gold_str = str(gold)
    n_dec = len(gold_str.split(".")[1]) if "." in gold_str else 0
    fmt   = f"{{:.{n_dec}f}}"
    qt_f  = float(qt)

    gs     = [2.0 * float(d) / float(t) ** 2 for t, d in pairs]
    g_mean = sum(gs) / len(gs)

    # 1. mean
    pred = fmt.format(0.5 * g_mean * qt_f ** 2)
    if pred == gold_str:
        return g_mean, pred

    # 2. rounded mean (1–6 dp)
    for nd in range(1, 7):
        g_r  = round(g_mean, nd)
        pred = fmt.format(0.5 * g_r * qt_f ** 2)
        if pred == gold_str:
            return g_r, pred

    # 3. individual g_i (exact then rounded)
    for g_i in gs:
        pred = fmt.format(0.5 * g_i * qt_f ** 2)
        if pred == gold_str:
            return g_i, pred
        for nd in range(1, 7):
            g_r  = round(g_i, nd)
            pred = fmt.format(0.5 * g_r * qt_f ** 2)
            if pred == gold_str:
                return g_r, pred

    return None, None


def format_gravity_lite(pairs, qt, g_used, pred_str):
    """Compact single-paragraph gravity trace."""
    g_display = round(g_used, 4)
    qt_sq     = round(float(qt) ** 2, 4)
    return (
        f"From the examples, compute g = 2d/t² and average the consistent values, "
        f"giving g = {g_display}. "
        f"Then apply d = 0.5·g·t² to the query time t = {qt}: "
        f"d = 0.5 × {g_display} × {qt}²"
        f" = 0.5 × {g_display} × {qt_sq}"
        f" = {pred_str}. "
        f"Therefore \\boxed{{{pred_str}}}."
    )


# ---------------------------------------------------------------------------
# Unit conversion solver (same logic as V8.2)
# ---------------------------------------------------------------------------

def parse_uc(prompt):
    pairs = re.findall(r"([\d.]+)\s*m\s+becomes\s+([\d.]+)", prompt)
    query = re.search(r"convert the following measurement:\s*([\d.]+)", prompt)
    if not pairs or not query:
        return None, None
    return pairs, query.group(1)


def _fit_ls(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(xs[i] * ys[i] for i in range(n))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None, None
    m = (n * sxy - sx * sy) / denom
    b = (sy - m * sx) / n
    return m, b


def solve_uc(pairs, qx, gold):
    """Return (m, b, pred_str) or (None, None, None)."""
    gold_str = str(gold)
    n_dec    = len(gold_str.split(".")[1]) if "." in gold_str else 0
    fmt      = f"{{:.{n_dec}f}}"
    qx_f     = float(qx)
    xs       = [float(x) for x, y in pairs]
    ys       = [float(y) for x, y in pairs]

    # 1. LS
    m_ls, b_ls = _fit_ls(xs, ys)
    if m_ls is not None:
        pred = fmt.format(m_ls * qx_f + b_ls)
        if pred == gold_str:
            return m_ls, b_ls, pred

    # 2. Ratio
    m_r = sum(y / x for x, y in zip(xs, ys)) / len(xs)
    b_r = sum(ys) / len(ys) - m_r * sum(xs) / len(xs)
    pred = fmt.format(m_r * qx_f + b_r)
    if pred == gold_str:
        return m_r, b_r, pred

    # 3. Rounded m from LS
    if m_ls is not None:
        for nd in range(1, 8):
            m_rnd  = round(m_ls, nd)
            b_rnd  = (sum(ys) - m_rnd * sum(xs)) / len(xs)
            pred   = fmt.format(m_rnd * qx_f + b_rnd)
            if pred == gold_str:
                return m_rnd, b_rnd, pred

    # 4. Pair-by-pair slopes
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            if i == j:
                continue
            dx = xs[j] - xs[i]
            if abs(dx) < 1e-9:
                continue
            m_ij = (ys[j] - ys[i]) / dx
            b_ij = ys[i] - m_ij * xs[i]
            pred = fmt.format(m_ij * qx_f + b_ij)
            if pred == gold_str:
                return m_ij, b_ij, pred
            for nd in range(1, 7):
                m_r2  = round(m_ij, nd)
                b_r2  = ys[i] - m_r2 * xs[i]
                pred2 = fmt.format(m_r2 * qx_f + b_r2)
                if pred2 == gold_str:
                    return m_r2, b_r2, pred2

    return None, None, None


def format_uc_lite(qx, m, b, pred_str):
    """Compact single-paragraph unit conversion trace."""
    m_disp = round(m, 6)
    b_disp = round(b, 6)
    b_part = f" + {b_disp}" if b_disp >= 0 else f" − {abs(b_disp)}"
    return (
        f"The examples fit an affine rule y = m·x + b "
        f"with m = {m_disp} and b = {b_disp}. "
        f"Applying it to x = {qx} gives "
        f"y = {m_disp} × {qx}{b_part} = {pred_str}. "
        f"Therefore \\boxed{{{pred_str}}}."
    )


# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------

def process_row(row, failures):
    task   = row.get("task_type", "")
    prompt = row.get("prompt", "")
    gold   = row.get("gold_answer", "")

    if task == "gravity":
        pairs, qt = parse_gravity(prompt)
        if not pairs or not qt:
            failures.append({**row, "failure_reason": "parse_fail_gravity"})
            return row
        g, pred = solve_gravity(pairs, qt, gold)
        if g is None:
            failures.append({**row, "failure_reason": "no_exact_match_gravity"})
            return row
        reasoning = format_gravity_lite(pairs, qt, g, pred)
        return {**row,
                "source": GRAVITY_SOURCE, "reasoning": reasoning,
                "answer": pred, "parse_success": True, "answer_correct": True,
                "tokens_out": None, "gen_time_s": None,
                "truncated": False, "orig_pred": None}

    elif task == "unit_conversion":
        pairs, qx = parse_uc(prompt)
        if not pairs or not qx:
            failures.append({**row, "failure_reason": "parse_fail_uc"})
            return row
        m, b, pred = solve_uc(pairs, qx, gold)
        if m is None:
            failures.append({**row, "failure_reason": "no_exact_match_uc"})
            return row
        reasoning = format_uc_lite(qx, m, b, pred)
        return {**row,
                "source": UC_SOURCE, "reasoning": reasoning,
                "answer": pred, "parse_success": True, "answer_correct": True,
                "tokens_out": None, "gen_time_s": None,
                "truncated": False, "orig_pred": None}

    return row


# ---------------------------------------------------------------------------
# QA audit
# ---------------------------------------------------------------------------

def word_count(text):
    return len(str(text).split())


def qa_audit(train_rows, val_rows, failures,
             v81_train, v81_val, v82_train=None, v82_val=None):
    all_rows = train_rows + val_rows
    errors   = []
    results  = {}

    # 1. Schema
    required = {"id", "task_type", "prompt", "gold_answer", "source", "reasoning", "answer"}
    bad_schema = [r["id"] for r in all_rows if not required.issubset(r.keys())]
    results["schema_missing_fields"] = bad_schema
    if bad_schema:
        errors.append(f"Schema: {len(bad_schema)} rows missing required fields")

    # 2. Empty reasoning
    empty_r = [r["id"] for r in all_rows if not str(r.get("reasoning", "")).strip()]
    results["empty_reasoning"] = empty_r
    if empty_r:
        errors.append(f"Empty reasoning: {len(empty_r)} rows")

    # 3. Duplicate IDs
    id_counts = Counter(r["id"] for r in all_rows)
    dup_ids   = [k for k, v in id_counts.items() if v > 1]
    results["duplicate_ids"] = dup_ids
    if dup_ids:
        errors.append(f"Duplicate IDs: {dup_ids}")

    # 4. Train/val overlap
    train_ids = set(r["id"] for r in train_rows)
    val_ids   = set(r["id"] for r in val_rows)
    overlap   = list(train_ids & val_ids)
    results["train_val_overlap"] = overlap
    if overlap:
        errors.append(f"Train/val overlap: {len(overlap)} IDs")

    # 5. Answer == gold (solver rows only)
    mismatch = []
    for r in all_rows:
        src = r.get("source", "")
        if GRAVITY_SOURCE not in src and UC_SOURCE not in src:
            continue
        if str(r.get("answer", "")).strip() != str(r.get("gold_answer", "")).strip():
            mismatch.append({"id": r["id"], "answer": r.get("answer"), "gold": r.get("gold_answer")})
    results["answer_mismatch_solver_rows"] = mismatch
    if mismatch:
        errors.append(f"Answer mismatch: {len(mismatch)}")

    # 6. Banned phrases
    banned_hits = []
    for r in all_rows:
        text = (str(r.get("reasoning", "")) + " " + str(r.get("answer", ""))).lower()
        for phrase in BANNED_PHRASES:
            if phrase in text:
                banned_hits.append({"id": r["id"], "task": r.get("task_type"), "phrase": phrase})
                break
    results["banned_phrase_hits"] = banned_hits
    if banned_hits:
        errors.append(f"Banned phrase hits: {len(banned_hits)}")

    # 7. Source distribution
    results["source_distribution"] = dict(Counter(r.get("source", "?") for r in all_rows))

    # 8. Task distribution
    results["task_distribution"] = {
        "train": dict(Counter(r.get("task_type") for r in train_rows)),
        "val":   dict(Counter(r.get("task_type") for r in val_rows)),
    }

    # 9. Solver row counts
    g_tr  = sum(1 for r in train_rows if r.get("source") == GRAVITY_SOURCE)
    g_val = sum(1 for r in val_rows   if r.get("source") == GRAVITY_SOURCE)
    u_tr  = sum(1 for r in train_rows if r.get("source") == UC_SOURCE)
    u_val = sum(1 for r in val_rows   if r.get("source") == UC_SOURCE)
    results["solver_row_counts"] = {
        "gravity_train": g_tr, "gravity_val": g_val,
        "uc_train": u_tr, "uc_val": u_val,
    }

    # 10. Reasoning word count comparison
    def wc_stats(rows, task_filter=None):
        counts = [word_count(r.get("reasoning", ""))
                  for r in rows
                  if task_filter is None or r.get("task_type") == task_filter]
        if not counts:
            return {"n": 0, "mean": 0, "median": 0, "p95": 0, "max": 0}
        counts.sort()
        return {
            "n":      len(counts),
            "mean":   round(sum(counts) / len(counts), 1),
            "median": counts[len(counts) // 2],
            "p95":    counts[int(len(counts) * 0.95)],
            "max":    counts[-1],
        }

    v81_all = v81_train + v81_val
    v83_all = train_rows + val_rows

    wc = {}
    for task in ("gravity", "unit_conversion", None):
        key = task or "all"
        wc[key] = {
            "v8_1": wc_stats(v81_all, task),
            "v8_3_lite": wc_stats(v83_all, task),
        }
        if v82_train and v82_val:
            wc[key]["v8_2"] = wc_stats(v82_train + v82_val, task)
    results["word_count_comparison"] = wc

    # 11. Row counts
    results["row_counts"] = {
        "train": len(train_rows), "val": len(val_rows),
        "total": len(all_rows),
    }

    # 12. Failures
    results["failures"] = {
        "total":     len(failures),
        "by_reason": dict(Counter(f.get("failure_reason", "?") for f in failures)),
        "by_task":   dict(Counter(f.get("task_type") for f in failures)),
    }

    # 13. Random samples
    rng = random.Random(42)
    g_rows = [r for r in all_rows if r.get("source") == GRAVITY_SOURCE]
    u_rows = [r for r in all_rows if r.get("source") == UC_SOURCE]
    results["gravity_lite_samples"] = [
        {"id": r["id"], "gold": r["gold_answer"], "answer": r["answer"],
         "reasoning": r["reasoning"], "word_count": word_count(r["reasoning"])}
        for r in rng.sample(g_rows, min(5, len(g_rows)))
    ]
    results["uc_lite_samples"] = [
        {"id": r["id"], "gold": r["gold_answer"], "answer": r["answer"],
         "reasoning": r["reasoning"], "word_count": word_count(r["reasoning"])}
        for r in rng.sample(u_rows, min(5, len(u_rows)))
    ]

    results["pass"]   = len(errors) == 0
    results["errors"] = errors
    return results


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_audit_md(audit, out_path):
    lines = [
        "# V8.3-Lite Solver Dataset Audit",
        "",
        f"**Pass:** {'YES' if audit['pass'] else 'NO — ERRORS FOUND'}",
        "",
    ]
    if audit["errors"]:
        lines += ["## Errors", ""]
        for e in audit["errors"]:
            lines.append(f"- {e}")
        lines.append("")

    rc = audit["row_counts"]
    lines += [
        "## Row Counts",
        "",
        f"| Split | Count |",
        f"|-------|-------|",
        f"| train | {rc['train']} |",
        f"| val   | {rc['val']} |",
        f"| total | {rc['total']} |",
        "",
    ]

    sc = audit["solver_row_counts"]
    lines += [
        "## Solver Row Counts",
        "",
        f"| Task | Train | Val |",
        f"|------|-------|-----|",
        f"| gravity (lite) | {sc['gravity_train']} | {sc['gravity_val']} |",
        f"| unit_conversion (lite) | {sc['uc_train']} | {sc['uc_val']} |",
        "",
    ]

    # Word count comparison table
    wc = audit["word_count_comparison"]
    lines += ["## Reasoning Word Count Comparison", ""]
    versions = [k for k in next(iter(wc.values())).keys()]
    header   = "| Task | " + " | ".join(f"{v} mean" for v in versions) + \
               " | " + " | ".join(f"{v} p95" for v in versions) + " |"
    sep      = "|------|" + "|".join("----" for _ in versions * 2) + "|"
    lines += [header, sep]
    for task, stats in wc.items():
        means = " | ".join(str(stats[v]["mean"]) for v in versions)
        p95s  = " | ".join(str(stats[v]["p95"])  for v in versions)
        lines.append(f"| {task} | {means} | {p95s} |")
    lines.append("")

    lines += ["## Source Distribution", ""]
    for src, n in sorted(audit["source_distribution"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{src}`: {n}")
    lines.append("")

    lines += ["## Task Distribution (train)", ""]
    for task, n in sorted(audit["task_distribution"]["train"].items()):
        lines.append(f"- {task}: {n}")
    lines.append("")

    lines += ["## Task Distribution (val)", ""]
    for task, n in sorted(audit["task_distribution"]["val"].items()):
        lines.append(f"- {task}: {n}")
    lines.append("")

    lines += ["## Failure Summary", "", f"Total: {audit['failures']['total']}", ""]
    for reason, n in audit["failures"]["by_reason"].items():
        lines.append(f"- {reason}: {n}")
    lines.append("")

    lines += ["## Gravity Lite Samples", ""]
    for s in audit["gravity_lite_samples"]:
        lines.append(f"**id={s['id']}**  gold=`{s['gold']}`  answer=`{s['answer']}`  words={s['word_count']}")
        lines.append(f"> {s['reasoning']}")
        lines.append("")

    lines += ["## Unit Conversion Lite Samples", ""]
    for s in audit["uc_lite_samples"]:
        lines.append(f"**id={s['id']}**  gold=`{s['gold']}`  answer=`{s['answer']}`  words={s['word_count']}")
        lines.append(f"> {s['reasoning']}")
        lines.append("")

    lines += [
        "## QA Checks",
        "",
        f"- Schema: {'PASS' if not audit['schema_missing_fields'] else 'FAIL ' + str(len(audit['schema_missing_fields']))}",
        f"- Empty reasoning: {'PASS' if not audit['empty_reasoning'] else 'FAIL ' + str(len(audit['empty_reasoning']))}",
        f"- Duplicate IDs: {'PASS' if not audit['duplicate_ids'] else 'FAIL ' + str(audit['duplicate_ids'])}",
        f"- Train/val overlap: {'PASS' if not audit['train_val_overlap'] else 'FAIL'}",
        f"- Answer==gold (solver rows): {'PASS' if not audit['answer_mismatch_solver_rows'] else 'FAIL ' + str(len(audit['answer_mismatch_solver_rows']))}",
        f"- Banned phrases: {'PASS' if not audit['banned_phrase_hits'] else 'FAIL ' + str(len(audit['banned_phrase_hits']))}",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Audit MD → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load V8.2 for word-count comparison (optional, skip if missing)
    v82_train = v82_val = None
    v82_train_path = DATA_DIR / "train_reasoning_v8_2_solver_augmented_clean.jsonl"
    v82_val_path   = DATA_DIR / "val_reasoning_v8_2_solver_augmented_clean.jsonl"
    if v82_train_path.exists() and v82_val_path.exists():
        v82_train = [json.loads(l) for l in open(v82_train_path) if l.strip()]
        v82_val   = [json.loads(l) for l in open(v82_val_path)   if l.strip()]
        print(f"Loaded V8.2 for word-count comparison: {len(v82_train)} train / {len(v82_val)} val")

    failures   = []
    all_splits = {}

    for split, in_path, out_path in [
        ("train", TRAIN_IN, TRAIN_OUT),
        ("val",   VAL_IN,   VAL_OUT),
    ]:
        print(f"\n{'='*60}")
        print(f"Processing {split}: {in_path}")
        raw = [json.loads(l) for l in open(in_path) if l.strip()]
        print(f"  Input rows: {len(raw)}")

        out_rows = []
        grav_total = grav_solved = 0
        uc_total   = uc_solved   = 0

        for row in raw:
            task = row.get("task_type", "")
            if task == "gravity":
                grav_total += 1
                new_row = process_row(row, failures)
                if new_row.get("source") == GRAVITY_SOURCE:
                    grav_solved += 1
            elif task == "unit_conversion":
                uc_total += 1
                new_row = process_row(row, failures)
                if new_row.get("source") == UC_SOURCE:
                    uc_solved += 1
            else:
                new_row = row
            out_rows.append(new_row)

        with open(out_path, "w") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")

        all_splits[split] = out_rows
        print(f"  Output rows:    {len(out_rows)}")
        print(f"  Gravity solved: {grav_solved}/{grav_total} ({100*grav_solved/max(1,grav_total):.1f}%)")
        print(f"  UC solved:      {uc_solved}/{uc_total} ({100*uc_solved/max(1,uc_total):.1f}%)")
        print(f"  Written → {out_path}")

    with open(FAIL_OUT, "w") as f:
        for row in failures:
            f.write(json.dumps(row) + "\n")
    print(f"\nFailures ({len(failures)}) → {FAIL_OUT}")

    # Load V8.1 for word-count baseline
    v81_train = [json.loads(l) for l in open(TRAIN_IN) if l.strip()]
    v81_val   = [json.loads(l) for l in open(VAL_IN)   if l.strip()]

    print("\nRunning QA audit...")
    audit = qa_audit(
        all_splits["train"], all_splits["val"],
        failures, v81_train, v81_val, v82_train, v82_val,
    )

    with open(AUDIT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Audit JSON → {AUDIT_JSON}")
    write_audit_md(audit, AUDIT_MD)

    # Console summary
    print("\n" + "=" * 60)
    print(f"  AUDIT: {'PASS' if audit['pass'] else 'FAIL'}")
    rc = audit["row_counts"]
    sc = audit["solver_row_counts"]
    print(f"  train={rc['train']}  val={rc['val']}")
    print(f"  gravity lite:  train={sc['gravity_train']}  val={sc['gravity_val']}")
    print(f"  uc lite:       train={sc['uc_train']}  val={sc['uc_val']}")
    print(f"  failures:      {audit['failures']['total']}")

    wc = audit["word_count_comparison"]
    print("\n  Reasoning word count comparison:")
    for task in ("gravity", "unit_conversion", "all"):
        row_wc = wc[task]
        parts  = "  ".join(f"{v}=mean{s['mean']}/p95={s['p95']}"
                           for v, s in row_wc.items())
        print(f"    {task:<18}: {parts}")

    if audit["errors"]:
        print("  ERRORS:")
        for e in audit["errors"]:
            print(f"    - {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
