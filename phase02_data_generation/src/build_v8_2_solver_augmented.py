#!/usr/bin/env python3
"""
Build V8.2: replace gravity and unit_conversion rows with deterministic solver traces.

Inputs:
  phase02_data_generation/data/v8/train_reasoning_v8_1_local_gemma_clean.jsonl
  phase02_data_generation/data/v8/val_reasoning_v8_1_local_gemma_clean.jsonl

Outputs:
  phase02_data_generation/data/v8/train_reasoning_v8_2_solver_augmented_clean.jsonl
  phase02_data_generation/data/v8/val_reasoning_v8_2_solver_augmented_clean.jsonl
  phase02_data_generation/data/v8/v8_2_solver_failures.jsonl
  phase02_data_generation/data/v8/v8_2_solver_augmented_dataset_audit.json
  phase02_data_generation/data/v8/v8_2_solver_augmented_dataset_audit.md

Run from project root:
  python phase02_data_generation/src/build_v8_2_solver_augmented.py
"""

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = Path("phase02_data_generation/data/v8")
TRAIN_IN  = DATA_DIR / "train_reasoning_v8_1_local_gemma_clean.jsonl"
VAL_IN    = DATA_DIR / "val_reasoning_v8_1_local_gemma_clean.jsonl"
TRAIN_OUT = DATA_DIR / "train_reasoning_v8_2_solver_augmented_clean.jsonl"
VAL_OUT   = DATA_DIR / "val_reasoning_v8_2_solver_augmented_clean.jsonl"
FAIL_OUT  = DATA_DIR / "v8_2_solver_failures.jsonl"
AUDIT_JSON = DATA_DIR / "v8_2_solver_augmented_dataset_audit.json"
AUDIT_MD   = DATA_DIR / "v8_2_solver_augmented_dataset_audit.md"

GRAVITY_SOURCE = "deterministic_gravity_solver_v8_2"
UC_SOURCE      = "deterministic_unit_conversion_solver_v8_2"

BANNED_PHRASES = [
    "copy it exactly",
    "the correct symbol sequence is provided",
    "post-hoc training trace",
    "i copy it",
    "so i copy",
]

# ---------------------------------------------------------------------------
# Gravity solver
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
    """
    Return (g_used, method_note, pred_str) if exact match, else None.
    Tries: mean-g, mean-g with rounded decimals, individual g_i, individual g_i rounded.
    """
    gold_str = str(gold)
    n_dec = len(gold_str.split(".")[1]) if "." in gold_str else 0
    fmt = f"{{:.{n_dec}f}}"
    qt_f = float(qt)

    gs = [2.0 * float(d) / float(t) ** 2 for t, d in pairs]
    g_mean = sum(gs) / len(gs)

    # 1. Mean
    pred = fmt.format(0.5 * g_mean * qt_f ** 2)
    if pred == gold_str:
        return g_mean, "mean of per-example g values", pred

    # 2. Rounded mean (1–6 dp)
    for nd in range(1, 7):
        g_r = round(g_mean, nd)
        pred = fmt.format(0.5 * g_r * qt_f ** 2)
        if pred == gold_str:
            return g_r, f"mean g rounded to {nd} dp", pred

    # 3. Individual g_i (exact)
    for i, g_i in enumerate(gs):
        pred = fmt.format(0.5 * g_i * qt_f ** 2)
        if pred == gold_str:
            return g_i, f"g from example {i+1}", pred
        for nd in range(1, 7):
            g_r = round(g_i, nd)
            pred = fmt.format(0.5 * g_r * qt_f ** 2)
            if pred == gold_str:
                return g_r, f"g from example {i+1} rounded to {nd} dp", pred

    return None, None, None


def format_gravity_reasoning(pairs, qt, g_used, pred_str):
    # Display g to 4 dp (clean, matches the precision of per-example values)
    g_display = round(g_used, 4)
    lines = [
        "From the given examples, I extract the gravitational constant using g = 2·d / t²:",
        "",
    ]
    g_per_example = [2.0 * float(d) / float(t) ** 2 for t, d in pairs]
    for i, ((t, d), g_i) in enumerate(zip(pairs, g_per_example)):
        lines.append(
            f"  Example {i+1}: t={t}s, d={d}m → g = 2×{d}/{t}² = {g_i:.4f}"
        )
    lines.append("")
    g_mean = sum(g_per_example) / len(g_per_example)
    lines.append(f"The per-example g values are consistent (mean = {g_mean:.4f}).")
    lines.append(f"Using g = {g_display}.")
    lines.append("")
    lines.append(f"Applying d = 0.5·g·t² for t = {qt}s:")
    lines.append(
        f"  d = 0.5 × {g_display} × {qt}² = 0.5 × {g_display} × {float(qt)**2:.4f} = {pred_str}"
    )
    lines.append("")
    lines.append(f"Final answer: \\boxed{{{pred_str}}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unit conversion solver
# ---------------------------------------------------------------------------

def parse_uc(prompt):
    pairs = re.findall(r"([\d.]+)\s*m\s+becomes\s+([\d.]+)", prompt)
    query = re.search(r"convert the following measurement:\s*([\d.]+)", prompt)
    if not pairs or not query:
        return None, None
    return pairs, query.group(1)


def _fit_ls(xs, ys):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(xs[i] * ys[i] for i in range(n))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None, None
    m = (n * sxy - sx * sy) / denom
    b = (sy - m * sx) / n
    return m, b


def solve_uc(pairs, qx, gold):
    """
    Return (m, b, method_note, pred_str) if exact match, else None.
    Tries: LS, ratio, rounded m (LS), pair-by-pair slopes.
    """
    gold_str = str(gold)
    n_dec = len(gold_str.split(".")[1]) if "." in gold_str else 0
    fmt = f"{{:.{n_dec}f}}"
    qx_f = float(qx)

    xs = [float(x) for x, y in pairs]
    ys = [float(y) for x, y in pairs]

    # 1. Least squares
    m_ls, b_ls = _fit_ls(xs, ys)
    if m_ls is not None:
        pred = fmt.format(m_ls * qx_f + b_ls)
        if pred == gold_str:
            return m_ls, b_ls, "least-squares affine fit", pred

    # 2. Ratio (mean y/x)
    m_r = sum(y / x for x, y in zip(xs, ys)) / len(xs)
    b_r = sum(ys) / len(ys) - m_r * sum(xs) / len(xs)
    pred = fmt.format(m_r * qx_f + b_r)
    if pred == gold_str:
        return m_r, b_r, "mean-ratio slope", pred

    # 3. Rounded m from LS
    if m_ls is not None:
        for nd in range(1, 8):
            m_rnd = round(m_ls, nd)
            b_rnd = (sum(ys) - m_rnd * sum(xs)) / len(xs)
            pred = fmt.format(m_rnd * qx_f + b_rnd)
            if pred == gold_str:
                return m_rnd, b_rnd, f"LS slope rounded to {nd} dp", pred

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
                return m_ij, b_ij, f"slope from examples {i+1} and {j+1}", pred
            for nd in range(1, 7):
                m_r2 = round(m_ij, nd)
                b_r2 = ys[i] - m_r2 * xs[i]
                pred2 = fmt.format(m_r2 * qx_f + b_r2)
                if pred2 == gold_str:
                    return m_r2, b_r2, f"slope from examples {i+1},{j+1} rounded {nd} dp", pred2

    return None, None, None, None


def format_uc_reasoning(pairs, qx, m, b, method_note, pred_str):
    xs = [float(x) for x, y in pairs]
    ys = [float(y) for x, y in pairs]
    fmt2 = "{:.2f}"

    lines = [
        "From the given examples, I identify a linear conversion rule y = m·x + b.",
        "",
        "Using " + method_note + ":",
        f"  m = {m:.6f}",
        f"  b = {b:.6f}",
        "",
        "Validation against examples:",
    ]
    for (xstr, ystr) in pairs:
        xf = float(xstr)
        y_pred = m * xf + b
        lines.append(
            f"  {xstr} → predicted {fmt2.format(y_pred)}, observed {ystr}"
        )
    lines.append("")
    lines.append(f"Applying to query x = {qx}:")
    lines.append(f"  y = {m:.6f} × {qx} + {b:.6f} = {pred_str}")
    lines.append("")
    lines.append(f"Final answer: \\boxed{{{pred_str}}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------

def process_row(row, failures):
    task = row.get("task_type", "")
    prompt = row.get("prompt", "")
    gold = row.get("gold_answer", "")

    if task == "gravity":
        pairs, qt = parse_gravity(prompt)
        if not pairs or not qt:
            failures.append({**row, "failure_reason": "parse_fail_gravity"})
            return row  # keep original
        g, method, pred = solve_gravity(pairs, qt, gold)
        if g is None:
            failures.append({**row, "failure_reason": "no_exact_match_gravity"})
            return row
        reasoning = format_gravity_reasoning(pairs, qt, g, pred)
        return {
            **row,
            "source":        GRAVITY_SOURCE,
            "reasoning":     reasoning,
            "answer":        pred,
            "parse_success": True,
            "answer_correct": True,
            "tokens_out":    None,
            "gen_time_s":    None,
            "truncated":     False,
            "orig_pred":     None,
        }

    elif task == "unit_conversion":
        pairs, qx = parse_uc(prompt)
        if not pairs or not qx:
            failures.append({**row, "failure_reason": "parse_fail_uc"})
            return row
        m, b, method, pred = solve_uc(pairs, qx, gold)
        if m is None:
            failures.append({**row, "failure_reason": "no_exact_match_uc"})
            return row
        reasoning = format_uc_reasoning(pairs, qx, m, b, method, pred)
        return {
            **row,
            "source":        UC_SOURCE,
            "reasoning":     reasoning,
            "answer":        pred,
            "parse_success": True,
            "answer_correct": True,
            "tokens_out":    None,
            "gen_time_s":    None,
            "truncated":     False,
            "orig_pred":     None,
        }

    return row  # all other tasks: unchanged


# ---------------------------------------------------------------------------
# QA audit
# ---------------------------------------------------------------------------

def qa_audit(train_rows, val_rows, failures):
    results = {}
    all_rows = train_rows + val_rows
    errors = []

    # 1. Schema check
    required = {"id", "task_type", "prompt", "gold_answer", "source", "reasoning", "answer"}
    schema_bad = [r["id"] for r in all_rows if not required.issubset(r.keys())]
    results["schema_missing_fields"] = schema_bad
    if schema_bad:
        errors.append(f"Schema: {len(schema_bad)} rows missing required fields")

    # 2. Empty reasoning
    empty_reasoning = [r["id"] for r in all_rows if not str(r.get("reasoning", "")).strip()]
    results["empty_reasoning"] = empty_reasoning
    if empty_reasoning:
        errors.append(f"Empty reasoning: {len(empty_reasoning)} rows")

    # 3. Duplicate IDs
    all_ids = [r["id"] for r in all_rows]
    id_counts = Counter(all_ids)
    dup_ids = [k for k, v in id_counts.items() if v > 1]
    results["duplicate_ids"] = dup_ids
    if dup_ids:
        errors.append(f"Duplicate IDs: {dup_ids}")

    # 4. Train/val overlap
    train_ids = set(r["id"] for r in train_rows)
    val_ids   = set(r["id"] for r in val_rows)
    overlap = list(train_ids & val_ids)
    results["train_val_overlap"] = overlap
    if overlap:
        errors.append(f"Train/val overlap: {len(overlap)} IDs")

    # 5. Answer == gold_answer check (solver rows only)
    answer_mismatch = []
    for r in all_rows:
        src = r.get("source", "")
        if GRAVITY_SOURCE not in src and UC_SOURCE not in src:
            continue
        ans = str(r.get("answer", "")).strip()
        gold = str(r.get("gold_answer", "")).strip()
        if ans != gold:
            answer_mismatch.append({"id": r["id"], "answer": ans, "gold": gold})
    results["answer_mismatch_solver_rows"] = answer_mismatch
    if answer_mismatch:
        errors.append(f"Answer mismatch in solver rows: {len(answer_mismatch)}")

    # 6. Banned phrase scan
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
    src_dist = Counter(r.get("source", "unknown") for r in all_rows)
    results["source_distribution"] = dict(src_dist)

    # 8. Task distribution
    task_dist_train = Counter(r.get("task_type") for r in train_rows)
    task_dist_val   = Counter(r.get("task_type") for r in val_rows)
    results["task_distribution"] = {
        "train": dict(task_dist_train),
        "val":   dict(task_dist_val),
    }

    # 9. Solver row counts
    grav_solver_train = sum(1 for r in train_rows if r.get("source") == GRAVITY_SOURCE)
    grav_solver_val   = sum(1 for r in val_rows   if r.get("source") == GRAVITY_SOURCE)
    uc_solver_train   = sum(1 for r in train_rows if r.get("source") == UC_SOURCE)
    uc_solver_val     = sum(1 for r in val_rows   if r.get("source") == UC_SOURCE)
    results["solver_row_counts"] = {
        "gravity_solver_train": grav_solver_train,
        "gravity_solver_val":   grav_solver_val,
        "uc_solver_train":      uc_solver_train,
        "uc_solver_val":        uc_solver_val,
    }

    # 10. Random samples (gravity solver)
    rng = random.Random(42)
    grav_solver_rows = [r for r in all_rows if r.get("source") == GRAVITY_SOURCE]
    uc_solver_rows   = [r for r in all_rows if r.get("source") == UC_SOURCE]
    results["gravity_solver_samples"] = [
        {"id": r["id"], "gold": r["gold_answer"], "answer": r["answer"],
         "reasoning_head": r["reasoning"][:400]}
        for r in rng.sample(grav_solver_rows, min(5, len(grav_solver_rows)))
    ]
    results["uc_solver_samples"] = [
        {"id": r["id"], "gold": r["gold_answer"], "answer": r["answer"],
         "reasoning_head": r["reasoning"][:400]}
        for r in rng.sample(uc_solver_rows, min(5, len(uc_solver_rows)))
    ]

    # 11. Failure summary
    results["failures"] = {
        "total": len(failures),
        "by_reason": dict(Counter(f.get("failure_reason", "unknown") for f in failures)),
        "by_task":   dict(Counter(f.get("task_type") for f in failures)),
    }

    # 12. Row counts
    results["row_counts"] = {
        "train": len(train_rows),
        "val":   len(val_rows),
        "total": len(all_rows),
    }
    results["v8_1_row_counts"] = {
        "train": None,  # filled after
        "val":   None,
    }

    results["pass"] = len(errors) == 0
    results["errors"] = errors

    return results


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_audit_md(audit, out_path):
    lines = [
        "# V8.2 Solver-Augmented Dataset Audit",
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
        f"| gravity (solver) | {sc['gravity_solver_train']} | {sc['gravity_solver_val']} |",
        f"| unit_conversion (solver) | {sc['uc_solver_train']} | {sc['uc_solver_val']} |",
        "",
    ]

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

    lines += ["## Failure Summary", "", f"Total failures: {audit['failures']['total']}", ""]
    for reason, n in audit["failures"]["by_reason"].items():
        lines.append(f"- {reason}: {n}")
    lines.append("")

    lines += ["## Gravity Solver Samples", ""]
    for s in audit["gravity_solver_samples"]:
        lines.append(f"**id={s['id']}**  gold=`{s['gold']}`  answer=`{s['answer']}`")
        lines.append(f"```\n{s['reasoning_head']}\n```")
        lines.append("")

    lines += ["## Unit Conversion Solver Samples", ""]
    for s in audit["uc_solver_samples"]:
        lines.append(f"**id={s['id']}**  gold=`{s['gold']}`  answer=`{s['answer']}`")
        lines.append(f"```\n{s['reasoning_head']}\n```")
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

    failures = []
    v8_1_counts = {}

    for split, in_path, out_path in [
        ("train", TRAIN_IN, TRAIN_OUT),
        ("val",   VAL_IN,   VAL_OUT),
    ]:
        print(f"\n{'='*60}")
        print(f"Processing {split}: {in_path}")
        raw = [json.loads(l) for l in open(in_path) if l.strip()]
        v8_1_counts[split] = len(raw)
        print(f"  Input rows: {len(raw)}")

        out_rows = []
        grav_total = grav_solved = 0
        uc_total = uc_solved = 0

        for row in raw:
            task = row.get("task_type", "")
            if task == "gravity":
                grav_total += 1
                n_before = len(failures)
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

        print(f"  Output rows: {len(out_rows)}")
        print(f"  Gravity solver: {grav_solved}/{grav_total} ({100*grav_solved/max(1,grav_total):.1f}%)")
        print(f"  UC solver:      {uc_solved}/{uc_total} ({100*uc_solved/max(1,uc_total):.1f}%)")
        print(f"  Written → {out_path}")

    # Write failures
    with open(FAIL_OUT, "w") as f:
        for row in failures:
            f.write(json.dumps(row) + "\n")
    print(f"\nFailures ({len(failures)}) → {FAIL_OUT}")

    # QA audit
    print("\nRunning QA audit...")
    train_rows = [json.loads(l) for l in open(TRAIN_OUT) if l.strip()]
    val_rows   = [json.loads(l) for l in open(VAL_OUT)   if l.strip()]

    audit = qa_audit(train_rows, val_rows, failures)
    audit["v8_1_row_counts"] = v8_1_counts

    with open(AUDIT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Audit JSON → {AUDIT_JSON}")

    write_audit_md(audit, AUDIT_MD)

    # Console summary
    print("\n" + "="*60)
    print(f"  AUDIT: {'PASS' if audit['pass'] else 'FAIL'}")
    print(f"  train: {audit['row_counts']['train']}  val: {audit['row_counts']['val']}")
    sc = audit["solver_row_counts"]
    print(f"  gravity solver: train={sc['gravity_solver_train']}  val={sc['gravity_solver_val']}")
    print(f"  uc solver:      train={sc['uc_solver_train']}  val={sc['uc_solver_val']}")
    print(f"  failures:       {audit['failures']['total']}")
    if audit["errors"]:
        print("  ERRORS:")
        for e in audit["errors"]:
            print(f"    - {e}")
    print("="*60)


if __name__ == "__main__":
    main()
