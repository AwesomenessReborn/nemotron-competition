import argparse
import json
import os
from collections import defaultdict

DEFAULT_PATHS = {
    "gemini-2.5-flash": "phase01_teacher_benchmark/data/generated/teacher_test_gemini_2.5_flash.jsonl",
    "claude-haiku-4-5": "phase01_teacher_benchmark/data/generated/teacher_test_haiku.jsonl",
}
REPORT_PATH = "phase01_teacher_benchmark/data/generated/comparison_report.json"


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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


def evaluate(records):
    by_task = defaultdict(list)
    for r in records:
        r["_correct"] = answer_correct(r["answer"], r["gold_answer"], r["task_type"])
        by_task[r["task_type"]].append(r)
    return by_task


def col(correct, total):
    pct = int(100 * correct / total) if total else 0
    return f"{correct}/{total}  {pct}%"


def _short(model_id):
    """Derive a compact display name from a model ID."""
    parts = model_id.split("-")
    # Pick the first part that is not purely numeric/dots (e.g. 'gemini', 'haiku')
    for p in parts:
        if p and not p.replace(".", "").isdigit():
            return p
    return parts[0]


def winner_label(scores, model_ids):
    """Return short name of winner or 'tie'. scores: {model_id: (correct, total)}"""
    rates = {m: scores[m][0] / scores[m][1] if scores[m][1] else 0 for m in model_ids}
    best_rate = max(rates.values())
    winners = [m for m, r in rates.items() if r == best_rate]
    if len(winners) > 1:
        return "tie"
    return _short(winners[0])


def print_accuracy_table(task_types, model_data):
    model_ids = list(model_data.keys())
    header = f"{'task_type':<20}" + "".join(f" | {m:<20}" for m in model_ids) + " | winner"
    sep = "-" * (22 + 23 * len(model_ids) + 9)
    print(header)
    print(sep)

    totals = {m: [0, 0] for m in model_ids}
    per_task = {}

    for tt in sorted(task_types):
        row_str = f"{tt:<20}"
        per_task[tt] = {}
        scores = {}
        for m in model_ids:
            rows = model_data[m].get(tt, [])
            c = sum(r["_correct"] for r in rows)
            n = len(rows)
            totals[m][0] += c
            totals[m][1] += n
            per_task[tt][m] = {"correct": c, "total": n}
            scores[m] = (c, n)
            row_str += f" | {col(c, n):<20}"
        row_str += f" | {winner_label(scores, model_ids)}"
        print(row_str)

    print(sep)
    total_row = f"{'TOTAL':<20}"
    total_scores = {}
    for m in model_ids:
        c, n = totals[m]
        total_scores[m] = (c, n)
        total_row += f" | {col(c, n):<20}"
    total_row += f" | {winner_label(total_scores, model_ids)}"
    print(total_row)
    return per_task, totals


def print_metrics_table(model_records):
    def metrics(records):
        n = len(records)
        if n == 0:
            return 0, 0, 0, 0
        match_rate = sum(r["_correct"] for r in records) / n
        parse_rate = sum(r["parse_success"] for r in records) / n
        avg_words = sum(len(r["reasoning"].split()) for r in records) / n
        valid_times = [r["gen_time"] for r in records if r["gen_time"] >= 0]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
        return match_rate, parse_rate, avg_words, avg_time

    model_ids = list(model_records.keys())
    computed = {m: metrics(model_records[m]) for m in model_ids}

    labels = [
        ("Answer match rate",   lambda v: f"{v[0]:.1%}"),
        ("Parse success rate",  lambda v: f"{v[1]:.1%}"),
        ("Avg reasoning words", lambda v: f"{v[2]:.0f}"),
        ("Avg gen time (s)",    lambda v: f"{v[3]:.2f}"),
    ]

    header = f"{'Metric':<25}" + "".join(f" | {m:<20}" for m in model_ids)
    print(header)
    print("-" * (27 + 23 * len(model_ids)))
    for label, fmt in labels:
        row = f"{label:<25}" + "".join(f" | {fmt(computed[m]):<20}" for m in model_ids)
        print(row)

    return {m: {"match_rate": computed[m][0], "parse_rate": computed[m][1],
                "avg_words": computed[m][2], "avg_time": computed[m][3]}
            for m in model_ids}


def print_sample_traces(model_records, n_samples=3):
    model_ids = list(model_records.keys())
    first_model = model_records[model_ids[0]]
    by_id = {m: {r["id"]: r for r in model_records[m]} for m in model_ids}

    wrong_by_task = defaultdict(list)
    for r in first_model:
        if not r["_correct"]:
            wrong_by_task[r["task_type"]].append(r["id"])

    samples = []
    seen_tasks = set()
    for tt in sorted(wrong_by_task.keys()):
        if len(samples) >= n_samples:
            break
        samples.append((tt, wrong_by_task[tt][0]))
        seen_tasks.add(tt)

    for r in first_model:
        if len(samples) >= n_samples:
            break
        if r["task_type"] not in seen_tasks:
            samples.append((r["task_type"], r["id"]))
            seen_tasks.add(r["task_type"])

    print(f"\n{'='*80}")
    print("SAMPLE TRACES (worst-performing first)")
    print("=" * 80)

    for tt, rid in samples:
        ref = by_id[model_ids[0]].get(rid)
        if not ref:
            continue
        print(f"\n--- task_type: {tt} | id: {rid} ---")
        print(f"PROMPT: {ref['prompt'][:200]}{'...' if len(ref['prompt']) > 200 else ''}")
        print(f"GOLD:   {ref['gold_answer']}")
        for m in model_ids:
            r = by_id[m].get(rid)
            if r:
                tick = "✓" if r["_correct"] else "✗"
                print(f"\n[{m}]")
                print(f"  REASONING: {r['reasoning'][:300]}{'...' if len(r['reasoning']) > 300 else ''}")
                print(f"  ANSWER: {r['answer']}  {tick}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=REPORT_PATH)
    for model_id, default_path in DEFAULT_PATHS.items():
        slug = model_id.replace(".", "_").replace("-", "_")
        parser.add_argument(f"--{slug}", default=default_path,
                            dest=slug, metavar="PATH")
    args = parser.parse_args()

    # Load whichever files exist
    model_records = {}
    for model_id, default_path in DEFAULT_PATHS.items():
        slug = model_id.replace(".", "_").replace("-", "_")
        path = getattr(args, slug)
        if os.path.exists(path):
            model_records[model_id] = load_jsonl(path)
            print(f"Loaded {len(model_records[model_id])} records for {model_id}")
        else:
            print(f"Skipping {model_id} (file not found: {path})")

    if len(model_records) < 2:
        print("Need at least 2 model result files to compare.")
        return

    model_data = {m: evaluate(recs) for m, recs in model_records.items()}
    task_types = sorted(set(tt for by_task in model_data.values() for tt in by_task))

    print("\n=== PER-TASK ACCURACY ===")
    per_task, totals = print_accuracy_table(task_types, model_data)

    print("\n=== QUALITY METRICS ===")
    quality = print_metrics_table(model_records)

    print_sample_traces(model_records, n_samples=3)

    # Recommendation — rank by match_rate, break ties by parse_rate
    ranked = sorted(quality.items(), key=lambda x: (x[1]["match_rate"], x[1]["parse_rate"]), reverse=True)
    winner, winner_metrics = ranked[0]
    second, second_metrics = ranked[1]
    reason = (f"highest answer match rate "
              f"({winner_metrics['match_rate']:.1%} vs {second_metrics['match_rate']:.1%} for {second})")
    recommendation = f"RECOMMENDATION: Use {winner} for full generation. Reason: {reason}."
    print(f"\n{recommendation}")

    report = {
        "per_task_accuracy": per_task,
        "totals": {m: {"correct": v[0], "total": v[1]} for m, v in totals.items()},
        "quality_metrics": quality,
        "recommendation": {"winner": winner, "reason": reason},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
