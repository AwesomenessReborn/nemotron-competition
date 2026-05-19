import json
import random
from collections import defaultdict
from pathlib import Path

V6_PATH = "/home/hareee234/Dev/kaggle/cmpe188-prof-teacher/CoderGym/Nemotron/train_reasoning_v6.jsonl"
V7_PATH = "phase02_data_generation/data/train_reasoning_v7_haiku.jsonl"

OUTPUT_DIR = Path("phase02_data_generation/data/merged")
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "val.jsonl"
SMOKE_PATH = OUTPUT_DIR / "smoke_50.jsonl"

SEED = 42
VAL_FRAC = 0.10
SMOKE_N = 50


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    print(f"Loading v6: {V6_PATH}")
    v6 = load_jsonl(V6_PATH)
    print(f"  {len(v6)} rows")

    print(f"Loading v7: {V7_PATH}")
    v7 = load_jsonl(V7_PATH)
    print(f"  {len(v7)} rows\n")

    # Index v7 by id for O(1) lookup
    v7_ids = {str(r["id"]) for r in v7}

    # Merge: keep v7 where overlap, add v6-only rows
    v6_only, overlap = [], 0
    for r in v6:
        if str(r["id"]) in v7_ids:
            overlap += 1
        else:
            v6_only.append(r)

    merged = v7 + v6_only

    print(f"Deduplication (prefer v7):")
    print(f"  v7 rows:       {len(v7)}")
    print(f"  v6 rows:       {len(v6)}")
    print(f"  Overlap (v7 wins): {overlap}")
    print(f"  v6-only added: {len(v6_only)}")
    print(f"  Total merged:  {len(merged)}\n")

    # Source breakdown
    source_v7 = len(v7)
    source_v6 = len(v6_only)
    print(f"Source breakdown:")
    print(f"  From v7 (haiku): {source_v7}")
    print(f"  From v6 (prof):  {source_v6}\n")

    # Per-task distribution
    by_task = defaultdict(int)
    for r in merged:
        by_task[r["task_type"]] += 1
    print("Per-task distribution:")
    for tt in sorted(by_task):
        print(f"  {tt:<22} {by_task[tt]}")
    print()

    # Shuffle and split
    rng = random.Random(SEED)
    rng.shuffle(merged)

    n_val = int(len(merged) * VAL_FRAC)
    val = merged[:n_val]
    train = merged[n_val:]

    print(f"Train/val split (90/10):")
    print(f"  train: {len(train)}")
    print(f"  val:   {len(val)}\n")

    # Smoke set: task-balanced sample from train
    by_task_train = defaultdict(list)
    for r in train:
        by_task_train[r["task_type"]].append(r)

    task_types = sorted(by_task_train.keys())
    per_task = SMOKE_N // len(task_types)
    remainder = SMOKE_N % len(task_types)

    smoke = []
    for i, tt in enumerate(task_types):
        n = per_task + (1 if i < remainder else 0)
        rows = by_task_train[tt]
        smoke.extend(rows[:n])

    print(f"Smoke set: {len(smoke)} samples ({per_task}–{per_task + (1 if remainder else 0)} per task type)")
    for tt in task_types:
        c = sum(1 for r in smoke if r["task_type"] == tt)
        print(f"  {tt:<22} {c}")
    print()

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path, rows in [(TRAIN_PATH, train), (VAL_PATH, val), (SMOKE_PATH, smoke)]:
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(rows):>5} rows → {path}")


if __name__ == "__main__":
    main()
