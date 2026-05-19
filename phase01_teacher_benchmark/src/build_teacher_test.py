import pandas as pd
import sys

CSV_PATH = "shared/data/raw/train_with_task_type.csv"
OUT_PATH = "phase01_teacher_benchmark/data/splits/teacher_test_50.csv"
TARGET_N = 50
SEED = 42


def main():
    df = pd.read_csv(CSV_PATH)
    print("Columns:", df.columns.tolist())

    labeled = df[df["task_type"].notna() & (df["task_type"].str.strip() != "")]
    print(f"Labeled rows: {len(labeled)} / {len(df)} total")

    task_types = labeled["task_type"].unique()
    n_types = len(task_types)
    per_type = TARGET_N // n_types      # 7
    extras = TARGET_N % n_types         # 1

    groups = []
    for tt in sorted(task_types):
        g = labeled[labeled["task_type"] == tt].sample(
            n=min(per_type + 1, len(labeled[labeled["task_type"] == tt])),
            random_state=SEED,
        )
        groups.append(g)

    combined = pd.concat(groups).sample(frac=1, random_state=SEED)

    # Trim exactly to TARGET_N: keep all per_type samples then add extras
    # from the largest groups to hit exactly 50
    trimmed_groups = []
    for i, tt in enumerate(sorted(task_types)):
        g = combined[combined["task_type"] == tt]
        quota = per_type + (1 if i < extras else 0)
        trimmed_groups.append(g.head(quota))

    result = pd.concat(trimmed_groups).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\nSample distribution (total={len(result)}):")
    print(f"{'task_type':<25} {'count':>5}")
    print("-" * 32)
    for tt, count in result["task_type"].value_counts().sort_index().items():
        print(f"{tt:<25} {count:>5}")

    result.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(result)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
