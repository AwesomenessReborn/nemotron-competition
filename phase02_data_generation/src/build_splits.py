import json, random, pathlib, argparse


def build_splits(input_path, out_dir, val_frac=0.1, smoke_n=20, seed=42):
    random.seed(seed)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    for i, row in enumerate(rows):
        assert "prompt" in row, f"Row {i} missing 'prompt'"
        assert "answer" in row, f"Row {i} missing 'answer'"
        assert "reasoning" in row, f"Row {i} missing 'reasoning'"

    random.shuffle(rows)
    val_n = max(1, int(len(rows) * val_frac))
    val, train = rows[:val_n], rows[val_n:]
    smoke = random.sample(train, min(smoke_n, len(train)))

    def write(path, data):
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(data)} rows -> {path}")

    write(out_dir / "train.jsonl", train)
    write(out_dir / "val.jsonl", val)
    write(out_dir / "smoke_20.jsonl", smoke)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="data/splits")
    args = parser.parse_args()
    build_splits(args.input, args.out_dir)
