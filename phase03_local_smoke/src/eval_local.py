"""
Evaluates a model (base or LoRA) via a running vLLM endpoint.
Outputs per-sample results and a per-task accuracy summary.
"""
import re, json, argparse, time
from openai import OpenAI
from pathlib import Path

# Handles nested braces one level deep — sufficient for this dataset
BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_answer(text: str) -> str:
    boxed = BOX_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip().splitlines()[-1].strip()


def score(pred: str, gold: str, task_type: str) -> bool:
    pred, gold = pred.strip(), str(gold).strip()
    if task_type == "roman":
        return pred.upper() == gold.upper()
    try:
        return abs(float(pred) - float(gold)) < 0.05 * max(abs(float(gold)), 1)
    except ValueError:
        return pred == gold


def main(endpoint, model_name, val_path, out_path):
    client = OpenAI(base_url=endpoint, api_key="unused")
    rows = [json.loads(l) for l in open(val_path) if l.strip()]

    results = []
    correct = 0

    for i, row in enumerate(rows):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": row["prompt"]}],
            max_tokens=512,
            temperature=0,
        )
        latency = time.time() - t0
        output = resp.choices[0].message.content or ""
        pred = extract_answer(output)
        is_correct = score(pred, row["answer"], row.get("task_type", ""))
        correct += is_correct

        results.append({
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "gold": row["answer"],
            "pred": pred,
            "correct": is_correct,
            "latency": latency,
            "has_boxed": bool(BOX_RE.search(output)),
        })
        print(f"[{i+1}/{len(rows)}] {row.get('task_type','?'):20s} | {'✓' if is_correct else '✗'} | pred={pred!r}")

    acc = correct / len(rows)
    boxed_rate = sum(r["has_boxed"] for r in results) / len(results)

    summary = {
        "model": model_name,
        "accuracy": acc,
        "boxed_rate": boxed_rate,
        "n": len(rows),
        "by_task": {},
    }
    for task in sorted(set(r["task_type"] for r in results if r["task_type"])):
        task_rows = [r for r in results if r["task_type"] == task]
        summary["by_task"][task] = round(sum(r["correct"] for r in task_rows) / len(task_rows), 4)

    print(f"\nAccuracy: {acc:.3f} | Boxed rate: {boxed_rate:.3f}")
    print("By task:", summary["by_task"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.endpoint, args.model, args.val, args.out)
