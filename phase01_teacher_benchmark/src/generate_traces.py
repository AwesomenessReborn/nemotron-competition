import argparse
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv

SYSTEM_PROMPT = """You are given a problem and its correct answer.
Write a concise explanation of WHY the answer is correct.

You MUST respond in this exact format:

REASONING: <concise explanation, 2-5 sentences max>
ANSWER: <copy the given answer exactly>

Rules:
- Keep reasoning SHORT and direct — explain the pattern/rule used, not every step
- Do not re-derive or verify the answer — just explain it
- Copy the answer field exactly as given, no changes
"""

OPENAI_MODELS = {"qwen3.5-plus", "deepseek-v4-pro"}
GEMINI_MODELS = {"gemini-2.5-flash"}
ANTHROPIC_MODELS = {"claude-haiku-4-5"}

MODEL_SLUGS = {
    "qwen3.5-plus": "qwen3.5plus",
    "deepseek-v4-pro": "deepseek_v4pro",
    "gemini-2.5-flash": "gemini_2.5_flash",
    "claude-haiku-4-5": "haiku",
}


def parse_response(text):
    reasoning_match = re.search(r"REASONING:\s*(.*?)(?=ANSWER:|$)", text, re.DOTALL)
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text)
    if reasoning_match and answer_match:
        return reasoning_match.group(1).strip(), answer_match.group(1).strip(), True
    return text.strip(), "PARSE_ERROR", False


def _user_content(row):
    return (
        f"Problem:\n{row['prompt']}\n\n"
        f"Correct answer: {row['answer']}\n\n"
        f"Explain concisely why this is correct."
    )


def generate_trace_openai(client, model_id, row):
    for attempt in range(3):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_content(row)},
                ],
                max_tokens=600,
                temperature=0.0,
            )
            raw = response.choices[0].message.content
            gen_time = time.time() - t0
            reasoning, answer, parse_ok = parse_response(raw)
            return {
                "id": row["id"],
                "prompt": row["prompt"],
                "reasoning": reasoning,
                "answer": answer,
                "task_type": row["task_type"],
                "gold_answer": str(row["answer"]),
                "source": f"teacher_test_{model_id}.jsonl",
                "model": model_id,
                "gen_time": round(gen_time, 2),
                "parse_success": parse_ok,
                "raw_response": raw,
            }
        except Exception as e:
            if "429" in str(e):
                print(f"  Rate limited, waiting 30s...")
                time.sleep(30)
            else:
                print(f"  Error attempt {attempt + 1}: {e}")
                time.sleep(5)

    return _error_record(row, model_id)


def generate_trace_gemini(gemini_model, model_id, row):
    from google.api_core.exceptions import ResourceExhausted
    for attempt in range(3):
        try:
            t0 = time.time()
            response = gemini_model.generate_content(_user_content(row))
            raw = response.text
            gen_time = time.time() - t0
            reasoning, answer, parse_ok = parse_response(raw)
            return {
                "id": row["id"],
                "prompt": row["prompt"],
                "reasoning": reasoning,
                "answer": answer,
                "task_type": row["task_type"],
                "gold_answer": str(row["answer"]),
                "source": f"teacher_test_{model_id}.jsonl",
                "model": model_id,
                "gen_time": round(gen_time, 2),
                "parse_success": parse_ok,
                "raw_response": raw,
            }
        except ResourceExhausted:
            print(f"  Rate limited, waiting 60s...")
            time.sleep(60)
        except Exception as e:
            print(f"  Error attempt {attempt + 1}: {e}")
            time.sleep(10)

    return _error_record(row, model_id)


def generate_trace_anthropic(client, model_id, row):
    for attempt in range(3):
        try:
            t0 = time.time()
            response = client.messages.create(
                model=model_id,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _user_content(row)}],
            )
            raw = response.content[0].text
            gen_time = time.time() - t0
            reasoning, answer, parse_ok = parse_response(raw)
            return {
                "id": row["id"],
                "prompt": row["prompt"],
                "reasoning": reasoning,
                "answer": answer,
                "task_type": row["task_type"],
                "gold_answer": str(row["answer"]),
                "source": f"teacher_test_{model_id}.jsonl",
                "model": model_id,
                "gen_time": round(gen_time, 2),
                "parse_success": parse_ok,
                "raw_response": raw,
            }
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                print(f"  Rate limited, waiting 30s...")
                time.sleep(30)
            else:
                print(f"  Error attempt {attempt + 1}: {e}")
                time.sleep(5)

    return _error_record(row, model_id)


def _error_record(row, model_id):
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "reasoning": "GENERATION_FAILED",
        "answer": "ERROR",
        "task_type": row["task_type"],
        "gold_answer": str(row["answer"]),
        "source": f"teacher_test_{model_id}.jsonl",
        "model": model_id,
        "gen_time": -1,
        "parse_success": False,
        "raw_response": "",
    }


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


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_SLUGS.keys()),
    )
    parser.add_argument(
        "--input",
        default="phase01_teacher_benchmark/data/splits/teacher_test_50.csv",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    slug = MODEL_SLUGS[args.model]
    out_path = (
        args.output
        or f"phase01_teacher_benchmark/data/generated/teacher_test_{slug}.jsonl"
    )

    # Build the appropriate runner
    if args.model in OPENAI_MODELS:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["OPENCODE_API_KEY"],
            base_url="https://opencode.ai/zen/go/v1",
        )
        def run(row):
            return generate_trace_openai(client, args.model, row)

    elif args.model in GEMINI_MODELS:
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("ERROR: GEMINI_API_KEY not set in environment / .env")
        from google import genai
        from google.genai import types
        g_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        gemini_cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=600,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        # Wrap into a simple callable that mimics GenerativeModel.generate_content
        class _GeminiModel:
            def generate_content(self, prompt):
                return g_client.models.generate_content(
                    model=args.model,
                    contents=prompt,
                    config=gemini_cfg,
                )
        gemini_model = _GeminiModel()
        def run(row):
            return generate_trace_gemini(gemini_model, args.model, row)

    elif args.model in ANTHROPIC_MODELS:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ERROR: ANTHROPIC_API_KEY not set in environment / .env")
        import anthropic
        a_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        def run(row):
            return generate_trace_anthropic(a_client, args.model, row)

    df = pd.read_csv(args.input)
    total = len(df)
    print(f"Generating {total} traces with {args.model} → {out_path}\n")

    with open(out_path, "w") as f:
        for i, (_, row) in enumerate(df.iterrows(), 1):
            result = run(row)
            correct = answer_correct(result["answer"], result["gold_answer"], row["task_type"])
            status = "OK" if result["parse_success"] else "PARSE_ERR"
            tick = "✓" if correct else "✗"
            print(
                f"[{i:02d}/{total}] {row['task_type']:<18} | {args.model} | "
                f"{status} | {result['gen_time']}s | "
                f"gold={result['gold_answer']} pred={result['answer']} {tick}"
            )
            f.write(json.dumps(result) + "\n")
            f.flush()

    print(f"\nDone. Results saved to {out_path}")


if __name__ == "__main__":
    main()
