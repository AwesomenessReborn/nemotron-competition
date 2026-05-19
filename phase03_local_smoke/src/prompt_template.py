SYSTEM_PROMPT = (
    "You are a precise reasoning model. "
    "Think through the problem step by step, "
    "then provide your final answer inside \\boxed{}."
)


def format_for_training(row: dict) -> dict:
    """
    Converts a merged JSONL row into system/user/assistant dict.
    The assistant turn contains the reasoning trace + boxed final answer.
    """
    reasoning = row.get("reasoning", "").strip()
    answer = str(row["answer"]).strip()
    assistant_text = f"{reasoning}\n\nFinal answer: \\boxed{{{answer}}}"
    return {
        "system": SYSTEM_PROMPT,
        "user": row["prompt"].strip(),
        "assistant": assistant_text,
    }
