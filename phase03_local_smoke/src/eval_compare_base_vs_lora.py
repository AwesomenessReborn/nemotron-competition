"""
Base model vs LoRA adapter answer-accuracy benchmark.

Runs deterministic inference on the validation set for:
  - base Nemotron model (no adapter)
  - current LoRA adapter (merged into base weights)

Produces per-sample JSONL predictions, per-model JSON summaries,
a side-by-side comparison JSONL, and a CSV accuracy table.

Usage:
    python eval_compare_base_vs_lora.py \\
      --base-model  nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \\
      --adapter-path phase03_local_smoke/outputs/adapters/local_4b/final_adapter \\
      --val-path    phase02_data_generation/data/merged/val.jsonl \\
      --output-dir  phase03_local_smoke/outputs/evals \\
      --max-new-tokens 512 --temperature 0 [--limit 10] [--run base|lora|both]
"""

import argparse
import csv
import functools
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import SYSTEM_PROMPT

BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

NUMERIC_TASKS = {"gravity", "unit_conversion"}
ROMAN_TASKS   = {"roman"}


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> tuple[str, bool]:
    """Return (extracted_answer, parse_success). parse_success=True iff \\boxed{} found."""
    boxed = BOX_RE.findall(text)
    if boxed:
        return boxed[-1].strip(), True
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1], False
    return text.strip().splitlines()[-1].strip() if text.strip() else "", False


# ---------------------------------------------------------------------------
# Standardised scoring
# ---------------------------------------------------------------------------

def score(pred: str, gold: str, task_type: str) -> tuple[bool, str]:
    """Return (correct, scoring_method)."""
    pred = str(pred).strip()
    gold = str(gold).strip()

    if task_type in ROMAN_TASKS:
        return pred.upper() == gold.upper(), "case_insensitive"

    if task_type in NUMERIC_TASKS:
        try:
            p, g = float(pred), float(gold)
            return abs(p - g) / max(abs(g), 1) < 0.01, "relative_1pct"
        except ValueError:
            return pred.lower() == gold.lower(), "normalized_exact"

    # cipher_text, bit_manipulation, symbol_transform, unknown
    norm = lambda s: " ".join(s.lower().split())
    return norm(pred) == norm(gold), "normalized_exact"


# ---------------------------------------------------------------------------
# Architecture patch (required for transformers 5.5.0 + HybridMambaAttentionDynamicCache)
# ---------------------------------------------------------------------------

def _patch_hybrid_cache_class():
    """
    Fix two bugs in HybridMambaAttentionDynamicCache that were copied verbatim
    from modeling_mamba2.py where conv_states/ssm_states are stacked tensors.
    Here they are Python lists, so .device doesn't exist.

    Bug A: update_conv_state / update_ssm_state call self.conv_states.device
           (list has no .device attribute).
    Bug B: __init__ computes conv_kernel_size locally but never stores it on
           self, so cuda_kernels_forward crashes on cache_params.conv_kernel_size.
    """
    import sys
    for mod_name, mod in sys.modules.items():
        if "nemotron_h" in mod_name.lower():
            cls = getattr(mod, "HybridMambaAttentionDynamicCache", None)
            if cls is None:
                continue

            def _upd_conv(self, layer_idx, new_conv_state, cache_init=False):
                dev = self.conv_states[layer_idx].device
                if cache_init:
                    self.conv_states[layer_idx] = new_conv_state.to(dev)
                else:
                    self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
                    self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(dev)
                return self.conv_states[layer_idx]

            def _upd_ssm(self, layer_idx, new_ssm_state):
                dev = self.ssm_states[layer_idx].device
                self.ssm_states[layer_idx] = new_ssm_state.to(dev)
                return self.ssm_states[layer_idx]

            cls.update_conv_state = _upd_conv
            cls.update_ssm_state  = _upd_ssm
            return True
    return False


def patch_model(model):
    """
    Apply fixes for HybridMambaAttentionDynamicCache bugs (see _patch_hybrid_cache_class).
    The actual generation is handled by nemotron_generate() which bypasses
    model.generate() entirely to avoid past_key_values / cache_params mismatch.
    """
    _patch_hybrid_cache_class()
    return model


def nemotron_generate(model, tokenizer, input_ids, max_new_tokens: int) -> torch.Tensor:
    """
    Minimal greedy generation loop for NemotronH that calls model.forward()
    directly with cache_params, bypassing model.generate() and the broken
    past_key_values / cache_params key-name mismatch in transformers 5.5.0.

    Returns the full sequence tensor (prompt + generated tokens).
    """
    import sys

    # Locate HybridMambaAttentionDynamicCache (loaded via trust_remote_code)
    HybridCache = None
    for mod_name, mod in sys.modules.items():
        if "nemotron_h" in mod_name.lower():
            HybridCache = getattr(mod, "HybridMambaAttentionDynamicCache", None)
            if HybridCache is not None:
                break
    if HybridCache is None:
        raise RuntimeError("Could not find HybridMambaAttentionDynamicCache in loaded modules")

    device    = input_ids.device
    dtype     = next(model.parameters()).dtype
    batch     = input_ids.shape[0]
    input_len = input_ids.shape[1]

    # Initialise cache and inject missing conv_kernel_size attribute
    cache = HybridCache(model.config, batch, dtype, device=device)
    cache.conv_kernel_size = model.config.conv_kernel

    eos_id   = tokenizer.eos_token_id
    generated = input_ids

    for step in range(max_new_tokens):
        if step == 0:
            cur_input = generated
            cache_pos = torch.arange(input_len, device=device)
        else:
            cur_input = generated[:, -1:]
            cache_pos = torch.tensor([generated.shape[1] - 1], device=device)

        with torch.no_grad():
            outputs = model(
                input_ids   = cur_input,
                cache_params = cache,
                cache_position = cache_pos,
                use_cache   = True,
                return_dict = True,
            )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated  = torch.cat([generated, next_token], dim=1)

        if eos_id is not None and (next_token == eos_id).all():
            break

    return generated


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base(base_model: str):
    from unsloth import FastLanguageModel
    print(f"\n[*] Loading BASE model: {base_model}")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
    )
    model = patch_model(model)
    model.eval()
    print(f"[*] Base model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def load_lora(adapter_path: str):
    from unsloth import FastLanguageModel
    print(f"\n[*] Loading LORA adapter: {adapter_path}")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
    )
    print("[*] Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model = patch_model(model)
    model.eval()
    print(f"[*] LoRA model loaded + merged in {time.time() - t0:.1f}s")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, rows, args, variant: str, smoke: bool) -> list[dict]:
    results = []
    total = len(rows)
    by_task: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "parse_ok": 0})

    for i, row in enumerate(rows, 1):
        gold = str(row.get("gold_answer", row.get("answer", ""))).strip()
        task = row.get("task_type", "unknown")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["prompt"].strip()},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        prompt_len = input_ids.shape[1]

        try:
            output_ids = nemotron_generate(model, tokenizer, input_ids, args.max_new_tokens)
            gen_ids = output_ids[0][prompt_len:]
            raw_gen = tokenizer.decode(gen_ids, skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i:03d}/{total}] OOM — skipping", flush=True)
            raw_gen = ""
        finally:
            torch.cuda.empty_cache()

        extracted, parse_ok = extract_answer(raw_gen)
        correct, method = score(extracted, gold, task)

        by_task[task]["total"] += 1
        by_task[task]["correct"] += int(correct)
        by_task[task]["parse_ok"] += int(parse_ok)

        tick = "✓" if correct else "✗"
        box  = "📦" if parse_ok else "  "
        print(
            f"[{i:03d}/{total}] {task:<22} {tick} {box} "
            f"gold={gold!r:>20}  pred={extracted!r}",
            flush=True,
        )

        if smoke:
            print(f"  PROMPT (last 200 chars): ...{prompt_text[-200:]}")
            print(f"  RAW GEN: {raw_gen[:400]}")
            print()

        results.append({
            "id":               row.get("id"),
            "task_type":        task,
            "prompt":           row["prompt"],
            "raw_generation":   raw_gen,
            "extracted_answer": extracted,
            "gold_answer":      gold,
            "correct":          correct,
            "parse_success":    parse_ok,
            "scoring_method":   method,
            "model_variant":    variant,
        })

    n_correct = sum(r["correct"] for r in results)
    n_parse   = sum(r["parse_success"] for r in results)
    print(f"\n[{variant}] accuracy={n_correct}/{total} ({n_correct/total:.1%})  "
          f"parse={n_parse}/{total} ({n_parse/total:.1%})")
    for t in sorted(by_task):
        v = by_task[t]
        print(f"  {t:<22} {v['correct']}/{v['total']}  "
              f"({v['correct']/v['total']:.1%})")

    return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def build_summary(results: list[dict], variant: str) -> dict:
    n = len(results)
    by_task: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "parse_ok": 0})
    for r in results:
        t = r["task_type"]
        by_task[t]["total"]   += 1
        by_task[t]["correct"] += int(r["correct"])
        by_task[t]["parse_ok"] += int(r["parse_success"])

    return {
        "model_variant":  variant,
        "n":              n,
        "accuracy":       round(sum(r["correct"] for r in results) / n, 4),
        "parse_rate":     round(sum(r["parse_success"] for r in results) / n, 4),
        "by_task": {
            t: {
                "accuracy":   round(v["correct"] / v["total"], 4),
                "parse_rate": round(v["parse_ok"] / v["total"], 4),
                "n":          v["total"],
            }
            for t, v in sorted(by_task.items())
        },
    }


def build_comparison(base_results: list[dict], lora_results: list[dict]) -> list[dict]:
    base_by_id = {r["id"]: r for r in base_results}
    lora_by_id = {r["id"]: r for r in lora_results}
    ids = list(dict.fromkeys(r["id"] for r in base_results))

    rows = []
    for rid in ids:
        b = base_by_id.get(rid)
        l = lora_by_id.get(rid)
        if b is None or l is None:
            continue
        bc, lc = b["correct"], l["correct"]
        if not b["parse_success"] and not l["parse_success"]:
            cls = "parse_failure"
        elif bc and lc:
            cls = "both_correct"
        elif lc and not bc:
            cls = "lora_only_correct"
        elif bc and not lc:
            cls = "base_only_correct"
        else:
            cls = "both_wrong"
        rows.append({
            "id":           rid,
            "task_type":    b["task_type"],
            "gold_answer":  b["gold_answer"],
            "base_answer":  b["extracted_answer"],
            "lora_answer":  l["extracted_answer"],
            "base_correct": bc,
            "lora_correct": lc,
            "class":        cls,
        })
    return rows


def print_final_report(base_summary: dict, lora_summary: dict, comparison: list[dict]):
    classes = defaultdict(int)
    for r in comparison:
        classes[r["class"]] += 1

    b_acc = base_summary["accuracy"]
    l_acc = lora_summary["accuracy"]
    delta = l_acc - b_acc

    sign = "+" if delta >= 0 else ""
    print("\n" + "=" * 62)
    print("BENCHMARK SUMMARY")
    print("=" * 62)
    print(f"{'':25s}  {'BASE':>8}  {'HAIKU_LORA':>10}  {'DELTA':>8}")
    print(f"{'Overall accuracy':25s}  {b_acc:>8.1%}  {l_acc:>10.1%}  {sign}{delta:>7.1%}")
    print(f"{'Parse rate':25s}  {base_summary['parse_rate']:>8.1%}  "
          f"{lora_summary['parse_rate']:>10.1%}")
    print()
    print(f"{'Per task:':25s}  {'BASE':>8}  {'HAIKU_LORA':>10}  {'DELTA':>8}")
    all_tasks = sorted(set(base_summary["by_task"]) | set(lora_summary["by_task"]))
    for t in all_tasks:
        ba = base_summary["by_task"].get(t, {}).get("accuracy", 0)
        la = lora_summary["by_task"].get(t, {}).get("accuracy", 0)
        d  = la - ba
        s  = "+" if d >= 0 else ""
        print(f"  {t:<23}  {ba:>8.1%}  {la:>10.1%}  {s}{d:>7.1%}")

    print()
    print(f"  LoRA fixed base:   {classes['lora_only_correct']:>4}")
    print(f"  LoRA broke base:   {classes['base_only_correct']:>4}")
    print(f"  Both correct:      {classes['both_correct']:>4}")
    print(f"  Both wrong:        {classes['both_wrong']:>4}")
    print(f"  Parse failure:     {classes['parse_failure']:>4}")
    print("=" * 62)

    verdict = "BETTER" if l_acc > b_acc else ("SAME" if l_acc == b_acc else "WORSE")
    print(f"\nVERDICT: LoRA is {verdict} than base  ({sign}{delta:.1%})")

    improved = [t for t in all_tasks
                if lora_summary["by_task"].get(t, {}).get("accuracy", 0)
                 > base_summary["by_task"].get(t, {}).get("accuracy", 0)]
    degraded = [t for t in all_tasks
                if lora_summary["by_task"].get(t, {}).get("accuracy", 0)
                 < base_summary["by_task"].get(t, {}).get("accuracy", 0)]
    if improved:
        print(f"  Improved tasks:   {', '.join(improved)}")
    if degraded:
        print(f"  Degraded tasks:   {', '.join(degraded)}")
    print()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)} rows → {path}")


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote → {path}")


def save_comparison_csv(path: Path, base_summary: dict, lora_summary: dict):
    all_tasks = sorted(set(base_summary["by_task"]) | set(lora_summary["by_task"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_type", "base_accuracy", "lora_accuracy", "delta", "base_n"])
        for t in all_tasks:
            ba = base_summary["by_task"].get(t, {}).get("accuracy", 0)
            la = lora_summary["by_task"].get(t, {}).get("accuracy", 0)
            n  = base_summary["by_task"].get(t, {}).get("n", 0)
            w.writerow([t, f"{ba:.4f}", f"{la:.4f}", f"{la - ba:+.4f}", n])
        ba = base_summary["accuracy"]
        la = lora_summary["accuracy"]
        w.writerow(["OVERALL", f"{ba:.4f}", f"{la:.4f}", f"{la - ba:+.4f}",
                    base_summary["n"]])
    print(f"  wrote → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_val(val_path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(val_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Base vs LoRA answer-accuracy benchmark")
    p.add_argument("--base-model",    default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    p.add_argument("--adapter-path",  required=True)
    p.add_argument("--val-path",      required=True)
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--limit",          type=int, default=None,
                   help="Evaluate only first N rows (smoke test)")
    p.add_argument("--run", choices=["base", "lora", "both"], default="both")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    smoke = args.limit is not None

    rows = load_val(args.val_path, args.limit)
    print(f"\n[*] Val rows to evaluate: {len(rows)}"
          f"{'  (smoke test)' if smoke else ''}")
    print(f"[*] max_new_tokens={args.max_new_tokens}  temperature={args.temperature}")
    print(f"[*] Output dir: {out}\n")

    base_results, lora_results = [], []

    # ── BASE MODEL ──────────────────────────────────────────────────────────
    if args.run in ("base", "both"):
        model, tokenizer = load_base(args.base_model)
        base_results = run_inference(model, tokenizer, rows, args, "base", smoke)
        del model
        torch.cuda.empty_cache()

        save_jsonl(out / "base_val_predictions.jsonl", base_results)
        base_summary = build_summary(base_results, "base")
        save_json(out / "base_val_summary.json", base_summary)
        save_jsonl(out / "wrong_examples_base.jsonl",
                   [r for r in base_results if not r["correct"]])

    # ── LORA MODEL ──────────────────────────────────────────────────────────
    if args.run in ("lora", "both"):
        model, tokenizer = load_lora(args.adapter_path)
        lora_results = run_inference(model, tokenizer, rows, args, "haiku_lora", smoke)
        del model
        torch.cuda.empty_cache()

        save_jsonl(out / "haiku_lora_val_predictions.jsonl", lora_results)
        lora_summary = build_summary(lora_results, "haiku_lora")
        save_json(out / "haiku_lora_val_summary.json", lora_summary)
        save_jsonl(out / "wrong_examples_haiku_lora.jsonl",
                   [r for r in lora_results if not r["correct"]])

    # ── COMPARISON (only when both ran) ─────────────────────────────────────
    if args.run == "both" and base_results and lora_results:
        comparison = build_comparison(base_results, lora_results)
        save_jsonl(out / "comparison_all.jsonl", comparison)
        save_comparison_csv(out / "comparison_by_task.csv", base_summary, lora_summary)
        print_final_report(base_summary, lora_summary, comparison)

    # ── LOAD EXISTING if only one side ran this invocation ───────────────────
    elif args.run == "base" and (out / "haiku_lora_val_predictions.jsonl").exists():
        print("\n[*] LoRA results found on disk — generating comparison...")
        lora_results = [json.loads(l) for l in
                        open(out / "haiku_lora_val_predictions.jsonl") if l.strip()]
        lora_summary = json.load(open(out / "haiku_lora_val_summary.json"))
        comparison = build_comparison(base_results, lora_results)
        save_jsonl(out / "comparison_all.jsonl", comparison)
        save_comparison_csv(out / "comparison_by_task.csv", base_summary, lora_summary)
        print_final_report(base_summary, lora_summary, comparison)

    elif args.run == "lora" and (out / "base_val_predictions.jsonl").exists():
        print("\n[*] Base results found on disk — generating comparison...")
        base_results = [json.loads(l) for l in
                        open(out / "base_val_predictions.jsonl") if l.strip()]
        base_summary = json.load(open(out / "base_val_summary.json"))
        comparison = build_comparison(base_results, lora_results)
        save_jsonl(out / "comparison_all.jsonl", comparison)
        save_comparison_csv(out / "comparison_by_task.csv", base_summary, lora_summary)
        print_final_report(base_summary, lora_summary, comparison)

    print("\n[*] Done.")


if __name__ == "__main__":
    main()
