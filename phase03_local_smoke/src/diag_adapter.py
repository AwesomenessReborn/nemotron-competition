"""
Adapter sanity-check diagnostic. Loads one model at a time to stay within VRAM.
"""

import json, re, sys, gc, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import SYSTEM_PROMPT, format_for_training

BASE_MODEL   = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
ADAPTER_PATH = "phase03_local_smoke/outputs/adapters/local_4b/final_adapter"
TRAIN_JSONL  = "phase02_data_generation/data/merged/train.jsonl"
VAL_JSONL    = "phase02_data_generation/data/merged/val.jsonl"
MAX_NEW      = 300

BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


# ---------------------------------------------------------------------------
# Cache / generation helpers
# ---------------------------------------------------------------------------

def _patch_hybrid_cache_class():
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
                self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
                return self.ssm_states[layer_idx]
            cls.update_conv_state = _upd_conv
            cls.update_ssm_state  = _upd_ssm
            return True
    return False


def _patch_block_forward():
    """
    NemotronHBlock.forward never passes cache_params to attention (only to Mamba).
    This means attention KV cache is never populated/used → single-token steps have no context.
    Fix: pass cache_params as past_key_value to the attention mixer.
    """
    import torch
    for mod_name, mod in sys.modules.items():
        if "nemotron_h" not in mod_name.lower():
            continue
        block_cls = getattr(mod, "NemotronHBlock", None)
        if block_cls is None or not isinstance(block_cls, type):
            continue

        def patched_forward(self, hidden_states, cache_params=None,
                            cache_position=None, attention_mask=None):
            with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
                residual = hidden_states
                hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
                if self.block_type == "mamba":
                    hidden_states = self.mixer(
                        hidden_states, cache_params=cache_params, cache_position=cache_position)
                elif self.block_type == "attention":
                    hidden_states = self.mixer(
                        hidden_states, past_key_value=cache_params, cache_position=cache_position)
                    hidden_states = hidden_states[0]
                elif self.block_type == "mlp":
                    hidden_states = self.mixer(hidden_states)
                else:
                    raise ValueError(f"Invalid block_type: {self.block_type}")
                hidden_states = residual + hidden_states
                return hidden_states

        block_cls.forward = patched_forward
        return True
    return False


def _init_cache(model):
    for mod_name, mod in sys.modules.items():
        if "nemotron_h" in mod_name.lower():
            cls = getattr(mod, "HybridMambaAttentionDynamicCache", None)
            if cls is not None and isinstance(cls, type):
                dev   = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                c     = cls(model.config, 1, dtype, device=dev)
                c.conv_kernel_size = model.config.conv_kernel
                return c
    raise RuntimeError("HybridMambaAttentionDynamicCache not found")


def generate(model, tokenizer, input_ids, max_new=MAX_NEW):
    cache     = _init_cache(model)
    generated = input_ids
    eos_id    = tokenizer.eos_token_id

    for step in range(max_new):
        if step == 0:
            cur = generated
            pos = torch.arange(generated.shape[1], device=generated.device)
        else:
            cur = generated[:, -1:]
            pos = torch.tensor([generated.shape[1] - 1], device=generated.device)

        with torch.no_grad():
            out = model(input_ids=cur, cache_params=cache,
                        cache_position=pos, use_cache=True, return_dict=True)

        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat([generated, nxt], dim=1)
        if eos_id is not None and (nxt == eos_id).all():
            break

    return generated


def build_prompt(tokenizer, row):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["prompt"].strip()}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def load_rows(path, n):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def eval_rows(model, tokenizer, rows, label):
    sep = "=" * 68
    print(f"\n{sep}\n  {label}\n{sep}")
    results = []
    for row in rows:
        prompt = build_prompt(tokenizer, row)
        ids    = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out    = generate(model, tokenizer, ids)
        gen    = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        boxed = BOX_RE.findall(gen)
        ext   = boxed[-1].strip() if boxed else ""
        gold  = str(row.get("gold_answer", row.get("answer", ""))).strip()
        ok    = ext.strip().lower() == gold.strip().lower() if ext else False

        print(f"\n  id={row.get('id','?')[:8]}  task={row.get('task_type','?')}")
        print(f"  gold:        {gold!r}")
        print(f"  prompt tail: ...{prompt[-70:]!r}")
        print(f"  raw_gen:     {gen[:300]!r}")
        print(f"  extracted:   {ext!r}   boxed={bool(boxed)}   correct={ok}")
        results.append({"id": row.get("id"), "task": row.get("task_type"),
                        "gold": gold, "extracted": ext,
                        "boxed": bool(boxed), "correct": ok, "raw": gen[:300]})
    return results


def free(model):
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  [mem] VRAM free after release: {free_gb:.2f} GB")


# ---------------------------------------------------------------------------
# Section 5 — merged vs unmerged comparison (single load, merge in-place)
# ---------------------------------------------------------------------------

def section5_merged_vs_unmerged(tokenizer, rows):
    from unsloth import FastLanguageModel
    sep = "=" * 68
    print(f"\n{sep}\n  SECTION 5 — MERGED vs UNMERGED LoRA\n{sep}")

    # Use only 1 row to limit VRAM pressure from two sequential loads
    row = rows[0]
    prompt = build_prompt(tokenizer, row)
    ids    = tokenizer(prompt, return_tensors="pt").input_ids

    # Unmerged
    print("  Loading unmerged...")
    model_um, _ = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=2048,
        load_in_4bit=False, dtype=None, trust_remote_code=True)
    model_um.eval()
    _patch_hybrid_cache_class()
    _patch_block_forward()
    out_u = generate(model_um, tokenizer, ids.to(model_um.device), max_new=80)
    gen_u = tokenizer.decode(out_u[0][ids.shape[1]:], skip_special_tokens=True)
    free(model_um)

    # Merged — load fresh, merge on same VRAM budget now freed
    print("  Loading merged...")
    model_m, _ = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=2048,
        load_in_4bit=False, dtype=None, trust_remote_code=True)
    try:
        model_m = model_m.merge_and_unload()
        merge_ok = True
    except torch.cuda.OutOfMemoryError:
        print("  OOM on merge — skipping merged generation")
        merge_ok = False
        gen_m = "OOM"
    if merge_ok:
        model_m.eval()
        _patch_hybrid_cache_class()
    _patch_block_forward()
        out_m = generate(model_m, tokenizer, ids.to(model_m.device), max_new=80)
        gen_m = tokenizer.decode(out_m[0][ids.shape[1]:], skip_special_tokens=True)
    free(model_m)

    print(f"\n  id={row.get('id','?')[:8]}  gold={row.get('answer')!r}")
    print(f"  UNMERGED: {gen_u[:200]!r}")
    print(f"  MERGED  : {gen_m[:200]!r}")
    print(f"  Match   : {gen_u[:200] == gen_m[:200]}")


# ---------------------------------------------------------------------------
# Section 6 — logit comparison (base vs LoRA)
# ---------------------------------------------------------------------------

def section6_logits(tokenizer, row):
    from unsloth import FastLanguageModel
    sep = "=" * 68
    print(f"\n{sep}\n  SECTION 6 — LOGIT COMPARISON\n{sep}")

    prompt = build_prompt(tokenizer, row)
    ids    = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"  Prompt tokens: {ids.shape[1]}  task: {row.get('task_type')}  gold: {row.get('answer')!r}")

    def get_logits(model):
        cache   = _init_cache(model)
        ids_dev = ids.to(model.device)
        pos     = torch.arange(ids_dev.shape[1], device=model.device)
        with torch.no_grad():
            out = model(input_ids=ids_dev, cache_params=cache,
                        cache_position=pos, use_cache=True, return_dict=True)
        return out.logits[0, -1, :].float().cpu()

    # Base
    base_m, base_tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048,
        load_in_4bit=False, dtype=None, trust_remote_code=True)
    base_m.eval()
    _patch_hybrid_cache_class()
    _patch_block_forward()
    base_logits = get_logits(base_m)
    free(base_m)

    # LoRA unmerged — avoids merge peak allocation (~16 GB) on 15.4 GB GPU
    lora_m, _ = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=2048,
        load_in_4bit=False, dtype=None, trust_remote_code=True)
    lora_m.eval()
    _patch_hybrid_cache_class()
    _patch_block_forward()
    lora_logits = get_logits(lora_m)
    free(lora_m)

    diff = (lora_logits - base_logits).abs()
    print(f"\n  Max  |lora - base| logit: {diff.max().item():.4f}")
    print(f"  Mean |lora - base| logit: {diff.mean().item():.6f}")

    def top10(logits, label):
        probs = torch.softmax(logits, dim=0)
        top   = probs.topk(10)
        print(f"\n  Top-10 {label}:")
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"    {i:7d}  {tokenizer.decode([i])!r:25s}  p={p:.4f}")

    top10(base_logits, "BASE")
    top10(lora_logits, "LoRA (unmerged)")


# ---------------------------------------------------------------------------
# Section 7 — teacher-forced loss
# ---------------------------------------------------------------------------

def section7_teacher_loss(tokenizer, rows):
    from unsloth import FastLanguageModel
    sep = "=" * 68
    print(f"\n{sep}\n  SECTION 7 — TEACHER-FORCED ASSISTANT-ONLY LOSS\n{sep}")
    print(f"  {'id':<12} {'task':<22} {'base_loss':>10} {'lora_loss':>10} {'delta':>10}")
    print(f"  {'-'*68}")

    # Precompute all inputs/labels before loading models
    all_data = []
    for row in rows:
        parts   = format_for_training(row)
        prompt  = build_prompt(tokenizer, row)
        full_t  = tokenizer.apply_chat_template(
            [{"role": "system",    "content": parts["system"]},
             {"role": "user",      "content": parts["user"]},
             {"role": "assistant", "content": parts["assistant"]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prefix_len = tokenizer(prompt,  return_tensors="pt").input_ids.shape[1]
        full_ids   = tokenizer(full_t,  return_tensors="pt").input_ids
        labels     = full_ids.clone()
        labels[:, :prefix_len] = -100
        all_data.append((row, full_ids, labels))

    def get_losses(model, label):
        losses = []
        for row, full_ids, labels in all_data:
            with torch.no_grad():
                out = model(input_ids=full_ids.to(model.device),
                            labels=labels.to(model.device),
                            use_cache=False, return_dict=True)
            losses.append(out.loss.item())
        return losses

    base_m, _ = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048,
        load_in_4bit=False, dtype=None, trust_remote_code=True)
    base_m.eval()
    base_losses = get_losses(base_m, "base")
    free(base_m)

    lora_m, _ = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=2048,
        load_in_4bit=False, dtype=None, trust_remote_code=True)
    lora_m.eval()
    lora_losses = get_losses(lora_m, "lora")
    free(lora_m)

    for (row, _, _), bl, ll in zip(all_data, base_losses, lora_losses):
        print(f"  {str(row.get('id',''))[:12]:<12} {row.get('task_type',''):<22} "
              f"{bl:>10.4f} {ll:>10.4f} {ll-bl:>+10.4f}")

    avg_base = sum(base_losses) / len(base_losses)
    avg_lora = sum(lora_losses) / len(lora_losses)
    print(f"\n  {'AVERAGE':<35} {avg_base:>10.4f} {avg_lora:>10.4f} {avg_lora-avg_base:>+10.4f}")


# ---------------------------------------------------------------------------
# Main — one model in VRAM at a time
# ---------------------------------------------------------------------------

def main():
    from unsloth import FastLanguageModel

    train_rows = load_rows(TRAIN_JSONL, 5)
    val_rows   = load_rows(VAL_JSONL,   5)

    # Tokenizer — load once from adapter (used during training)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

    # ── Section 3: LoRA (unmerged PEFT) on training rows ────────────────────
    # Use unmerged to avoid OOM from merge op on 15 GB GPU.
    # Behaviorally identical to merged for generation purposes.
    print("\n[SEC 3] Loading LoRA (unmerged) for training rows...")
    m, _ = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=2048,
        load_in_4bit=False, dtype=None, trust_remote_code=True)
    m.eval()
    _patch_hybrid_cache_class()
    _patch_block_forward()
    train_results = eval_rows(m, tokenizer, train_rows, "LoRA (unmerged) — 5 TRAINING ROWS")
    free(m)

    # ── Section 4: LoRA (unmerged) on validation rows ────────────────────────
    print("\n[SEC 4] Loading LoRA (unmerged) for validation rows...")
    m, _ = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=2048,
        load_in_4bit=False, dtype=None, trust_remote_code=True)
    m.eval()
    _patch_hybrid_cache_class()
    _patch_block_forward()
    val_results = eval_rows(m, tokenizer, val_rows, "LoRA (unmerged) — 5 VALIDATION ROWS")
    free(m)

    # ── Section 5: merged vs unmerged ───────────────────────────────────────
    section5_merged_vs_unmerged(tokenizer, train_rows)

    # ── Section 6: logit comparison ─────────────────────────────────────────
    section6_logits(tokenizer, train_rows[0])

    # ── Section 7: teacher-forced loss ──────────────────────────────────────
    section7_teacher_loss(tokenizer, train_rows)

    # ── Final summary ────────────────────────────────────────────────────────
    sep = "=" * 68
    print(f"\n{sep}\n  DIAGNOSTIC COMPLETE\n{sep}")
    n_box_train = sum(1 for r in train_results if r["boxed"])
    n_ok_train  = sum(1 for r in train_results if r["correct"])
    n_box_val   = sum(1 for r in val_results if r["boxed"])
    n_ok_val    = sum(1 for r in val_results if r["correct"])
    print(f"\n  Train rows: boxed={n_box_train}/5  correct={n_ok_train}/5")
    print(f"  Val rows:   boxed={n_box_val}/5    correct={n_ok_val}/5")


if __name__ == "__main__":
    main()
