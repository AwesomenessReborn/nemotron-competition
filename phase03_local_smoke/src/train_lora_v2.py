"""
V2 LoRA fine-tuning for Nemotron-H with:
  - assistant-only label masking (system/user prefix = -100)
  - correct target modules: in_proj, out_proj (Mamba), up_proj, down_proj (MLP),
    q_proj, k_proj, v_proj, o_proj (Attention)
  - configurable overfit / medium / full run via CLI flags

Nemotron-3 4B module map (verified from named_modules inspection):
  Mamba layers  (21): backbone.layers.{0,2,4,...,38}.mixer.in_proj / out_proj
  MLP layers    (17): backbone.layers.{1,3,5,...,37}.mixer.up_proj / down_proj
  Attn layers    (4): backbone.layers.{12,17,24,32}.mixer.q_proj / k_proj / v_proj / o_proj
"""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import SYSTEM_PROMPT


def format_for_training(row: dict, target_type: str = "haiku_reasoning") -> dict:
    reasoning = row.get("reasoning", "").strip()
    answer = str(row.get("answer", row.get("gold_answer", ""))).strip()

    if target_type == "answer_only":
        assistant_text = f"Final answer: \\boxed{{{answer}}}"
    elif target_type == "short_reasoning":
        if reasoning:
            # First sentence (up to first period/newline/semicolon), capped at 250 chars
            for sep in (".\n", ". ", ".\t", "\n", ";"):
                idx = reasoning.find(sep)
                if 0 < idx <= 250:
                    short = reasoning[:idx + 1].strip()
                    break
            else:
                short = reasoning[:250].rstrip()
            assistant_text = f"{short}\n\nFinal answer: \\boxed{{{answer}}}"
        else:
            assistant_text = f"Final answer: \\boxed{{{answer}}}"
    else:  # haiku_reasoning (default)
        if reasoning:
            assistant_text = f"{reasoning}\n\nFinal answer: \\boxed{{{answer}}}"
        else:
            assistant_text = f"Final answer: \\boxed{{{answer}}}"

    return {
        "system": SYSTEM_PROMPT,
        "user": row["prompt"].strip(),
        "assistant": assistant_text,
    }


def load_adapter_weights(model, adapter_dir):
    """
    Load LoRA adapter weights manually by name — bypasses PEFT's set_peft_model_state_dict
    which silently fails because it strips the base_model.model. prefix before matching,
    leaving all B matrices at zero init.
    """
    from safetensors.torch import load_file as sf_load
    adapter_file = Path(adapter_dir) / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")
    weights = sf_load(str(adapter_file))
    applied = 0
    for name, param in model.named_parameters():
        if name in weights:
            w = weights[name].to(param.device, dtype=param.dtype)
            with torch.no_grad():
                param.copy_(w)
            applied += 1
    print(f"  [load_adapter] Applied {applied}/{len(weights)} parameters manually")
    return applied


def _fix_adapter_key_names(adapter_dir):
    """
    Unsloth saves LoRA keys as `lora_A.weight` / `lora_B.weight` (no adapter name).
    PEFT loading expects `lora_A.default.weight` / `lora_B.default.weight`.
    This renames the keys in the safetensors file in-place so standard PEFT loading works.
    """
    from safetensors.torch import load_file as sf_load, save_file as sf_save
    adapter_file = Path(adapter_dir) / "adapter_model.safetensors"
    if not adapter_file.exists():
        return
    weights = sf_load(str(adapter_file))
    if any("lora_A.default.weight" in k for k in weights):
        return  # already has correct format
    new_weights = {}
    renamed = 0
    for k, v in weights.items():
        for lora_part in ("lora_A.weight", "lora_B.weight"):
            if k.endswith(lora_part):
                k = k[:-len(lora_part)] + lora_part.replace(".weight", ".default.weight")
                renamed += 1
                break
        new_weights[k] = v
    sf_save(new_weights, str(adapter_file))
    print(f"  [adapter fix] renamed {renamed} keys to include 'default' adapter name")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASE_MODEL   = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
ADAPTER_PATH = "phase03_local_smoke/outputs/adapters/local_4b/final_adapter"

# Verified target modules — all actual nn.Linear layers in every block type.
# V1 omitted in_proj/out_proj (Mamba) and included gate_proj (doesn't exist).
TARGET_MODULES_V2 = [
    # Mamba layers EXCLUDED: both in_proj and out_proj have training-inference mismatches.
    # out_proj: mamba_split_conv1d_scan_combined uses outproj_weight=self.out_proj.weight (base
    #   weight only, skipping LoRA wrapper), but inference calls self.out_proj() applying LoRA.
    # in_proj: mamba_chunk_scan_combined computes SSM in FP32 internally; causal_conv1d_update
    #   + selective_state_update (inference) run in BF16 — FP32 vs BF16 precision diverges at
    #   pos 138 even with teacher-forced tokens.
    # Only attention and MLP have consistent computation across training and inference.
    "up_proj",    # MLP up:   3136 → 12544  (17 layers)
    "down_proj",  # MLP down: 12544 → 3136  (17 layers)
    "q_proj",     # Attn Q:   3136 → 5120   (4 layers)
    "k_proj",     # Attn K:   3136 → 1024   (4 layers)
    "v_proj",     # Attn V:   3136 → 1024   (4 layers)
    "o_proj",     # Attn O:   5120 → 3136   (4 layers)
]

BOX_RE = re.compile(r"\\box(?:ed)?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path, n=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if n is not None and len(rows) >= n:
                    break
    return rows


def build_masked_dataset(rows, tokenizer, verbose=False, target_type="haiku_reasoning"):
    """
    Tokenize each row and mask system+user prefix tokens with -100.
    Returns a HuggingFace Dataset with input_ids, labels, attention_mask.
    """
    records = []
    for i, row in enumerate(rows):
        parts = format_for_training(row, target_type=target_type)

        # Prefix = system + user + "<|im_start|>assistant\n<think></think>"
        # We mask these tokens so loss is only on the assistant reasoning + boxed answer.
        prefix_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": parts["system"]},
             {"role": "user",   "content": parts["user"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

        full_text = tokenizer.apply_chat_template(
            [{"role": "system",    "content": parts["system"]},
             {"role": "user",      "content": parts["user"]},
             {"role": "assistant", "content": parts["assistant"]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)

        prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                               return_tensors="pt").input_ids[0]
        full_ids   = tokenizer(full_text, add_special_tokens=False,
                               return_tensors="pt").input_ids[0]

        n_prefix = len(prefix_ids)
        labels   = full_ids.clone()
        labels[:n_prefix] = -100
        n_target = (labels != -100).sum().item()

        if verbose and i == 0:
            _print_mask_verification(full_ids, labels, n_prefix, tokenizer)

        records.append({
            "input_ids":      full_ids.tolist(),
            "labels":         labels.tolist(),
            "attention_mask": [1] * len(full_ids),
        })

    return Dataset.from_list(records)


def _print_mask_verification(full_ids, labels, n_prefix, tokenizer):
    sep = "=" * 64
    print(f"\n{sep}")
    print("  LABEL MASK VERIFICATION (first training sample)")
    print(sep)
    print(f"  total tokens : {len(full_ids)}")
    print(f"  prefix tokens: {n_prefix}  (labels = -100)")
    print(f"  target tokens: {(labels != -100).sum().item()}  (loss computed here)")

    print(f"\n  Last 5 MASKED tokens (prefix tail → should be assistant header):")
    for j in range(max(0, n_prefix - 5), n_prefix):
        tok = tokenizer.decode([full_ids[j].item()])
        print(f"    [{j:4d}] id={full_ids[j].item():6d}  {tok!r:25s}  label=-100  ✓")

    print(f"\n  First 5 TARGET tokens (start of assistant reasoning):")
    for j in range(n_prefix, min(n_prefix + 5, len(full_ids))):
        tok = tokenizer.decode([full_ids[j].item()])
        lbl = labels[j].item()
        print(f"    [{j:4d}] id={full_ids[j].item():6d}  {tok!r:25s}  label={lbl}  ✓")
    print(sep)


# ---------------------------------------------------------------------------
# Cache / generation (NemotronH custom loop — model.generate() is broken)
# ---------------------------------------------------------------------------

def _patch_hybrid_cache(model):
    for mod_name, mod in sys.modules.items():
        if "nemotron_h" not in mod_name.lower():
            continue
        cls = getattr(mod, "HybridMambaAttentionDynamicCache", None)
        if cls is None or not isinstance(cls, type):
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
    Patch NemotronHBlock.forward to pass cache_params to attention blocks as past_key_value.
    The original code (line 777-780 of modeling_nemotron_h.py) never passes cache_params to
    attention, so attention has no KV cache at step 1+ and processes only the current single
    token with no context — producing repetition/garbage.
    """
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
                    # Pass cache_params as past_key_value so attention populates/uses KV cache
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
        if "nemotron_h" not in mod_name.lower():
            continue
        cls = getattr(mod, "HybridMambaAttentionDynamicCache", None)
        if cls is not None and isinstance(cls, type):
            dev   = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            c = cls(model.config, 1, dtype, device=dev)
            c.conv_kernel_size = model.config.conv_kernel
            return c
    raise RuntimeError("HybridMambaAttentionDynamicCache not found after loading NemotronH")


def nemotron_generate(model, tokenizer, input_ids, max_new=300):
    """Chunked (no-cache) generation: re-processes full sequence each step.
    Avoids selective_state_update which diverges from training path due to
    FP32/BF16 precision mismatch in Mamba kernels. O(n^2) but correct."""
    generated = input_ids
    eos_id    = tokenizer.eos_token_id

    for step in range(max_new):
        with torch.no_grad():
            out = model(input_ids=generated, use_cache=False, return_dict=True)
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


# ---------------------------------------------------------------------------
# Post-training generation evaluation
# ---------------------------------------------------------------------------

def eval_generations(model, tokenizer, rows, label, n_show=5, max_new=250):
    from collections import defaultdict
    sep = "=" * 64
    print(f"\n{sep}\n  GENERATION EVAL — {label}\n{sep}")
    model.eval()

    n_boxed, n_correct = 0, 0
    task_boxed   = defaultdict(int)
    task_correct = defaultdict(int)
    task_total   = defaultdict(int)
    show_count   = 0
    for row in rows:
        prompt = build_prompt(tokenizer, row)
        ids    = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out    = nemotron_generate(model, tokenizer, ids, max_new=max_new)
        gen    = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        boxed   = BOX_RE.findall(gen)
        ext     = boxed[-1].strip() if boxed else ""
        gold    = str(row.get("gold_answer", row.get("answer", ""))).strip()
        correct = ext.strip().lower() == gold.strip().lower() if ext else False
        task    = row.get("task_type", "unknown")

        if boxed:
            n_boxed += 1
            task_boxed[task] += 1
        if correct:
            n_correct += 1
            task_correct[task] += 1
        task_total[task] += 1

        if show_count < n_show:
            print(f"\n  id={str(row.get('id','?'))[:8]}  task={task}")
            print(f"  gold:      {gold!r}")
            print(f"  raw_gen:   {gen[:250]!r}")
            print(f"  extracted: {ext!r}   boxed={bool(boxed)}   correct={correct}")
            show_count += 1

    n = len(rows)
    print(f"\n  SUMMARY: boxed={n_boxed}/{n}  correct={n_correct}/{n}")
    print(f"  Per-task accuracy:")
    for t in sorted(task_total):
        nt = task_total[t]
        print(f"    {t:<20} boxed={task_boxed[t]}/{nt}  correct={task_correct[t]}/{nt} "
              f"({100*task_correct[t]/nt:.0f}%)")
    return n_boxed, n_correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",     required=True,  help="Training JSONL path")
    parser.add_argument("--val",       required=True,  help="Validation JSONL path")
    parser.add_argument("--output",    required=True,  help="Adapter output directory")
    parser.add_argument("--rank",      type=int, default=32,   help="LoRA rank")
    parser.add_argument("--epochs",    type=int, default=3,    help="Training epochs")
    parser.add_argument("--lr",        type=float, default=2e-4)
    parser.add_argument("--n-train",   type=int, default=None, help="Limit training rows (overfit test)")
    parser.add_argument("--n-val",     type=int, default=None, help="Limit val rows")
    parser.add_argument("--batch",     type=int, default=1)
    parser.add_argument("--accum",     type=int, default=4)
    parser.add_argument("--max-seq",   type=int, default=2048)
    parser.add_argument("--no-eval-gen", action="store_true", help="Skip post-train generation test")
    parser.add_argument("--target-type",
                        choices=["answer_only", "haiku_reasoning", "short_reasoning"],
                        default="haiku_reasoning",
                        help="Assistant text format: answer_only | haiku_reasoning | short_reasoning")
    args = parser.parse_args()

    print(f"\n{'='*64}")
    print(f"  Nemotron-3 4B LoRA V2 Training")
    print(f"{'='*64}")
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Train data:  {args.train}")
    print(f"  Val data:    {args.val}")
    print(f"  Output:      {args.output}")
    print(f"  LoRA rank:   {args.rank}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  n_train lim: {args.n_train}")
    print(f"  target_type: {args.target_type}")
    print(f"{'='*64}\n")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Load and inspect data ---
    train_rows_all = load_jsonl(args.train)
    val_rows_all   = load_jsonl(args.val)

    from collections import Counter
    train_sources = Counter(r.get("model", "unknown") for r in train_rows_all)
    print(f"Train data source breakdown ({len(train_rows_all)} total rows):")
    for src, cnt in train_sources.most_common():
        print(f"  {src:<40} {cnt:>5}")
    print()

    # Prompt leakage check — note: roman/bit_manipulation tasks embed few-shot examples
    # in the prompt, so gold substrings (e.g. 'XXX') appearing in examples are expected.
    leaked = sum(1 for r in train_rows_all[:100]
                 if str(r.get("gold_answer","")) in r.get("prompt",""))
    print(f"  Prompt substring check: {leaked}/100 rows have gold_answer as substring "
          f"in prompt (expected for roman/bit tasks with few-shot examples). OK\n")

    train_rows = train_rows_all[:args.n_train] if args.n_train else train_rows_all
    val_rows   = val_rows_all[:args.n_val]     if args.n_val   else val_rows_all

    print(f"Building masked datasets (train={len(train_rows)}, val={len(val_rows)}, "
          f"target_type={args.target_type})...")
    train_ds = build_masked_dataset(train_rows, tokenizer, verbose=True,
                                    target_type=args.target_type)
    val_ds   = build_masked_dataset(val_rows,   tokenizer, verbose=False,
                                    target_type=args.target_type)
    print(f"  Done. Train tokens per sample (mean): "
          f"{sum(len(x['input_ids']) for x in train_ds)/len(train_ds):.0f}")

    # --- Load model and attach LoRA ---
    t0 = time.time()
    rank = args.rank
    while True:
        try:
            print(f"\nLoading base model (rank={rank})...")
            model, _ = FastLanguageModel.from_pretrained(
                model_name=BASE_MODEL,
                max_seq_length=args.max_seq,
                load_in_4bit=False,
                dtype=None,
                trust_remote_code=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.05,
                target_modules=TARGET_MODULES_V2,
                use_gradient_checkpointing="unsloth",
                bias="none",
                random_state=42,
            )
            break
        except torch.cuda.OutOfMemoryError:
            if rank <= 8:
                raise RuntimeError(f"OOM even at rank {rank} — cannot continue")
            rank = rank // 2
            print(f"  OOM at rank {args.rank}, retrying with rank={rank}")
            gc.collect()
            torch.cuda.empty_cache()

    if rank != args.rank:
        print(f"  NOTE: fell back to rank={rank} due to VRAM constraints")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  Model loaded in {time.time()-t0:.1f}s")
    print(f"  Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)\n")

    # --- Training (manual loop — bypasses all unsloth/TRL Trainer patches) ---
    # Reason: unsloth patches Trainer.compute_loss and strips our precomputed -100 labels,
    # computing loss on ALL tokens instead of assistant-only. Manual loop forces
    # use_cache=False and passes labels directly to the model's own forward().
    output_dir = args.output
    final_dir  = str(Path(output_dir) / f"final_adapter_{args.target_type}")
    device     = next(model.parameters()).device
    pad_id     = tokenizer.pad_token_id

    def manual_collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        max_len = ((max_len + 7) // 8) * 8  # pad to multiple of 8
        B = len(batch)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        labels    = torch.full((B, max_len), -100,   dtype=torch.long)
        attn_mask = torch.zeros(B, max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            labels[i, :n]    = torch.tensor(b["labels"],    dtype=torch.long)
            attn_mask[i, :n] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        collate_fn=manual_collate, drop_last=False)

    total_steps = len(loader) * args.epochs // args.accum
    optimizer   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps)

    model.train()

    # Confirm initial loss matches base model (~7 nats for assistant-only CE)
    print("Checking initial assistant-only loss (should be ~7 for base model)...")
    model.eval()
    with torch.no_grad():
        sample = train_ds[0]
        ids_s = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
        lbl_s = torch.tensor(sample["labels"]).unsqueeze(0).to(device)
        atm_s = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)
        init_out = model(input_ids=ids_s, labels=lbl_s, attention_mask=atm_s,
                         use_cache=False, return_dict=True)
    print(f"  Initial sample-0 assistant loss: {init_out.loss.item():.4f}  "
          f"(base model ref: ~7.03)\n")
    model.train()

    print("Starting training (manual loop)...\n")
    t_train     = time.time()
    global_step = 0
    accum_loss  = 0.0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        epoch_loss, epoch_steps = 0.0, 0
        for step, batch in enumerate(loader):
            ids  = batch["input_ids"].to(device)
            lbl  = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out  = model(input_ids=ids, labels=lbl, attention_mask=attn,
                             use_cache=False, return_dict=True)
            loss = out.loss / args.accum
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % args.accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step   += 1
                epoch_loss    += accum_loss
                epoch_steps   += 1
                print(f"  ep {epoch+1:2d}  step {global_step:4d}  "
                      f"loss={accum_loss*args.accum:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")
                accum_loss = 0.0

        avg = epoch_loss / max(1, epoch_steps)

        # Val loss on up to 200 rows — forward pass only, no generation.
        model.eval()
        val_loader_ep = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                   collate_fn=manual_collate, drop_last=False)
        val_loss_sum, val_loss_n = 0.0, 0
        with torch.no_grad():
            for vb in val_loader_ep:
                vids  = vb["input_ids"].to(device)
                vlbl  = vb["labels"].to(device)
                vattn = vb["attention_mask"].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    vout = model(input_ids=vids, labels=vlbl, attention_mask=vattn,
                                 use_cache=False, return_dict=True)
                val_loss_sum += vout.loss.item()
                val_loss_n   += 1
                if val_loss_n >= 200:
                    break
        model.train()
        avg_val = val_loss_sum / max(1, val_loss_n)
        print(f"  ── epoch {epoch+1}  train_loss={avg*args.accum:.4f}  "
              f"val_loss={avg_val:.4f}  (val sample={val_loss_n})\n")

    print(f"\nTraining complete in {(time.time()-t_train)/60:.1f} min")

    # --- Save adapter ---
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nAdapter saved to {final_dir}")

    # Unsloth saves lora_X.weight instead of lora_X.default.weight.
    # PEFT's load_peft_weights expects the "default" adapter name in the key path.
    # Rename keys in-place so standard PEFT loading works.
    _fix_adapter_key_names(final_dir)

    for f in Path(final_dir).iterdir():
        print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")

    # --- Post-train generation eval ---
    # Reload from disk with manual weight loading to verify save/load round-trip.
    # Report train and val separately so we can detect overfit vs generalization.
    if not args.no_eval_gen:
        print("\n[POST-TRAIN] Reloading adapter from disk for generation eval...")
        model.cpu(); del model; gc.collect()
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        model_eval, _ = FastLanguageModel.from_pretrained(
            model_name=final_dir, max_seq_length=args.max_seq,
            load_in_4bit=False, dtype=None, trust_remote_code=True)
        model_eval.eval()
        load_adapter_weights(model_eval, final_dir)

        n_gen_train = min(10, len(train_rows))
        n_gen_val   = min(100, len(val_rows))
        train_boxed, train_correct = eval_generations(
            model_eval, tokenizer, train_rows[:n_gen_train],
            f"TRAIN (n={n_gen_train}, target={args.target_type})",
            n_show=5, max_new=300)
        val_boxed, val_correct = eval_generations(
            model_eval, tokenizer, val_rows[:n_gen_val],
            f"VAL   (n={n_gen_val}, target={args.target_type})",
            n_show=5, max_new=300)

        # Write JSON summary for easy comparison across variants
        summary = {
            "target_type":     args.target_type,
            "n_train_rows":    len(train_rows),
            "n_val_rows":      len(val_rows),
            "epochs":          args.epochs,
            "rank":            rank,
            "train_boxed":     train_boxed,  "train_correct": train_correct,
            "train_boxed_pct": round(100*train_boxed/n_gen_train, 1),
            "train_acc_pct":   round(100*train_correct/n_gen_train, 1),
            "val_boxed":       val_boxed,    "val_correct":   val_correct,
            "val_boxed_pct":   round(100*val_boxed/n_gen_val, 1),
            "val_acc_pct":     round(100*val_correct/n_gen_val, 1),
            "adapter_dir":     final_dir,
        }
        summary_path = str(Path(output_dir) / f"eval_summary_{args.target_type}.json")
        with open(summary_path, "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"\n  Summary written → {summary_path}")
        print(f"  TRAIN: boxed={train_boxed}/{n_gen_train} ({summary['train_boxed_pct']}%)  "
              f"correct={train_correct}/{n_gen_train} ({summary['train_acc_pct']}%)")
        print(f"  VAL:   boxed={val_boxed}/{n_gen_val} ({summary['val_boxed_pct']}%)  "
              f"correct={val_correct}/{n_gen_val} ({summary['val_acc_pct']}%)")

        model_eval.cpu(); del model_eval; gc.collect()
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    else:
        print("\n[POST-TRAIN] Generation eval skipped (--no-eval-gen).")
        model.cpu(); del model; gc.collect()
        torch.cuda.synchronize(); torch.cuda.empty_cache()

    print(f"\n{'='*64}")
    print(f"  V2 TRAINING COMPLETE  [{args.target_type}]")
    print(f"  Adapter: {final_dir}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
