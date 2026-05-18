# Nemotron Competition — Technical Reference

## Architecture: Nemotron-H Mamba-2 Hybrid

Nemotron-H is **not** a standard transformer. It is a Mamba-2 Transformer Hybrid:
- ~75% Mamba-2 SSM layers (selective state-space model, no Q/K/V attention)
- ~25% standard transformer attention layers (only **4** attention layers in the 4B model)
- MLP blocks between layers

Three consequences that will cause failures if ignored:

### 1. Flash Attention 2 is not supported
FA2 dispatch operates on Q/K/V projections. Mamba layers have none. Do not set
`attn_implementation="flash_attention_2"` — it will crash or silently fail on the Mamba layers.
The professor's notebook detects FA2 failures and falls back to `eager` automatically.
Always use `attn_implementation="eager"` (or omit it, let the model default).

### 2. LoRA target modules must exclude Mamba weights
Mamba SSM layers use `in_proj`, `out_proj`, `x_proj`, `dt_proj`, `A_log`, `D` — none of these
are valid LoRA targets for reasoning fine-tuning (they control the state-space dynamics).

**Valid targets** (attention + MLP projections only):
```
q_proj, k_proj, v_proj, o_proj   ← the 4 attention layers
gate_proj, up_proj, down_proj    ← MLP blocks
```

### 3. vLLM requires `--mamba_ssm_cache_dtype float32`
The Mamba state cache must be float32 for numerical stability. Without this flag, the Mamba
layers produce NaN/garbage outputs during inference.

---

## Why `save_only_model=True` is critical

With LoRA rank=32 on a 30B model: ~883M LoRA parameters × 2 AdamW moments × 4 bytes ≈ **6 GB**
per checkpoint (optimizer state alone). With `save_total_limit=1`, the sequence at step 400 is:
1. Write checkpoint-400 (adapter + optimizer = ~6.3 GB)
2. Delete checkpoint-200 (~6.3 GB)
3. Peak disk usage: **12.6 GB** — causes write corruption on Kaggle's 20 GB working dir limit

`save_only_model=True` writes only the adapter weights (~300 MB), skipping optimizer state.
**Trade-off:** cannot resume training from these checkpoints. For a one-shot competition run,
that's acceptable.

---

## LoRA Hyperparameters (from professor's final working config)

| Parameter | Local 4B (smoke) | Cloud 30B (submission) |
|---|---|---|
| lora_r | 8 | 32 |
| lora_alpha | 16 → 64\* | 64 |
| learning_rate | 2e-4 | 1e-4 |
| optimizer | adamw_torch_fused | adamw_torch_fused |
| max_seq_length | 2048 | 4096 |
| reasoning word cap | — | 600 words |

\*Professor used alpha=64 in final run. Local smoke test uses alpha=16 (cheaper).

---

## vLLM Serving (local 4B inference test)

```bash
# Download the custom reasoning parser first
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/resolve/main/nano_v3_reasoning_parser.py

# Serve base model + LoRA adapter
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --served-model-name nemotron-4b \
  --trust-remote-code \
  --mamba_ssm_cache_dtype float32 \
  --enable-lora \
  --lora-modules local_lora=outputs/adapters/local_4b/final_adapter \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --port 8000
```

Verify both models are listed:
```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

---

## Execution Checklist

### Mac → Git → Linux setup

- [ ] Professor's files copied to project root
- [ ] `.gitignore` excludes `*.jsonl`, `*.csv`, `outputs/`, `data/splits/`
- [ ] `git init` + initial commit (code only, no data)
- [ ] `gh repo create nemotron-competition --public` + push
- [ ] Linux: `git clone` the new repo
- [ ] Linux: `conda create -n nemotron-train python=3.11`
- [ ] Linux: `pip install -r requirements-train.txt`
- [ ] Data files transferred to Linux separately (not via git)

### Local 4B smoke test

- [ ] Build splits: `python -m src.build_splits --input train_reasoning_v6.jsonl --out-dir data/splits`
- [ ] Smoke train: `python -m src.train_lora --config configs/local_4b.yaml --train data/splits/smoke_20.jsonl --val data/splits/val.jsonl`
- [ ] Confirm: loss decreases, no OOM, adapter saves cleanly
- [ ] Validate: `python -m src.validate_adapter --adapter-dir outputs/adapters/local_4b/final_adapter`
- [ ] (Optional) vLLM serve + eval_local.py for before/after accuracy

### Cloud 30B submission run

- [ ] Run A — smoke_20 on 30B: confirm loads (no Mamba-layer errors), ~5 min
- [ ] Run B — 500 samples: confirm loss curve and checkpoint integrity
- [ ] Run C — full dataset: final adapter
- [ ] Validate final adapter
- [ ] `python -m src.package_submission --adapter-dir outputs/adapters/cloud_30b/final_adapter`
- [ ] Submit zip to Kaggle
- [ ] Screenshot confirmation page → Canvas upload

---

## Before/After Accuracy (fill in after eval runs)

| Model | Overall | roman | gravity | unit_conversion | bit_manipulation | cipher_text | symbol_transform |
|---|---|---|---|---|---|---|---|
| Base 4B | — | — | — | — | — | — | — |
| LoRA 4B | — | — | — | — | — | — | — |
| Base 30B | — | — | — | — | — | — | — |
| LoRA 30B | — | — | — | — | — | — | — |

---

## Reference Files

| File | Purpose |
|---|---|
| `compare_lora_before_after_v2.py` | Professor's eval script — supports local Unsloth + NVIDIA API |
| `nvidia-nemotron-my-train.ipynb` | Professor's training notebook — architecture notes, working cmds |
| `configs/local_4b.yaml` | Smoke test config (4B model, r=8) |
| `configs/cloud_30b.yaml` | Submission config (30B model, r=32) |
| `src/train_lora.py` | Training script (Unsloth-based) |
| `src/build_splits.py` | Splits train_reasoning_v6.jsonl → train/val/smoke_20 |
| `src/validate_adapter.py` | Checks adapter before packaging |
| `src/eval_local.py` | Runs eval against vLLM endpoint |
| `src/package_submission.py` | Validates + zips adapter for Kaggle |
