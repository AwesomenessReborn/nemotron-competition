# Nemotron Competition — Project Overview

Fine-tune NVIDIA Nemotron for the Kaggle competition using a 4-phase pipeline.

## Phase Execution Order

```
phase01  →  phase02  →  phase03  →  phase04
benchmark    generate    smoke        cloud
teachers     dataset     test (4B)    train (30B)
```

## Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 01 — Teacher Benchmark | ✅ Complete | Gemini 2.5 Flash + Claude Haiku 4.5 both passed ≥90% gate |
| Phase 02 — Data Generation | ✅ Complete | 9,500 rows generated; train=8,550 / val=950 |
| Phase 03 — Local Smoke Test (4B) | ✅ Complete | LoRA adapter trained; eval loss 6.50 → 1.44 (78% reduction) |
| Phase 04 — Cloud Training (30B) | ⏳ Pending | Requires GPU VM with ≥80 GB VRAM — see below |

---

## Phase 03 — LoRA Adapter (4B, local)

A validated LoRA adapter for `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` is saved at:

```
phase03_local_smoke/outputs/adapters/local_4b/final_adapter/
```

**Training details:**

| Parameter | Value |
|-----------|-------|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Target modules | q/k/v/o/gate/up/down proj |
| Training rows | 8,550 |
| Steps | 1,100 |
| Eval loss (start → end) | 6.50 → 1.44 (78% reduction) |
| Hardware | RTX 5070 Ti (16 GB VRAM) |

---

## Next Step — Phase 04: Cloud Training on 30B Model

The 30B MoE model (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`) requires a GPU with **≥80 GB VRAM** (A100 80 GB or H100 recommended). All deployment scripts are ready.

**See [`phase04_cloud_train/DEPLOY.md`](phase04_cloud_train/DEPLOY.md) for the full step-by-step guide.**

Quick summary:
1. Rent a GPU VM (Lambda Labs / RunPod / Vast.ai — H100 or A100 80 GB)
2. Clone this repo on the VM
3. `bash phase04_cloud_train/setup.sh` — installs conda env, PyTorch, unsloth, mamba-ssm (~20 min)
4. Upload `train.jsonl` and `val.jsonl` via scp
5. `bash phase04_cloud_train/run_training.sh` — runs preflight checks then training (~3–7 hours)
6. Download the final adapter from `phase04_cloud_train/outputs/adapters/cloud_30b/final_adapter/`

**Estimated cost:** $6–$20 depending on provider and GPU.

---

## Phase Descriptions

### Phase 01 — Teacher Benchmark (`phase01_teacher_benchmark/`)
Compare candidate teacher models on a held-out slice. Pick the best one before spending API quota on full generation.

### Phase 02 — Data Generation (`phase02_data_generation/`)
Stream all training examples through the winning teacher models. Validate, clean, and split into train/val JSONL files.

### Phase 03 — Local Smoke Test (`phase03_local_smoke/`)
Train a LoRA adapter on the 4B model locally. Fast iteration to validate the training script, data format, and eval harness before touching cloud GPUs.

### Phase 04 — Cloud Training (`phase04_cloud_train/`)
Full training run on the 30B MoE model. Same training script as Phase 03, different config. Produces the final submission adapter.

---

## Shared Utilities (`shared/`)

- `shared/src/prompt_template.py` — canonical prompt format used across all phases
- `shared/src/package_submission.py` — packages the adapter for Kaggle submission
- `shared/data/raw/` — put `train.csv` and `train_with_task_type.csv` here (not committed)

## Root-Level Reference Files

| File | Purpose |
|------|---------|
| `compare_lora_before_after_v2.py` | Professor's reference script — do not modify |
| `nvidia-nemotron-my-train.ipynb` | Professor's reference notebook — do not modify |
| `PLAN.md` | Project planning notes |
| `requirements-train.txt` | Dependencies for training (unsloth, trl, etc.) |
| `requirements-vllm.txt` | Dependencies for vLLM inference |
