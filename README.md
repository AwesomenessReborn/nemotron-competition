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
| Phase 01 — Teacher Benchmark | ✅ Complete | Gemini 2.5 Flash selected as primary teacher |
| Phase 02 — Data Generation | ✅ Complete | 9,500 rows generated; train=8,550 / val=950 |
| Phase 03 — Local Smoke Test (4B) | ✅ Complete | LoRA adapter trained; eval loss 6.50 → 1.44 (78% reduction) |
| Phase 04 — Cloud Training (30B) | ⏳ Pending | Requires GPU VM with ≥80 GB VRAM — see below |

---

## Phase 01 — Teacher Benchmark (`phase01_teacher_benchmark/`)

**Goal:** Find the best teacher model to generate reasoning traces before spending API quota on the full 9,500-row dataset.

**Models evaluated:**
- `gemini-2.5-flash` (Google) — with `thinking_budget=0` (mandatory; extended thinking costs ~10x more per sample)
- `claude-haiku-4-5` (Anthropic)

Both models were run on a 50-sample stratified smoke test covering all task types (bit manipulation, cipher text, gravity/physics, roman numerals, unit conversion, etc.).

**Results:**

| Model | Parse success | Answer match |
|-------|--------------|-------------|
| Gemini 2.5 Flash | ≥ 90% | ≥ 90% |
| Claude Haiku 4.5 | ≥ 90% | ≥ 90% |

Both passed the ≥90% gate. **Gemini 2.5 Flash was selected as the primary teacher** for full generation due to its stronger structured reasoning output format. Haiku was kept as a backup and used to generate a parallel v7 dataset for comparison/merge.

---

## Phase 02 — Data Generation (`phase02_data_generation/`)

**Goal:** Generate reasoning traces for all 9,500 competition rows using the selected teacher model.

**Process:**
1. Ran `generate_full_gemini.py` — streamed all 9,500 rows through Gemini 2.5 Flash, producing `train_reasoning_v7_haiku.jsonl` (v7 dataset)
2. Merged with the professor's existing `train_reasoning_v6.jsonl` dataset — deduplication by row ID with v7 taking priority
3. Result: complete overlap (all v6 IDs existed in v7), so the final merged dataset is 9,500 v7 rows

**Dataset split:**

| Split | Rows | Path |
|-------|------|------|
| Train | 8,550 | `phase02_data_generation/data/merged/train.jsonl` |
| Val | 950 | `phase02_data_generation/data/merged/val.jsonl` |
| Smoke (stratified) | 50 | `phase02_data_generation/data/merged/smoke_50.jsonl` |

**Training objective note:** Each row is formatted as *problem + gold answer → reasoning explanation*. The model learns to explain why the gold answer is correct, not to solve from scratch. This is intentional — the competition provides the problem, and the model needs to produce high-quality reasoning traces.

---

## Phase 03 — Local Smoke Test (`phase03_local_smoke/`)

**Goal:** Validate the full training pipeline end-to-end on the 4B model before spending cloud GPU time on the 30B.

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`

**Key architectural notes discovered during this phase:**
- Nemotron-H is a hybrid architecture: ~75% Mamba-2 SSM layers + ~25% standard attention layers
- No Flash Attention 2 support — uses xformers or falls back to standard attention
- Requires `trust_remote_code=True`
- `mamba-ssm` must be compiled from source on CUDA 13.0 (RTX 5070 Ti / Blackwell GPUs — no prebuilt wheels exist yet)
- `transformers 5.5.0` has a known incompatibility with Nemotron-H's `HybridMambaAttentionDynamicCache` during generation — eval used loss-based validation as a workaround

**Training results:**

| Metric | Value |
|--------|-------|
| Steps | 1,100 |
| Eval loss (start) | 6.50 |
| Eval loss (end) | 1.44 |
| Reduction | 78% |
| Hardware | RTX 5070 Ti (16 GB VRAM) |
| Training rows | 8,550 |

**LoRA adapter saved at:** `phase03_local_smoke/outputs/adapters/local_4b/final_adapter/`

| LoRA Parameter | Value |
|----------------|-------|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` |
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q/k/v/o/gate/up/down proj |
| Bias | none |

**Pipeline validated:** model loads, LoRA attaches, training runs, loss decreases, adapter saves and reloads cleanly. Training script (`phase03_local_smoke/src/train_lora.py`) is the same script used for Phase 04 — only the config changes.

---

## Phase 04 — Cloud Training (`phase04_cloud_train/`)

**Goal:** Full training run on the 30B MoE model to produce the final competition submission adapter.

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- 30B total parameters, ~3B active per token (mixture-of-experts)
- BF16 weights = ~60 GB; peak VRAM with gradient checkpointing ≈ 70–75 GB
- **Requires a GPU with ≥80 GB VRAM** (A100 80 GB or H100 80 GB)

**Status: All deployment scripts are ready — waiting on GPU provisioning.**

See [`phase04_cloud_train/DEPLOY.md`](phase04_cloud_train/DEPLOY.md) for the full step-by-step guide.

**Quick start:**
```bash
# 1. Provision: Ubuntu 22.04, CUDA 12.x, ≥80 GB VRAM, ≥100 GB disk
# 2. Clone repo and run setup (~20 min, compiles mamba-ssm from source)
bash phase04_cloud_train/setup.sh

# 3. Log in to HuggingFace and accept model license
conda run -n nemotron-train huggingface-cli login

# 4. Upload training data via scp from local machine
scp phase02_data_generation/data/merged/train.jsonl user@VM_IP:~/nemotron-competition/phase02_data_generation/data/merged/
scp phase02_data_generation/data/merged/val.jsonl   user@VM_IP:~/nemotron-competition/phase02_data_generation/data/merged/

# 5. Run training (preflight checks run automatically)
bash phase04_cloud_train/run_training.sh
```

**30B LoRA config:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank | 32 | Higher than 4B run — more model capacity |
| LoRA alpha | 64 | 2× rank |
| Learning rate | 0.0001 | Lower than 4B run — larger model = smaller LR |
| Gradient accumulation | 8 | Effective batch size = 8 |
| Max seq length | 2048 | |
| `save_only_model` | true | Skips optimizer state — saves ~30 GB per checkpoint |
| `load_in_4bit` | false | Set true if <80 GB VRAM (slight quality loss) |

**Recommended providers:** Lambda Labs, RunPod, Vast.ai, CoreWeave
**Estimated training time:** 3–5 hours on H100, 5–7 hours on A100
**Estimated cost:** $6–$20

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
