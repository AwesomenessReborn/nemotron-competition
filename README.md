# Nemotron Competition — Project Overview

Fine-tune NVIDIA Nemotron-H 4B for the Kaggle competition using a 4-phase pipeline.

## Phase Execution Order

```
phase01  →  phase02  →  phase03  →  phase04
benchmark    generate    smoke        cloud
teachers     dataset     test (4B)    train (30B)
```

## Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 01 — Teacher Benchmark | ✅ Complete | Claude Haiku 4.5 selected for v7; local Gemma 4 12B used for v8 |
| Phase 02 — Data Generation | ✅ v7 + v8 complete; v8.1 in progress | v7: 9,500 haiku rows; v8: 8,344 local Gemma rows; v8.1: symbol repair regen |
| Phase 03 — Local Smoke Test (4B) | ✅ V8 trained + eval complete | 96.1% parse / 24.5% accuracy — +4.9% over haiku baseline |
| Phase 04 — Cloud Training (30B) | ⏳ Pending | Waiting for v8.1 clean dataset and user approval |

---

## Phase 01 — Teacher Benchmark (`phase01_teacher_benchmark/`)

**Goal:** Evaluate teacher models on reasoning trace quality before committing API quota or GPU time to full generation.

**Models evaluated:**
- `claude-haiku-4-5` (Anthropic) — run on a 50-sample stratified smoke test
- `gemini-2.5-flash` (Google) — with `thinking_budget=0`
- `local Gemma 4 12B` via llama.cpp raw `/completion` — evaluated for V8 generation

**Results:**
- Both Haiku and Gemini passed the ≥90% parse/answer-match gate on the 50-sample test.
- **Claude Haiku 4.5 was used for the v7 dataset** (full 9,500 rows, haiku_reasoning format).
- **Local Gemma 4 12B was selected for the v8 dataset** — 100% parse on 100-row gate, $0 cost, richer post-hoc rationale format. Access via raw `/completion` endpoint (not `/v1/chat/completions` — that endpoint strips thinking tokens).

---

## Phase 02 — Data Generation (`phase02_data_generation/`)

**Goal:** Generate reasoning traces for all 9,500 competition rows using the selected teacher model.

### V7 Dataset — Claude Haiku 4.5 (baseline)

**Process:**
1. Ran `generate_full_gemini.py` / haiku variant — streamed all 9,500 rows through Claude Haiku 4.5 producing `train_reasoning_v7_haiku.jsonl`
2. Result: 9,500 rows, 100% quality, `haiku_reasoning` format

**Dataset split:**

| Split | Rows | Path |
|-------|------|------|
| Train | 8,550 | `phase02_data_generation/data/merged/train.jsonl` |
| Val | 950 | `phase02_data_generation/data/merged/val.jsonl` |
| Smoke (stratified) | 50 | `phase02_data_generation/data/merged/smoke_50.jsonl` |

### V8 Dataset — Local Gemma 4 12B (current)

**Motivation:** Haiku reasoning traces are short and approximate — the v7-trained adapter scored 0% on gravity, unit_conversion, bit_manipulation, and symbol_transform. V8 uses a **post-hoc rationale format**: the gold answer is provided in the prompt and the model writes detailed reasoning explaining why it's correct. This produces higher-quality training signal with no API cost.

**Teacher model:** `local Gemma 4 12B` via llama.cpp raw `/completion`, workers=8, n_predict=512

**Generation stats:**
- 9,376 raw rows processed → 596 empty-reasoning rows excluded → **8,780 usable**
- 95/5 stratified train/val split: **8,344 train / 436 val**
- 355 train + 30 val rows used a fixed generic repair string (symbol_transform leakage — see V8.1)

**Script:** `phase02_data_generation/src/generate_v8_local_gemma.py`

**Dataset split:**

| Split | Rows | Path |
|-------|------|------|
| Train | 8,344 | `phase02_data_generation/data/v8/train_reasoning_v8_local_gemma_clean.jsonl` |
| Val | 436 | `phase02_data_generation/data/v8/val_reasoning_v8_local_gemma_clean.jsonl` |

### V8.1 Dataset — Symbol Repair Fix (in progress)

**Problem:** 355 train / 30 val symbol_transform rows used a fixed generic reasoning string
(`"The correct symbol sequence is provided for this post-hoc training trace, so I copy it exactly."`)
during V8 generation. The model learned to emit this string at inference time, providing no
puzzle-solving signal (symbol_transform accuracy stayed at 0%).

**Fix:** `phase02_data_generation/src/regen_v8_1_symbol_repairs.py` regenerates those 385 rows
with task-specific reasoning that names the actual character mapping rule. Rows that fail the
banned-phrase check or don't reproduce gold_answer are excluded rather than kept with generic text.

**Output (pending):**
- `phase02_data_generation/data/v8/train_reasoning_v8_1_local_gemma_clean.jsonl`
- `phase02_data_generation/data/v8/val_reasoning_v8_1_local_gemma_clean.jsonl`

---

## Phase 03 — Local Smoke Test (`phase03_local_smoke/`)

**Goal:** Validate the full training pipeline end-to-end on the 4B model before spending cloud GPU time on the 30B.

**Model:** `nvidia/NVIDIA-Nemotron-H-4B-BF16`

**Key architectural constraints discovered during this phase:**
- Nemotron-H is a hybrid: ~75% Mamba-2 SSM layers + ~25% standard attention layers
- **LoRA target modules: Attn+MLP only** (`q/k/v/o/up/down_proj`) — Mamba layers excluded due to FP32/BF16 kernel mismatch that causes degenerate generation at position 138
- **Chunked (no-cache) generation is mandatory** — `selective_state_update` diverges from the training path; `model.generate()` is broken for NemotronH
- `mamba-ssm` must be compiled from source on CUDA 13.0 (RTX 5070 Ti / Blackwell GPUs — no prebuilt wheels)
- Manual training loop used (not TRL SFTTrainer) — Unsloth patches `compute_loss` and strips precomputed -100 labels

### V7 Haiku Baseline (Experiment 001)

Adapter path: `phase03_local_smoke/outputs/adapters/local_4b/final_adapter_haiku_reasoning`

| Metric | Value |
|--------|-------|
| Training rows | 8,550 (v7 haiku) |
| Rank | 8 |
| Epochs | 5 |
| LR | 2e-4 |
| Parse % | 86.3% |
| Accuracy % | 19.6% |
| Hardware | RTX 5070 Ti (16 GB VRAM) |

Per-task eval (n=102, n_per_task=17, max_new=250):

| task_type | parse | accuracy |
|-----------|-------|----------|
| roman | 100% | 100% |
| cipher_text | 94.1% | 17.6% |
| gravity | 100% | 0% |
| unit_conversion | 100% | 0% |
| bit_manipulation | 64.7% | 0% |
| symbol_transform | 58.8% | 0% |

### V8 Local Gemma Clean (Experiment 003) — **current best**

Adapter path: `phase03_local_smoke/outputs/adapters/v8_local_gemma_clean/final_adapter_haiku_reasoning`

| Metric | V8 Local Gemma | V7 Haiku Baseline | Δ |
|--------|---------------|-------------------|---|
| Training rows | 8,344 | 8,550 | −206 |
| Rank | 32 | 8 | +24 |
| Epochs | 5 | 5 | — |
| LR | 2e-4 | 2e-4 | — |
| Trainable params | 0.507% | ~0.13% | — |
| Wall clock | 63.3 min | — | — |
| Val loss (best) | 0.344 (ep4) | 0.477 | −28% |
| **Parse %** | **96.1%** | 86.3% | **+9.8%** |
| **Accuracy %** | **24.5%** | 19.6% | **+4.9%** |

Per-task eval (n=102, n_per_task=17, max_new=250):

| task_type | parse | accuracy | Δ vs baseline |
|-----------|-------|----------|---------------|
| roman | 100% | 100% | 0% |
| cipher_text | 100% | 35.3% | **+17.7%** |
| bit_manipulation | 100% | 5.9% | +5.9% |
| unit_conversion | 100% | 5.9% | +5.9% |
| gravity | 100% | 0% | 0% |
| symbol_transform | 76.5% | 0% | 0% |

**Key observations:**
- Cipher_text breakthrough (+17.7%) — longer post-hoc reasoning traces directly improved performance
- Symbol_transform accuracy stayed at 0% due to repair-row reasoning leakage → being fixed in V8.1
- Gravity remains at 0% — model learns the formula but extracts incorrect g constant; may need longer reasoning traces or larger max_new at eval
- V8 data produces cleaner output format (parse +9.8%) despite fewer training rows

---

## Phase 04 — Cloud Training (`phase04_cloud_train/`)

**Goal:** Full training run on the 30B MoE model to produce the final competition submission adapter.

**Model:** `nvidia/NVIDIA-Nemotron-H-47B-A22B-BF16` (or equivalent 30B variant)
- BF16 weights ≈ 60 GB; peak VRAM with gradient checkpointing ≈ 70–75 GB
- **Requires a GPU with ≥80 GB VRAM** (A100 80 GB or H100 80 GB)

**Status: Waiting on V8.1 clean dataset completion before starting cloud run.**

See [`phase04_cloud_train/DEPLOY.md`](phase04_cloud_train/DEPLOY.md) for the full step-by-step guide.

**Quick start:**
```bash
# 1. Provision: Ubuntu 22.04, CUDA 12.x, ≥80 GB VRAM, ≥100 GB disk
# 2. Clone repo and run setup (~20 min, compiles mamba-ssm from source)
bash phase04_cloud_train/setup.sh

# 3. Log in to HuggingFace and accept model license
conda run -n nemotron-train huggingface-cli login

# 4. Copy v8.1 training data to VM (from local machine)
scp phase02_data_generation/data/v8/train_reasoning_v8_1_local_gemma_clean.jsonl user@VM_IP:~/nemotron-competition/phase02_data_generation/data/v8/
scp phase02_data_generation/data/v8/val_reasoning_v8_1_local_gemma_clean.jsonl   user@VM_IP:~/nemotron-competition/phase02_data_generation/data/v8/

# 5. Run training (preflight checks run automatically)
bash phase04_cloud_train/run_training.sh
```

**30B LoRA config:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank | 32 | Same as V8 smoke run |
| LoRA alpha | 64 | 2× rank |
| Learning rate | 0.0001 | Lower than 4B run — larger model = smaller LR |
| Gradient accumulation | 8 | Effective batch size = 8 |
| Max seq length | 2048 | |
| Target modules | Attn+MLP only | Mamba excluded — permanent constraint |
| `save_only_model` | true | Skips optimizer state — saves ~30 GB per checkpoint |

**Recommended providers:** Lambda Labs, RunPod, Vast.ai, CoreWeave
**Estimated training time:** 3–5 hours on H100, 5–7 hours on A100
**Estimated cost:** $6–$20

---

## Shared Utilities (`shared/`)

- `shared/src/prompt_template.py` — canonical prompt format used across all phases
- `shared/src/package_submission.py` — packages the adapter for Kaggle submission
- `shared/data/raw/` — put `train.csv` and `train_with_task_type.csv` here (not committed)

## Reference Files

| File | Purpose |
|------|---------|
| `notebooks/nvidia-nemotron-my-train.ipynb` | Professor's reference notebook — do not modify |
| `notebooks/compare_lora_before_after_v2.py` | Professor's reference script — do not modify |
| `PLAN.md` | Project planning notes |
| `HANDOFF.md` | Current status, open questions, and next action for each session |
| `CLAUDE.md` | Agent rules: training constraints, inference rules, backup rules |
| `docs/REPO_CONVENTIONS.md` | Phase responsibilities, script naming, commit hygiene |
| `docs/experiments/` | Per-experiment records (adapter path, settings, eval results) |
| `requirements-train.txt` | Dependencies for training (unsloth, trl, etc.) |
| `requirements-vllm.txt` | Dependencies for vLLM inference |
| `scripts/backup_nemotron_to_drive.sh` | Sync local artifacts to Google Drive via rclone |
