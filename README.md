# Nemotron Competition — Project Overview

Fine-tune NVIDIA Nemotron for the Kaggle competition using a 4-phase pipeline.

## Phase Execution Order

```
phase01  →  phase02  →  phase03  →  phase04
benchmark    generate    smoke        cloud
teachers     dataset     test (4B)    train (30B)
```

### Phase 01 — Teacher Benchmark (`phase01_teacher_benchmark/`)
Compare candidate teacher models (Qwen3.5 Plus, DeepSeek V4 Pro) on a small held-out slice. Pick the best one before spending API quota on full generation.

### Phase 02 — Data Generation (`phase02_data_generation/`)
Stream all training examples through the winning teacher model. Validate, clean, and split into train/val JSONL files.

### Phase 03 — Local Smoke Test (`phase03_local_smoke/`)
Train a LoRA adapter on the 4B model locally. Fast iteration to validate the training script, data format, and eval harness before touching cloud GPUs.

### Phase 04 — Cloud Training (`phase04_cloud_train/`)
Full training run on the 30B MoE model. Same training script as Phase 03, different config. Produces the final submission adapter.

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
