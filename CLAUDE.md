# CLAUDE.md — Agent Rules

## Repository layout
This project uses a sequential phase structure. Do not collapse wrapper directories or rename phase folders.

```
phase01_teacher_benchmark/   teacher model evaluation and trace quality checks
phase02_data_generation/     LLM-generated training data and splits
phase03_local_smoke/         local LoRA training and eval (RTX 5070 Ti)
phase04_cloud_train/         cloud-scale training and submission packaging
shared/                      raw competition data and shared utilities
docs/                        conventions and experiment records
notebooks/                   exploratory notebooks (not part of training pipeline)
```

Each phase owns `src/`, `configs/`, and `outputs/`. The `outputs/` tree is gitignored everywhere.

## Script discipline

**Modify an existing script** for: bug fixes, new CLI flags, provider swaps (e.g. Gemini→DeepSeek), parsing fixes, eval metric additions, small refactors that preserve the same input/output contract.

**Create a new script** only when: the new script has a distinct purpose, a changed input/output contract (different data format, different task), or must preserve the old script as a reproducible path for a completed experiment.

When in doubt: modify, not create. One script per job, not one script per run.

## Data and model artifacts

Generated data, model checkpoints, logs, adapter weights, and caches must never be committed. They live under `**/outputs/` (gitignored) or cloud storage.

Dataset versions are named `v<N>` and stored under `phase02_data_generation/data/v<N>/`. Never overwrite a prior version in place — bump the version number.

## Secrets

Never print or commit API keys. Use `.env` (gitignored) or shell environment only. `.env.*` variants are also gitignored.

## Training

Training is LoRA-only unless explicitly stated otherwise. The trainable modules are Attn+MLP only (Mamba layers excluded due to FP32/BF16 kernel mismatch). Do not change this without explicit instruction.

## Generation / inference

Chunked (no-cache) generation (`use_cache=False` at every step) is the correct path for NemotronH. Cached generation (`selective_state_update`) diverges from training and produces degenerate output. This is an inference fix only — it does not affect model architecture or LoRA weights.

Do not use `model.generate()` — broken for NemotronH due to KV cache bug.

## Experiment records

Every experiment that produces an eval result must write a short record under `docs/experiments/`. Name it `NNN_<slug>.md` (zero-padded three-digit index). Minimum fields: adapter path, dataset path, training settings, eval command, parse%, accuracy%, per-task breakdown, and path to the raw eval JSON.
