# Phase 03 — Local Smoke Test (4B)

Train a LoRA adapter on the 4B model locally to validate the training pipeline before committing GPU hours on cloud.

## What this phase does

1. Trains a LoRA adapter on a small subset using the 4B model
2. Validates the adapter loads and generates correct output
3. Runs a lightweight eval to sanity-check quality

## Order of execution

```bash
python src/train_lora.py     --config configs/local_4b.yaml
python src/validate_adapter.py
python src/eval_local.py
```

Outputs land in `outputs/adapters/` and `outputs/evals/` (gitignored).

## Config

`configs/local_4b.yaml` — model name, LoRA rank/alpha, training hyperparams.
