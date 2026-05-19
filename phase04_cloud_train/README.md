# Phase 04 — Cloud Training (30B)

Full training run on the 30B MoE model using cloud GPU resources.

## What this phase does

Same pipeline as Phase 03 but targeting the 30B model with a higher-rank LoRA config.
Produces the submission adapter.

## Order of execution

```bash
python ../phase03_local_smoke/src/train_lora.py --config configs/cloud_30b.yaml
python ../phase03_local_smoke/src/validate_adapter.py
python ../../shared/src/package_submission.py   # writes to outputs/submissions/
```

Outputs land in `outputs/adapters/` and `outputs/submissions/` (gitignored).

## Config

`configs/cloud_30b.yaml` — 30B model name, lora_rank 32, longer eval/save cadence.
