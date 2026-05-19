# Phase 02 — Full Dataset Generation

Use the winning teacher model from Phase 01 to generate the complete training dataset.

## What this phase does

1. Streams all training examples through the teacher model
2. Validates outputs for format and quality
3. Builds train/val splits and writes them to `data/`

## Order of execution

```bash
python src/generate_full_dataset.py  # calls teacher API, writes raw JSONL to data/
python src/validate_dataset.py       # checks format, logs bad rows
python src/build_splits.py           # train/val split → data/train.jsonl, data/val.jsonl
```

## Config

`configs/generation.yaml` — teacher model, batch size, temperature, output paths.
