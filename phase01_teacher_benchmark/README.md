# Phase 01 — Teacher Model Benchmark

Test candidate teacher models on a small held-out slice of the training data and pick the best one for full-scale generation.

## What this phase does

1. Samples a test split from the raw training data
2. Sends prompts to each teacher model and records its responses
3. Scores and compares responses to choose the primary teacher for Phase 02

## Order of execution

```bash
python src/build_teacher_test.py   # sample test slice, write to data/splits/
python src/generate_traces.py      # call each model, write to data/generated/
python src/compare_teachers.py     # score outputs, print ranking
```

## Config

`configs/models.yaml` — teacher model IDs, rate limits, and API settings.
