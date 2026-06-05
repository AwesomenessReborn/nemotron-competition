# Experiment 001 — full_haiku_9500

**Date:** 2026-06-04
**Status:** Complete

## What this experiment tested

Full-dataset LoRA training with `haiku_reasoning` target format on all 9,500 rows, following the variant comparison (experiments at 1k rows). Primary question: does accuracy on numerical tasks (gravity, unit_conversion, cipher_text, bit_manipulation, symbol_transform) improve with 9.5× more data?

## Adapter

```
phase03_local_smoke/outputs/adapters/full_haiku_9500/final_adapter_haiku_reasoning/
```

## Dataset

```
phase02_data_generation/data/merged/train.jsonl   (9,025 rows, haiku_reasoning format)
phase02_data_generation/data/merged/val.jsonl     (475 rows, stratified by task_type)
```

Dataset version: v7 (Haiku traces, 9,500 total rows, 99.9% correctness, shuffled).

## Training settings

| Parameter | Value |
|---|---|
| Script | `phase03_local_smoke/src/train_lora_v2.py` |
| `--target-type` | `haiku_reasoning` |
| `--rank` | 32 |
| `--epochs` | 5 |
| `--lr` | 2e-4 |
| `--batch` | 1 |
| `--accum` | 4 |
| `--max-seq` | 2048 |
| Trainable modules | Attn q/k/v/o + MLP gate/up (50 linear layers, 0.507% of params) |
| Hardware | RTX 5070 Ti, 16 GB VRAM, BF16 |
| Generation mode | Chunked (`use_cache=False`) — cached generation is broken for NemotronH |

Training command:
```bash
/home/hareee234/miniconda3/envs/nemotron-train/bin/python3.11 \
  phase03_local_smoke/src/train_lora_v2.py \
  --train phase02_data_generation/data/merged/train.jsonl \
  --val   phase02_data_generation/data/merged/val.jsonl \
  --output phase03_local_smoke/outputs/adapters/full_haiku_9500 \
  --target-type haiku_reasoning \
  --rank 32 --epochs 5 --lr 2e-4 --batch 1 --accum 4 \
  --max-seq 2048 \
  2>&1 | tee phase03_local_smoke/outputs/logs/full_haiku_9500.log
```

## Eval command

```bash
/home/hareee234/miniconda3/envs/nemotron-train/bin/python3.11 \
  phase03_local_smoke/src/eval_chunked_full.py \
  --adapter-dir phase03_local_smoke/outputs/adapters/full_haiku_9500/final_adapter_haiku_reasoning \
  --val phase02_data_generation/data/merged/val.jsonl \
  --n-per-task 17 --max-new 250 \
  --output-dir phase03_local_smoke/outputs/evals
```

## Eval results

**Raw eval JSON:** `phase03_local_smoke/outputs/evals/eval_final_adapter_haiku_reasoning_20260604_124340.json`

### Overall

| Metric | Value |
|---|---|
| Samples | 102 (17 per task × 6 tasks) |
| Parse rate (boxed%) | **86.3%** (88/102) |
| Accuracy | **19.6%** (20/102) |
| Elapsed | 8.1 min |

### Per-task

| Task | Parse% | Accuracy% | Correct/N | Failure mode |
|---|---|---|---|---|
| roman | 100% | **100%** | 17/17 | None — fully solved |
| cipher_text | 94.1% | **17.6%** | 3/17 | Truncation: generation cuts off before `\boxed{}` at max_new=250 |
| bit_manipulation | 64.7% | **0%** | 0/17 | Rule induction: plausible-but-wrong hypothesis; 35% fail to produce boxed output |
| gravity | 100% | **0%** | 0/17 | Arithmetic precision: formula structure correct, multiplier approximate (off ≈2%) |
| symbol_transform | 58.8% | **0%** | 0/17 | Rule induction: partial-output extraction; 41% fail to produce boxed output |
| unit_conversion | 100% | **0%** | 0/17 | Arithmetic precision: wrong multiplier; answer close but not exact |

### Named failure modes

**1. Arithmetic precision (gravity, unit_conversion)**
Model identifies the correct relationship (kinematic equation, unit ratio) but computes an approximate multiplier from the training examples. The extracted answer is off by 1–3% relative to the gold. Exact-string match fails. Fix path: solver-augmented traces that compute the exact multiplier and verify against all training examples before writing the answer.

**2. Rule induction failure (bit_manipulation, symbol_transform)**
Model generates a plausible transformation hypothesis consistent with some examples but not the query. No symbolic search — it proposes one hypothesis and runs with it. Fix path: traces that enumerate candidate rules, verify each against all provided examples, and only commit when one rule passes all checks.

**3. Output truncation (cipher_text)**
94.1% parse rate but most incorrect outputs have an empty box. Generation reaches max_new=250 tokens mid-reasoning before writing `\boxed{}`. Fix path: increase max_new, or reformat traces to front-load the answer (answer-first format), or shorten the reasoning chain.

## Comparison to 1k-row baseline

| Metric | 1k/3ep | 9500/5ep | Delta |
|---|---|---|---|
| Parse rate | 88.0% | 86.3% | −1.7pp |
| Accuracy | 12.0% | 19.6% | +7.6pp |
| roman | 80% | 100% | +20pp |
| cipher_text | 0% | 17.6% | +17.6pp |
| unit_conversion | 13% | 0% | −13pp (regression) |
| gravity | 0% | 0% | 0 |
| bit_manipulation | 0% | 0% | 0 |
| symbol_transform | 0% | 0% | 0 |

Unit_conversion regressed — likely the model learned a better approximate but still wrong multiplier. The underlying issue (no exact arithmetic) was not addressed by more data alone.

## Conclusion

More data improved roman (solved) and unlocked cipher_text. The four remaining 0%-accuracy tasks all require a different fix — either exact arithmetic in the traces (gravity, unit_conversion) or systematic rule-enumeration (bit_manipulation, symbol_transform). More data with the same trace format will not close these gaps. Next step: V8 solver traces (see `002_v8_solver_trace_plan.md`).
