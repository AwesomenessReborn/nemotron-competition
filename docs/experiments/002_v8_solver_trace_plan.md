# Experiment 002 — V8 Solver Trace Plan

**Date:** 2026-06-04
**Status:** Planning — not started

## Motivation

Experiment 001 (`full_haiku_9500`) established that four of six task types remain at 0% accuracy with haiku_reasoning traces on 9,500 rows:
- gravity (arithmetic precision)
- unit_conversion (arithmetic precision)
- bit_manipulation (rule induction)
- symbol_transform (rule induction)

Cipher_text improved to 17.6% but is bottlenecked by output truncation. Roman is solved (100%).

More data in the same format will not fix these. The traces themselves must change.

## Core idea

Replace approximate LLM-generated reasoning with **deterministic solver traces**: Python-computed intermediate steps embedded directly into the reasoning chain. The model trains on traces where every arithmetic step and every rule verification is correct by construction.

The final artifact is still a LoRA adapter. There is **no runtime solver** in the final Kaggle submission — the solver is used only at data-generation time to write better training traces.

## Target data path

```
phase02_data_generation/data/v8/          raw solver traces (gitignored *.jsonl)
phase02_data_generation/outputs/v8/       generation logs and stats (gitignored)
```

Final merged split (when ready): overwrite `phase02_data_generation/data/merged/` and tag commit `dataset-v8`.

## Task-by-task trace strategy

### gravity and unit_conversion — exact arithmetic

**Problem:** Haiku guesses an approximate multiplier. Model imitates the approximation.

**Trace strategy:**
1. Solver reads the provided (input, output) examples for the problem.
2. Fits the exact constant (gravity `g`, or conversion factor `k`) by solving the formula analytically using the examples (e.g. `g = 2*d/t²` for each example, then average/mode if consistent).
3. Verifies the fitted constant reproduces all training examples exactly (within float tolerance).
4. Writes the reasoning: "fitting g from example 1: ..., from example 2: ..., consistent: yes. Applying to t=X: d = 0.5 * g * t² = Y."
5. Final `\boxed{Y}` matches the gold answer.

Reject any row where the solver cannot fit a consistent constant — queue for DeepSeek pilot.

### bit_manipulation and symbol_transform — rule enumeration

**Problem:** Haiku proposes one hypothesis without verifying it. Model imitates the pattern of guessing.

**Trace strategy:**
1. Solver enumerates a candidate rule set (rotation, NOT, XOR, reversal, shift combinations for bit_manipulation; character lookup tables for symbol_transform).
2. For each candidate, checks whether it correctly transforms all provided (input, output) examples.
3. First rule that passes all examples is selected.
4. Trace writes out: "Candidate: NOT+left-rotate-1. Check example 1: input→expected, got X. ✓. Check example 2: ... ✓. Rule confirmed. Applying to query: ..."
5. Final `\boxed{answer}`.

If no candidate in the enumerated set passes all examples, the row is rejected and queued for DeepSeek pilot.

### cipher_text — truncation fix

**Problem:** Reasoning chain exhausts max_new=250 before writing `\boxed{}`.

**Trace strategy (two options, pick one at generation time):**

Option A — shorten traces: generate a compact reasoning chain (decode mapping as a list, apply mechanically, answer). Target ≤150 tokens for the assistant turn.

Option B — answer-first format: write `\boxed{answer}` first, then optionally add explanation. Model learns to emit the box early.

Recommendation: Option A first (less format change). Fall back to Option B if parse rate does not improve.

### roman — no change needed

100% accuracy at 001. Keep existing haiku_reasoning traces for roman in the v8 dataset.

## DeepSeek pilot

Rows rejected by the solver (inconsistent constants, unknown rules) will be sent to DeepSeek for trace generation. DeepSeek has stronger mathematical reasoning than Haiku and is more likely to write exact arithmetic.

Scope: rejected rows only — not a full re-generation. Estimate 5–20% of numerical-task rows.

DeepSeek traces will be validated by the solver before inclusion: the extracted `\boxed{}` answer must match gold.

Script to modify: `phase02_data_generation/src/generate_llm.py` — add `--provider deepseek` flag.

## What is NOT changing

- LoRA adapter is still the only trainable artifact
- No runtime solver in the Kaggle submission kernel
- No change to model architecture or training loop
- Chunked generation remains mandatory for inference
- The v7 dataset (`merged/`) is not modified

## Expected outputs

| Artifact | Path |
|---|---|
| Solver trace generator | `phase02_data_generation/src/generate_solver_traces.py` (new — distinct purpose and output contract) |
| V8 raw JSONL | `phase02_data_generation/data/v8/*.jsonl` (gitignored) |
| V8 stats/logs | `phase02_data_generation/outputs/v8/` (gitignored) |
| Merged v8 split | `phase02_data_generation/data/merged/` (overwrite when ready, tag commit) |
| Experiment record | `docs/experiments/003_v8_solver_full.md` (when training is done) |

## Success criteria

A V8-trained adapter should show:
- gravity: >50% accuracy (currently 0%)
- unit_conversion: >50% accuracy (currently 0%)
- bit_manipulation: >30% accuracy (currently 0%)
- symbol_transform: >20% accuracy (currently 0%)
- cipher_text: parse rate >90% and accuracy >20% (currently 94.1% parse / 17.6% acc)
- roman: maintain 100% accuracy
