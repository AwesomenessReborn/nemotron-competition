# Dataset Version Changelog

Each version entry answers: what changed, why, what went wrong, and what it led to.
For full eval numbers see `docs/experiments/`.

---

## v7 — Claude Haiku 4.5 haiku_reasoning
**Status:** Production (current merged/ split)
**Date:** 2026-06-04
**Teacher:** Claude Haiku 4.5 via Anthropic API
**Format:** `haiku_reasoning` — short chain-of-thought reasoning traces

### What it is
First full-coverage dataset for all 9,500 competition rows. Haiku generates a
reasoning chain, then produces the final `\boxed{answer}`. The model trains on
`(problem → reasoning + answer)` pairs.

### Rows
| Split | Count | Path |
|-------|-------|------|
| Train | 8,550 | `phase02_data_generation/data/merged/train.jsonl` |
| Val   | 950   | `phase02_data_generation/data/merged/val.jsonl`   |
| Raw   | 9,500 | `phase02_data_generation/data/train_reasoning_v7_haiku.jsonl` |

### Eval results (Experiment 001)
n=102, n_per_task=17, max_new=250

| task_type       | parse  | accuracy |
|-----------------|--------|----------|
| roman           | 100%   | 100%     |
| cipher_text     | 94.1%  | 17.6%    |
| gravity         | 100%   | 0%       |
| unit_conversion | 100%   | 0%       |
| bit_manipulation| 64.7%  | 0%       |
| symbol_transform| 58.8%  | 0%       |
| **overall**     | **86.3%** | **19.6%** |

### What went wrong
- **Four task types at 0% accuracy.** Haiku writes short, approximate reasoning traces.
  For tasks requiring exact arithmetic (gravity, unit_conversion), Haiku guesses a plausible
  constant rather than deriving the exact value from examples. For rule-induction tasks
  (bit_manipulation, symbol_transform), Haiku proposes one hypothesis without verifying it.
  The model imitates the approximation pattern instead of learning to solve.
- **Low parse rate on bit_manipulation and symbol_transform.** Haiku occasionally
  produces malformed output structure on these tasks, causing `\boxed{}` to be missing.
- **Cipher_text truncation.** At max_new=250, the reasoning chain often runs out of
  tokens before writing `\boxed{}`.

### Led to
→ v8: switch to post-hoc rationale format with a stronger local teacher model

---

## v8 — Local Gemma 4 12B post-hoc rationale
**Status:** Trained; V8.1 supersedes for symbol_transform
**Date:** 2026-06-08
**Teacher:** Local Gemma 4 12B via llama.cpp raw `/completion` (not `/v1/chat/completions` —
that endpoint strips thinking tokens)
**Format:** post-hoc rationale — gold answer is given in the prompt; model writes reasoning
explaining why the answer is correct, then copies the answer into `"answer"` field

### What it is
The gold answer is provided upfront (`CORRECT_ANSWER: <gold>`), so the model focuses
entirely on producing high-quality reasoning rather than solving from scratch. This produces
longer, more specific reasoning chains at $0 API cost (fully local). Workers=8, n_predict=512
selected by concurrency benchmark (`bench_concurrency.py`): 1.71 rows/s, flat VRAM.

### Rows
| Split | Raw   | After cleaning | Path |
|-------|-------|----------------|------|
| Train | ~8,900 | 8,344 | `phase02_data_generation/data/v8/train_reasoning_v8_local_gemma_clean.jsonl` |
| Val   | ~480  | 436  | `phase02_data_generation/data/v8/val_reasoning_v8_local_gemma_clean.jsonl`   |

Cleaning removed 596 empty-reasoning rows (rows where the model returned a valid JSON
structure but left `"reasoning"` blank). These are bad SFT targets and were excluded rather
than replaced with generic text.

### Eval results (Experiment 003)
n=102, n_per_task=17, max_new=250

| task_type       | parse  | accuracy | Δ vs v7   |
|-----------------|--------|----------|-----------|
| roman           | 100%   | 100%     | 0%        |
| cipher_text     | 100%   | 35.3%    | **+17.7%** |
| bit_manipulation| 100%   | 5.9%     | +5.9%     |
| unit_conversion | 100%   | 5.9%     | +5.9%     |
| gravity         | 100%   | 0%       | 0%        |
| symbol_transform| 76.5%  | 0%       | 0%        |
| **overall**     | **96.1%** | **24.5%** | **+4.9%** |

Val loss: 0.344 (epoch 4 best) vs v7 baseline 0.477 — V8 data is a stronger training signal.

### What went wrong

**Symbol_transform repair row leakage (→ V8.1):**
During V8 generation, 355 train + 30 val symbol_transform rows were produced by a
deterministic fallback (`source=deterministic_symbol_copy_repair_v8`). All 385 rows share
an identical generic reasoning string:
> "The correct symbol sequence is provided for this post-hoc training trace, so I copy it exactly."

The model learned to emit this exact string at inference time. Because this reasoning
encodes no actual symbol mapping information, the model still cannot solve symbol_transform
puzzles. The phrase is detectable at inference from the training verbatim reproduction.

**Gravity and unit_conversion remain at 0%:**
Post-hoc rationale helps but the traces still vary in how they express the constant
derivation (`g = 2*d/t²`). The model hasn't converged on a consistent extraction strategy.
Likely needs longer `max_new` at eval (current=250) or traces that spell out the arithmetic
step-by-step more explicitly.

**Bit_manipulation at 5.9%:**
One success shows the pattern is learnable. Short one-sentence reasoning doesn't encode
the exact bit operation precisely enough in most rows.

### Led to
→ v8.1: regenerate the 385 repair rows with task-specific reasoning

---

## v8.1 — Symbol repair rows regenerated
**Status:** In progress (generation script written; llama.cpp server needed)
**Date:** 2026-06-09
**Teacher:** Local Gemma 4 12B (same as v8)
**Change:** Narrow fix — only the 385 symbol_transform repair rows are regenerated.
All other v8 rows are carried over unchanged.

### What changes
The 385 rows where `source == "deterministic_symbol_copy_repair_v8"` are sent back to
Gemma 4 12B with a stricter prompt that:
1. Explicitly instructs the model to name the actual character mapping rule from the examples
2. Bans a list of generic-copy phrases (`"copy it exactly"`, `"given answer"`, etc.)
3. Requires `answer == gold_answer` to accept a row

Rows that fail the banned-phrase check or produce a wrong answer are excluded entirely
(not replaced with another generic string).

### Rows
| Split | V8 count | Expected v8.1 | Path |
|-------|----------|---------------|------|
| Train | 8,344    | ~8,344 (−failed repairs) | `phase02_data_generation/data/v8/train_reasoning_v8_1_local_gemma_clean.jsonl` |
| Val   | 436      | ~436 (−failed repairs)   | `phase02_data_generation/data/v8/val_reasoning_v8_1_local_gemma_clean.jsonl`   |

Exact row counts depend on how many of the 385 repair rows Gemma accepts vs rejects.

### Script
`phase02_data_generation/src/regen_v8_1_symbol_repairs.py`
- `--dry-run`: print first prompt and exit (no server needed)
- `--workers N`: parallelism (default 8, matches llama.cpp slot count)
- Writes QA audit to `v8_1_local_gemma_clean_dataset_audit.{json,md}` on completion
- Do NOT train until audit passes

### Remaining open issues (not fixed in v8.1)
- Gravity 0%: exact g-constant derivation still not reliable
- Unit_conversion ~6%: same arithmetic precision issue
- Bit_manipulation ~6%: short reasoning traces; may need explicit operation enumeration

---

## Planned

### v9 — Solver-augmented traces (not started)
See `docs/experiments/002_v8_solver_trace_plan.md` for the full design.

For gravity and unit_conversion: Python solver fits exact constant analytically from
in-context examples, writes step-by-step arithmetic into the reasoning trace.

For bit_manipulation: solver enumerates candidate operations (NOT, rotate, XOR, shift
combinations), verifies each against all provided examples, writes the verification
chain into the trace.

Blocked on: v8.1 completing and being eval'd first to confirm symbol_transform is fixed
before starting a new generation pass.
