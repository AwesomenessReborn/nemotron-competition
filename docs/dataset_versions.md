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
**Status:** Complete
**Date:** 2026-06-09
**Teacher:** Local Gemma 4 12B (same as v8)
**Change:** Narrow fix — only the 385 symbol_transform repair rows are regenerated.
All other v8 rows are carried over unchanged.

### What changed
The 385 rows where `source == "deterministic_symbol_copy_repair_v8"` were sent back to
Gemma 4 12B with a stricter prompt requiring:
1. The model names the actual character-level mapping rule from the examples
2. No generic-copy phrases (`"copy it exactly"`, `"given answer"`, etc.) — hard banned
3. `answer == gold_answer` required to accept a row

**Outcome:** 88 of 385 rows accepted; 297 excluded (failed answer-match or banned phrase).
The repair row pool was much harder than expected — most symbol_transform puzzles have
ambiguous mapping rules that Gemma struggles to articulate precisely.

### Rows
| Split | V8 count | V8.1 count | Delta |
|-------|----------|-----------|-------|
| Train | 8,344    | ~8,077    | −267 (297 excl. − 88 regen accepted = net −267 repair rows) |
| Val   | 436      | ~418      | −18  |

Paths: `phase02_data_generation/data/v8/train_reasoning_v8_1_local_gemma_clean.jsonl`,
`phase02_data_generation/data/v8/val_reasoning_v8_1_local_gemma_clean.jsonl`

### Eval results (controlled 102-row fixed eval set)
| Metric | V8 | V8.1 | Δ |
|--------|-----|------|---|
| parse% | 96.1% | 98.0% | +1.9% |
| accuracy% | 22.5% | 21.6% | −0.9% |

| task_type | V8 acc | V8.1 acc | Δ |
|-----------|--------|---------|---|
| roman | 100% | 100% | 0% |
| cipher_text | 0% | 5.9% | +5.9% |
| bit_manipulation | 17.6% | 17.6% | 0% |
| unit_conversion | 11.8% | 0% | −11.8% |
| gravity | 0% | 0% | 0% |
| symbol_transform | 5.9% | 5.9% | 0% |

Symbol_transform accuracy improved from 0% to 5.9% (1/17). Repair-row leakage string
no longer appears at inference. Unit_conversion regressed to 0% — likely sampling variance
(small n=17). Overall accuracy slightly down due to UC regression.

### Led to
→ v8.2: add deterministic solver traces for gravity and unit_conversion

---

## v8.2 — Verbose solver traces for gravity + unit_conversion
**Status:** Complete (trained; eval confirms parse regression — do not use as base)
**Date:** 2026-06-09
**Base:** v8.1
**Change:** Replace all gravity and unit_conversion rows with deterministic Python solver
traces. Solver fits exact g-constant or conversion factor from in-context examples,
writes multi-step arithmetic into reasoning.

### What changed
gravity traces: `2d/t²` derivation written out per example + verification step (~93 words)
unit_conversion traces: slope+intercept fit with example verification (~60 words)
41 rows where solver could not reproduce gold_answer exactly were kept with original V8.1 reasoning.

### Rows
| Split | V8.1 count | V8.2 count |
|-------|-----------|-----------|
| Train | ~8,077    | ~8,077 (same row count, gravity+UC rows replaced in place) |
| Val   | ~418      | ~418 |

### Eval results (controlled 102-row fixed eval set, max_new=250)
| Metric | V8.1 | V8.2 | Δ |
|--------|------|------|---|
| parse% | 98.0% | 93.1% | −4.9% |
| accuracy% | 21.6% | 19.6% | −2.0% |

| task_type | V8.1 acc | V8.2 acc | Δ |
|-----------|---------|---------|---|
| gravity | 0% | 0% | 0% |
| unit_conversion | 0% | 0% | 0% |
| bit_manipulation | 17.6% | 11.8% | −5.9% |

### What went wrong
**Parse regression:** gravity parse dropped to 0% at max_new=250. All non-parsed rows
are repetition loops ("Let's recalculate: g = 10.55 × 13.5481 = 71.84. Let's recalculate: ..."),
not truncations. The verbose 93-word traces shifted the model's token distribution — it learned
to enter a recalculation loop rather than write `\boxed{}`.

**bit_manipulation regression:** Verbose gravity/UC traces diluted the bit_manipulation training
signal, likely via distribution shift toward longer arithmetic sequences.

**Arithmetic still wrong:** Even the 12 gravity rows that did parse produced completely wrong
arithmetic (e.g. "0.5 × 13.01 × 3.18² = 10.39" when correct value ≈ 65.8). The 4B model
cannot execute multi-digit floating-point arithmetic at inference time regardless of trace format.

### Led to
→ v8.3-lite: compact single-paragraph solver traces to fix parse regression

---

## v8.3-lite — Compact solver traces for gravity + unit_conversion
**Status:** Complete (trained + eval'd; gravity/UC arithmetic still wrong at 4B scale)
**Date:** 2026-06-10
**Base:** v8.1 (not v8.2 — verbose traces caused parse regression)
**Change:** Same deterministic solver logic as v8.2 but traces written as compact single
paragraphs: ~45 words for gravity, ~37 words for unit_conversion. Fits within max_new=250.

### What changed
gravity trace format: `"g = 2d/t² from examples: ex1→X, ex2→Y, avg g=Z. Answer: 0.5·Z·t²=A."`
unit_conversion trace format: `"Fit y=m·x+b: ex1→slope, ex2→confirm. Answer: m·q+b=A."`
41 rows where solver could not reproduce gold_answer: kept with V8.1 reasoning (same as V8.2).

### Rows
| Split | V8.1 count | V8.3-lite count |
|-------|-----------|----------------|
| Train | ~8,077    | 8,069 |
| Val   | ~418      | 414 |

### Eval results (controlled 102-row fixed eval set, max_new=250)
| Metric | V8 | V8.1 | V8.2 | V8.3-lite | Δ vs V8.1 |
|--------|----|------|------|-----------|-----------|
| parse% | 96.1% | 98.0% | 93.1% | 94.1% | −3.9% |
| accuracy% | 22.5% | 21.6% | 19.6% | 19.6% | −2.0% |

| task_type | V8.1 acc | V8.3-lite acc | Δ |
|-----------|---------|--------------|---|
| gravity | 0% | 0% | 0% |
| unit_conversion | 0% | 5.9% | +5.9% |
| bit_manipulation | 17.6% | 11.8% | −5.9% |
| symbol_transform | 5.9% | 0% | −5.9% |

### What went wrong
**Gravity parse still 70.6%** (5/17 produce repetition loops, not `\boxed{}`). Compact
traces helped vs V8.2 (parse 0%→70.6%) but loop behavior persists. The 12 parsed gravity
rows still have completely wrong arithmetic (hallucinated values).

**Accuracy flat across V8.1→V8.2→V8.3-lite:** 21.6% → 19.6% → 19.6%. The fundamental
blocker is that the 4B model cannot execute floating-point arithmetic at inference time.
The solver writes the correct answer into the training trace, but the model pattern-matches
the output format rather than learning to compute the value. Traces that contain the correct
answer at training time do not transfer the arithmetic computation ability to inference.

### Core finding
**Solver traces do not give the model arithmetic ability — they only teach output format.**
At 4B scale, NemotronH cannot reliably compute `2 × 9.12 / 1.31²` even when it has seen
thousands of correctly solved examples in training.

### Led to
→ Open question: cloud 30B training may have sufficient capacity for arithmetic. Alternatively,
skip gravity/UC improvement and focus on bit_manipulation with rule-enumeration traces.

---

## Planned

### v9 — Bit manipulation rule enumeration traces
For bit_manipulation: solver enumerates candidate operations (NOT, rotate, XOR, shift
combinations), verifies each against all provided examples, writes the verification
chain into the trace. Current 5.9% suggests the pattern is learnable at 4B.

See `docs/experiments/002_v8_solver_trace_plan.md` for the original design.
