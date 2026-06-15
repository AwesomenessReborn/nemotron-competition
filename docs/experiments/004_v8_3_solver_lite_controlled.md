# 004 — V8.3-lite Controlled Eval (max_new=250)

## Adapter
`phase03_local_smoke/outputs/adapters/v8_3_solver_lite_clean/final_adapter_haiku_reasoning`

## Dataset
- Train: `phase02_data_generation/data/v8/train_reasoning_v8_3_solver_lite_clean.jsonl` (8069 rows)
- Val: `phase02_data_generation/data/v8/val_reasoning_v8_3_solver_lite_clean.jsonl` (414 rows)

## Training settings
rank=32, epochs=5, lr=2e-4, batch=1, accum=4, max-seq=2048, target-type=haiku_reasoning
Training time: 66.4 min on RTX 5070 Ti

## Eval command
```
python phase03_local_smoke/src/eval_v8_3_controlled.py
```
Controlled 102-row eval (same fixed IDs as V8/V8.1/V8.2), val source: `val_reasoning_v8_1_local_gemma_clean.jsonl`
max_new=250, chunked no-cache generation

## Results

### Parse %
| V8 | V8.1 | V8.2 | V8.3-lite |
|----|------|------|-----------|
| 96.1% | 98.0% | 93.1% | 94.1% |

### Accuracy %
| V8 | V8.1 | V8.2 | V8.3-lite |
|----|------|------|-----------|
| 22.5% | 21.6% | 19.6% | 19.6% |

### Per-task accuracy
| Task | V8 | V8.1 | V8.2 | V8.3-lite | parse |
|------|----|------|------|-----------|-------|
| bit_manipulation | 17.6% | 17.6% | 11.8% | 11.8% | 100% |
| cipher_text | 0% | 5.9% | 0% | 0% | 100% |
| gravity | 0% | 0% | 0% | 0% | 70.6% |
| roman | 100% | 100% | 100% | 100% | 100% |
| symbol_transform | 5.9% | 5.9% | 5.9% | 0% | 94.1% |
| unit_conversion | 11.8% | 0% | 0% | 5.9% | 100% |

## Raw eval JSON
`phase03_local_smoke/outputs/evals/v8_3_lite_controlled_max250_results.json`

## Report
`phase03_local_smoke/outputs/evals/v8_3_lite_controlled_max250_report.md`

## Key findings

**UC parse fixed, arithmetic still wrong:** V8.3-lite compact traces fixed unit_conversion parse rate to 100% (was 0% in V8.2). The model correctly adopts `y = m·x + b` formula structure. However slope estimation is slightly off (~1-3%), giving wrong final answers. Only 1/17 UC correct.

**Gravity parse regressed to 70.6%:** 5 of 17 gravity rows produce no boxed answer. All 5 are repetition loops ("Let's recalculate: g = 10.55 × 13.5481 = 71.84. Let's recalculate: ..."), not truncations. The compact `g = 2d/t²` pattern was not cleanly learned. The model generates "compute g = g * t^2, compute g = 1.14..." — a confused circular formula. Even the 12 parsed rows have completely wrong arithmetic (e.g., "0.5 × 13.01 × 3.18² = 10.39" when 0.5 × 13.01 × 10.11 ≈ 65.8).

**max_new=500 not run:** All gravity non-parsed rows confirmed as loops. More budget would not fix them.

**Overall trend:** Accuracy is flat across V8.1→V8.2→V8.3-lite (21.6% → 19.6% → 19.6%). The fundamental blocker is arithmetic — the 4B model cannot reliably compute floating-point expressions at inference time regardless of trace format. The solver-trace approach fixes parse format but does not give the model numerical computation ability.
