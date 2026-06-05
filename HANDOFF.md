# Handoff — 2026-06-04T19:00:00

## Mode
Repo (git-grounded)

## Goal
Maximize accuracy across 6 puzzle task types (bit_manipulation, gravity, unit_conversion, cipher_text, roman, symbol_transform) on the Kaggle Nemotron-H 4B LoRA competition. Phase 03 full-dataset training (`full_haiku_9500`) is done and evaluated. Next phase is V8 solver generation and/or DeepSeek pilot to attack the 0%-accuracy tasks.

## full_haiku_9500 Baseline (current best adapter)

**Adapter:** `phase03_local_smoke/outputs/adapters/full_haiku_9500/final_adapter_haiku_reasoning`
**Eval file:** `phase03_local_smoke/outputs/evals/eval_final_adapter_haiku_reasoning_20260604_124340.json`
**Config:** 9,500 rows, haiku_reasoning target, rank 32, 5 epochs, max_new=250, chunked generation (use_cache=False)

| Metric | Value |
|---|---|
| Overall parse (boxed%) | **86.3%** (88/102 samples) |
| Overall accuracy | **19.6%** (20/102 samples) |

| Task | Parse% | Accuracy% | Notes |
|---|---|---|---|
| roman | 100% | **100%** | Solved — all 17/17 correct |
| cipher_text | 94.1% | **17.6%** | Partial — 3/17 correct; truncation failures |
| bit_manipulation | 64.7% | **0%** | Fails to infer rule; generates plausible-but-wrong patterns |
| gravity | 100% | **0%** | Reasoning step correct but arithmetic off; off-by-small-amount |
| symbol_transform | 58.8% | **0%** | Rule induction fails; often extracts partial output |
| unit_conversion | 100% | **0%** | Identifies wrong multiplier; answer close but not exact |

### Main Failure Modes
1. **Arithmetic precision (gravity, unit_conversion):** Model infers the correct formula structure but uses an approximate multiplier. Off-by-1–3% on final value — exact string match fails. Requires solver-augmented traces or higher numerical precision training.
2. **Rule induction failures (bit_manipulation, symbol_transform):** Generates a plausible transformation hypothesis that fits some examples but not the query. No symbolic/exact reasoning.
3. **Output truncation (cipher_text):** Parse rate 94% but 82% of incorrect are empty-boxed (generation cuts off mid-sentence before `\boxed{}`). Low parse + truncation = context exhausted at max_new=250.

## Repo State
- **Directory:** /home/hareee234/Dev/kaggle/nemotron-competition-may/nemotron-competition
- **Branch:** main
- **Recent commits:**
  ```
  59c1b29 feat: add placeholder submission.zip for initial Kaggle submission (no-op 30B adapter)
  4160982 feat: add placeholder adapter generator for initial Kaggle submission
  00ff561 docs: expand README with detailed per-phase findings, adapter specs, and cloud training guide
  ae3c099 docs: update README with phase progress, 4B adapter details, and Phase 04 next steps
  009b672 feat: phase04 cloud training deployment scripts for 30B Nemotron
  ```
- **New/modified files (pending commit):**
  - `phase03_local_smoke/src/train_lora_v2.py` (modified) — V2 training script with `--target-type`, chunked generation, BOX_RE fix
  - `phase03_local_smoke/src/eval_chunked_full.py` (untracked) — full eval script used to generate the baseline above
  - `phase02_data_generation/src/generate_llm.py` (untracked) — LLM generation helpers
  - `run_variant_experiments.sh`, `run_chunked_eval.sh` — orchestration scripts
  - `phase03_local_smoke/src/compare_lora_before_after_v2.py` (moved from root)
  - `notebooks/nvidia-nemotron-my-train.ipynb` (moved from root)

## Key Decisions
| Decision | Rationale | Alternatives Rejected |
|---|---|---|
| Switch to chunked generation (use_cache=False) | selective_state_update (cached Mamba SSM) diverges from training path → degenerate outputs for all adapters | KV cache patch (Bug 1 fix), accepting cached generation |
| haiku_reasoning on full 9,500 rows | answer_only gives 0% on all numerical tasks; reasoning traces required to compute answers | answer_only, short_reasoning, new Gemini traces |
| Do not use short_reasoning | Truncated first-sentence targets degenerate even with chunked inference | short_reasoning as fallback |
| Dataset is 9,500 rows (not 69,029) | Embedded newlines; pandas reads 9,500 actual rows | None — factual correction |
| Haiku data is complete | JSONL covers all 9,500 rows at 99.9% correctness | Resuming Haiku generation |
| BOX_RE matches \box{} and \boxed{} | After 3 epochs, model skips 'ed' token; liberal regex avoids false negatives | Strict \boxed{} only |

## Constraints and Preferences
- Only Attn+MLP LoRA is trainable: Mamba in_proj/out_proj excluded (FP32/BF16 kernel mismatch)
- 0.507% trainable params at rank 32 (4 attn layers × 4 modules + 17 MLP layers × 2 modules = 50 linear)
- Manual training loop required (bypasses unsloth Trainer which strips -100 labels)
- Adapter save/load: must call `_fix_adapter_key_names()` after save, then `load_adapter_weights()` after reload
- RTX 5070 Ti (16GB VRAM) — BF16, no 4-bit, batch=1 + accum=4
- Chunked generation is O(n²) per step — slow but correct; cached generation is broken

## Do Not Do
- Do not use `selective_state_update` (cached) generation — always degenerates
- Do not train `short_reasoning` variant
- Do not train `answer_only` for numerical tasks
- Do not generate new Haiku/Gemini traces — 9,500 rows at 99.9% quality
- Do not use `model.generate()` — broken for NemotronH (KV cache bug)
- Do not call `_patch_hybrid_cache` / `_patch_block_forward` before chunked generation

## Open Questions / Risks
- gravity/unit_conversion fail due to arithmetic precision — can V8 solver traces fix this?
- bit_manipulation/symbol_transform need rule-induction reasoning — more examples or different trace format?
- cipher_text truncation — increase max_new or reformat to put answer earlier?
- Chunked generation O(n²): competition inference time at full context not measured

## Next Action
Design and generate V8 solver traces to address the 0%-accuracy tasks. Key questions:
1. For gravity/unit_conversion: generate traces that solve via exact arithmetic (no approximation)
2. For bit_manipulation/symbol_transform: generate traces that enumerate pattern candidates systematically
3. For cipher_text: consider shorter traces or answer-first format to avoid truncation

Data target: `phase02_data_generation/data/v8/` / `phase02_data_generation/outputs/v8/`
