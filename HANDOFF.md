# Handoff — 2026-06-03T21:00:00

## Mode
Repo (git-grounded)

## Goal
Determine the best data/training strategy for the Nemotron-H 4B LoRA competition entry before generating more teacher traces. Three small V2 experiments (1k train / 100 val, rank 32, 3 epochs) compared three assistant-target formats: `answer_only`, `short_reasoning`, and `haiku_reasoning`. The overarching objective is to maximize accuracy across 6 puzzle task types (bit_manipulation, gravity, unit_conversion, cipher_text, roman, symbol_transform) on the Kaggle competition.

## Current Status
All three variant experiments ran and were evaluated. A second critical bug was found and fixed: **cached generation (selective_state_update) is broken for all adapters** — the Mamba SSM state diverges during sequential inference regardless of which target format was used. The fix is chunked generation (`use_cache=False` at every step), which re-processes the full growing sequence. With chunked inference, `answer_only` achieved 97% parse / 18% accuracy (roman: 100%, all numerical: 0%), and `haiku_reasoning` achieved 88% parse / 12% accuracy (roman: 80%, unit_conv: 13%) at max_new=250. The `short_reasoning` variant failed completely even with chunked inference. The code is ready to train `haiku_reasoning` on all 9,500 rows.

## Repo State
- **Directory:** /home/hareee234/Dev/kaggle/nemotron-competition-may/nemotron-competition
- **Branch:** main (behind origin/main by 1 commit — not pushed this session)
- **Git status:** All new files are untracked (never committed). No modifications to tracked files.
- **Recent commits:**
  ```
  59c1b29 feat: add placeholder submission.zip for initial Kaggle submission (no-op 30B adapter)
  4160982 feat: add placeholder adapter generator for initial Kaggle submission
  00ff561 docs: expand README with detailed per-phase findings, adapter specs, and cloud training guide
  ae3c099 docs: update README with phase progress, 4B adapter details, and Phase 04 next steps
  009b672 feat: phase04 cloud training deployment scripts for 30B Nemotron
  ```
- **Changed files (untracked, all new this session):**
  - `phase03_local_smoke/src/train_lora_v2.py` — V2 training script; major updates: `--target-type` flag (answer_only / short_reasoning / haiku_reasoning), `format_for_training` moved inline with 3 variants, `nemotron_generate` replaced with chunked (no-cache) implementation, `BOX_RE` updated to match `\box(?:ed)?{`, post-train eval writes JSON summary per variant
  - `phase03_local_smoke/src/diag_chunked_gen.py` — diagnostic that confirmed chunked vs cached generation; tests both modes on any adapter + optional base model control
  - `phase03_local_smoke/src/diag_base_model.py` — diagnostic for base model generation (shows base model produces garbage without adapter)
  - `run_variant_experiments.sh` — runs all 3 target-type variants sequentially (1k rows, 3 epochs, rank 32)
  - `run_chunked_eval.sh` — evaluates all 3 saved adapters with chunked generation (100 samples, max_new=80)
  - `phase03_local_smoke/outputs/adapters/variants_1k/` — three saved adapters (answer_only, short_reasoning, haiku_reasoning), each with `final_adapter_<type>/` and `eval_summary_<type>.json`
  - `phase03_local_smoke/outputs/logs/` — full training and eval logs for all variants
- **Tests / build / lint:** not checked

## Key Decisions
| Decision | Rationale | Alternatives Rejected |
|---|---|---|
| Switch to chunked generation (use_cache=False at every step) | selective_state_update (sequential Mamba SSM at step 1+) diverges from training path → degenerate "the the the" / "Final answer \\ \\}}}" for all adapters | KV cache patch (already applied as Bug 1 fix), accepting cached generation |
| Recommend haiku_reasoning on full 9,500 rows | answer_only achieves 0% on all numerical tasks regardless of data size; reasoning is required to compute numerical answers | answer_only training, generating new Gemini/Haiku traces |
| Do not use short_reasoning variant | Truncated first-sentence targets are harder to learn than either full reasoning or answer-only; degenerates even with chunked inference | Keeping short_reasoning as a fallback |
| Dataset is 9,500 rows total (not 69,029) | wc -l showed 69,030 because prompts contain embedded newlines; pandas reads 9,500 actual rows | None — factual correction |
| Haiku data is complete (no more generation needed) | Haiku JSONL covers all 9,500 rows at 99.9% correctness, shuffled | Resuming Haiku generation, running Gemini forward-solve traces |
| BOX_RE updated to match \box{} and \boxed{} | After 3 epochs, model occasionally generates \box{ instead of \boxed{ (skips 'ed' token); liberal regex avoids false negatives | Strict \boxed{} only |

## Constraints and Preferences
- Only Attn+MLP LoRA is trainable: Mamba in_proj/out_proj excluded (FP32/BF16 kernel mismatch per Bugs 2-3)
- 0.507% trainable params at rank 32 (4 attn layers × 4 modules + 17 MLP layers × 2 modules = 50 linear)
- Manual training loop required (bypasses unsloth Trainer which strips -100 labels)
- Adapter save/load: must call `_fix_adapter_key_names()` after save, then `load_adapter_weights()` after reload (PEFT standard loading silently leaves B matrices at zero)
- RTX 5070 Ti (16GB VRAM) — BF16, no 4-bit, batch=1 + accum=4
- Chunked generation is O(n²) per step — slow but correct; cached generation is broken

## Do Not Do
- Do not use `selective_state_update` (cached) generation — it always degenerates
- Do not train `short_reasoning` variant — performs worse than both answer_only and haiku_reasoning
- Do not train answer_only for numerical task accuracy — pattern memorization only gives roman numerals
- Do not generate new Haiku/Gemini traces — 9,500 rows at 99.9% quality already covers the full dataset
- Do not use `model.generate()` — broken for NemotronH (KV cache bug in modeling_nemotron_h.py)
- Do not call `_patch_hybrid_cache` / `_patch_block_forward` before chunked generation — unnecessary for use_cache=False

## Open Questions / Risks
- haiku_reasoning at 1k rows / 3 epochs gives 88% parse / 12% accuracy; will accuracy scale with full 9,500 rows and 5 epochs? — expected yes, but untested
- gravity / bit_manipulation / cipher_text / symbol_transform show 0% at 1k rows — harder tasks or just need more data? — not yet determined
- Chunked generation is O(n²): inference time at competition scale not measured
- PEFT "missing keys" warning on every load is expected and harmless (load_adapter_weights() handles it)

## Next Action
Run full haiku_reasoning training on all 9,500 rows for 5 epochs with chunked generation:
```bash
cd /home/hareee234/Dev/kaggle/nemotron-competition-may/nemotron-competition
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
After training, evaluate with `diag_chunked_gen.py --max-new 250 --n 100` against all 6 task types to confirm accuracy improvement.
