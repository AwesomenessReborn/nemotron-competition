# Handoff — 2026-06-09T16:15:00Z

## Mode
Repo (git-grounded)

## Goal
Generate a high-quality V8 post-hoc rationale dataset using local Gemma 4 12B, train a NemotronH 4B LoRA adapter on it, evaluate it against the previous full-haiku-9500 baseline, and iterate toward a competitive Kaggle submission. Phase 2 (data generation) and the first V8 local smoke training run (phase 3) are now complete. The next phase is to address failure modes identified in the smoke eval — specifically the symbol_transform repair row leakage — before cloud/30B training.

## Current Status
**V8 LoRA smoke training and eval are complete.** The adapter at `phase03_local_smoke/outputs/adapters/v8_local_gemma_clean/final_adapter_haiku_reasoning` was trained for 5 epochs (63.3 min, RTX 5070 Ti) and evaluated on the same 102-sample stratified set used for the haiku baseline. Results: parse 96.1% (+9.8%), accuracy 24.5% (+4.9%). Cipher_text improved +17.7%. Gravity and symbol_transform remain at 0% accuracy. A repair-row reasoning leakage issue was identified in symbol_transform — 355 training rows use a fixed generic reasoning string ("The correct symbol sequence is provided for this post-hoc training trace, so I copy it exactly.") that the model learned to emit at inference time. User has not yet reviewed or approved next steps.

## Repo State
- **Directory:** `/home/hareee234/Dev/kaggle/nemotron-competition-may/nemotron-competition`
- **Branch:** `feat/v8-data-generation`
- **Git status:**
  ```
  M phase03_local_smoke/src/eval_chunked_full.py
  ?? docs/experiments/003_v8_local_gemma_clean.md
  ?? phase02_data_generation/data/v8/bench_concurrency_report.json
  ?? phase02_data_generation/data/v8/bench_w{4,16_n512,16_n768}_report.json
  ?? phase02_data_generation/data/v8/local_gemma_full_report.json
  ?? phase02_data_generation/data/v8/v8_local_gemma_{clean_,}dataset_audit.{json,md}
  ?? phase02_data_generation/data/v8/v8_local_gemma_clean_dataset_audit.{json,md}
  (data JSONL files are gitignored — correct)
  ```
- **Recent commits:**
  ```
  6f604de chore: add Google Drive backup scripts and rclone rules
  29c4367 docs: update HANDOFF with V8 generation plan and smoke test results
  2c6c1d7 feat: add generate_v8_local_gemma — full V8 dataset generation script
  5d0230a feat: add local Gemma 4 12B pilots and concurrency benchmark
  b6715ea feat: add Fireworks provider exploration scripts (pilot, repair, recovery)
  ```
- **Changed files:**
  - `phase03_local_smoke/src/eval_chunked_full.py` — added `BASELINE_FULL_HAIKU_9500` constant (parse 86.3%, acc 19.6% per-task numbers), swapped comparison print from BASELINE_1K to BASELINE_FULL_HAIKU_9500, added baseline to saved JSON output
  - `docs/experiments/003_v8_local_gemma_clean.md` — new experiment record (untracked)
  - `phase02_data_generation/data/v8/*.json` — benchmark and audit reports (untracked, data dir is gitignored for JSONL but JSON reports are not)
- **Tests / build / lint:** not checked

## Key Decisions
| Decision | Rationale | Alternatives Rejected |
|---|---|---|
| Use local Gemma 4 12B via raw `/completion` for V8 data | 100% parse, 96% copy on gate, $0 cost | Fireworks DeepSeek V4 Flash (90% good, $7.28/run, persistent cipher/sym failures) |
| workers=8 for full generation (not 16) | workers=16 benchmark showed -7% throughput regression vs workers=8 (1.59 vs 1.71 rows/s); did not meet 25% improvement threshold | workers=16 (slower due to VRAM/KV cache pressure) |
| n_predict=512 (not 768) | Smoke test showed 0 truncations at 512; 768 reduces throughput to 1.30 rows/s with no quality gain | n_predict=768 (slower, same quality) |
| Exclude 596 empty-reasoning rows (not add generic reasoning) | Empty reasoning rows are bad SFT targets; generic reasoning teaches shortcut/copy behavior | Add generic reasoning string |
| Patch val task_type from staging file | pandas 3.x groupby.apply() drops the groupby key column; staging has the correct task_type for all rows | Regenerate val split (unnecessary) |
| TARGET_MODULES_V2: Attn+MLP only, no Mamba | Mamba in_proj/out_proj have FP32/BF16 training-inference mismatch causing divergence at position 138 | Including Mamba (V1 config — caused degenerate generation) |
| Manual training loop (not TRL Trainer) | Unsloth patches Trainer.compute_loss and strips precomputed -100 labels, computing loss on all tokens | TRL SFTTrainer (incorrect masking) |
| Chunked (no-cache) generation for eval | selective_state_update diverges from training path due to FP32/BF16 Mamba kernel mismatch | Cached generation / model.generate() (broken for NemotronH) |

## Constraints and Preferences
- **Do NOT train LoRA** without explicit user approval per session
- **Do NOT commit** `.jsonl`, `.json` data files, `.csv` files, or `.claude/` directory
- **Do NOT use `/v1/chat/completions`** for local Gemma — raw `/completion` only
- **Do NOT use the sentinel symbol prompt** (`<ANSWER>` variants) — causes tag-bleed regressions
- **Do NOT push commits** without explicit user instruction
- **Do NOT start cloud/30B training** without explicit user approval
- **Do NOT run `rclone sync`** — use `rclone copy` only (sync deletes Drive-only data)
- Cost hard stop: $10 (applies if Fireworks used as fallback; moot for local runs)
- Training target modules: Attn+MLP only (Mamba excluded — permanent constraint per CLAUDE.md)
- Chunked (no-cache) generation is the only correct inference path for NemotronH

## Do Not Do
- Do NOT run `generate_v8_local_gemma.py` or any full generation run without explicit approval
- Do NOT train LoRA on unclean files — always use `*_clean.jsonl` variants
- Do NOT modify `repair_pilot_30.csv`, `rejected_v8_pool.csv`, or v7 haiku outputs
- Do NOT use `model.generate()` — broken for NemotronH due to KV cache bug
- Do NOT merge or start cloud training after smoke eval — user review required
- Do NOT commit generated JSONL/JSON data outputs or model adapter weights

## Open Questions / Risks
- **Symbol_transform repair row leakage (blocking for next run):** 355 training rows (source=`deterministic_symbol_copy_repair_v8`) use the fixed generic reasoning string "The correct symbol sequence is provided for this post-hoc training trace, so I copy it exactly." The model learned to emit this string at inference time, which provides no puzzle-solving signal. Fix options: (a) exclude these 355 rows from training, (b) regenerate their reasoning using a non-generic string, (c) replace with real reasoning traces. User has not decided.
- **Gravity 0% accuracy:** Model learns the d=0.5·g·t² formula but consistently extracts the wrong g constant from in-context examples. May require longer max_new at eval (current=250) or multi-step reasoning traces. Not yet tried.
- **Unit_conversion 1/17:** One success suggests the pattern is learnable. May benefit from same max_new increase. Not yet tried.
- **Val task_type bug in generate script:** `make_train_val_split()` in `generate_v8_local_gemma.py` uses `df.groupby("task_type").apply(...)` which drops the task_type column in pandas 3.x. Workaround (patch from staging) is in place for existing files, but the generate script itself is not fixed. If V8 is regenerated, the bug will recur.
- **llama.cpp server is stopped** — was killed to free VRAM for LoRA training. Must be restarted if any further generation or benchmarking is needed.

## Next Action
Address the symbol_transform repair row leakage before the next training run. The 355 repair rows are in `phase02_data_generation/data/v8/local_gemma_symbol_repairs.jsonl` and identifiable by `source == "deterministic_symbol_copy_repair_v8"` in the clean JSONL files. User decision needed: exclude them (simplest — reduces symbol_transform training data from 1,149 to ~794 rows) or replace their reasoning strings with task-specific content. Once decided, create a new `train_reasoning_v8_v2_clean.jsonl` and retrain. Do not train until user approves the fix strategy.
