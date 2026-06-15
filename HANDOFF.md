# Handoff — 2026-06-10T00:00:00Z

## Mode
Repo (git-grounded)

## Goal
Iteratively improve NemotronH 4B LoRA accuracy on a 6-task competition benchmark (roman, bit_manipulation, cipher_text, gravity, unit_conversion, symbol_transform). The current focus is on fixing the two zero-accuracy tasks — gravity and unit_conversion — by replacing LLM-generated reasoning traces with deterministic solver-generated traces, then training and evaluating to see if the model can learn the correct rule-extraction approach.

## Current Status
Three dataset versions and two trained adapters exist. V8.3-lite dataset has been built and audited (PASS) but **not yet trained**. The core finding from V8.2 training is that verbose solver traces caused max_new=250 parse truncation and training distribution shift. V8.3-lite addresses this with compact 45-word (gravity) / 37-word (UC) traces, well within the token budget. Gravity and UC remain at 0% accuracy on all trained adapters so far — the model learns the correct formula structure but cannot execute floating-point arithmetic reliably at 4B scale. The next step is to train V8.3-lite and evaluate whether the shorter traces fix parse while preserving the formula-learning signal.

## Repo State
- **Directory:** /home/hareee234/Dev/kaggle/nemotron-competition-may/nemotron-competition
- **Branch:** feat/v8-data-generation
- **Git status:**
  ```
   M phase03_local_smoke/src/eval_chunked_full.py
  ?? phase02_data_generation/src/build_v8_2_solver_augmented.py
  ?? phase02_data_generation/src/build_v8_3_solver_lite.py
  ?? phase02_data_generation/src/regen_v8_1_symbol_repairs.py
  ?? phase03_local_smoke/src/eval_compare_v8_v8_1.py
  ?? phase03_local_smoke/src/eval_v8_2_controlled.py
  ```
- **Recent commits:**
  ```
  bfa8c15 docs: add dataset version changelog and fix stale REPO_CONVENTIONS
  7413616 docs: update README to V8 state and tighten gitignore
  f3ff241 feat: V8 local Gemma clean LoRA — training complete, 24.5% accuracy
  6f604de chore: add Google Drive backup scripts and rclone rules
  29c4367 docs: update HANDOFF with V8 generation plan and smoke test results
  ```
- **Changed files:**
  - `phase03_local_smoke/src/eval_chunked_full.py` — modified: added V8 local Gemma baseline constants, updated per-task print to show Δ vs both haiku_9500 and V8 baselines, added baseline to saved JSON
  - `phase02_data_generation/src/build_v8_2_solver_augmented.py` — new: builds V8.2 verbose solver-trace dataset (gravity + UC replaced, ~93-word gravity traces)
  - `phase02_data_generation/src/build_v8_3_solver_lite.py` — new: builds V8.3-lite compact solver-trace dataset (same solver logic, ~45-word gravity traces, ~37-word UC traces)
  - `phase02_data_generation/src/regen_v8_1_symbol_repairs.py` — new: regenerated 355 symbol_transform repair rows; 88 accepted, 297 excluded
  - `phase03_local_smoke/src/eval_compare_v8_v8_1.py` — new: controlled same-eval-set comparison script (V8 vs V8.1, same 102 fixed rows)
  - `phase03_local_smoke/src/eval_v8_2_controlled.py` — new: 3-way comparison eval (V8 vs V8.1 vs V8.2); reuse for V8.3-lite by updating V82_ADAPTER path and MAX_NEW; has versioned output filenames
- **Tests / build / lint:** not checked

## Key Decisions
| Decision | Rationale | Alternatives Rejected |
|---|---|---|
| Use V8.1 as base for V8.3-lite (not V8.2) | V8.2 verbose traces caused gravity parse=0% at max_new=250; V8.1 is the clean LLM-reasoned base | Starting from V8.2 |
| Compact single-paragraph solver traces for V8.3-lite | V8.2 gravity p95=104 words caused truncation before \boxed{}. V8.3-lite gravity=45w, UC=37w fits max_new=250 | Multi-step verbose traces (V8.2 style) |
| Keep original V8.1 row when solver cannot reproduce gold_answer exactly | Do not insert incorrect reasoning; 41 rows kept | Excluding failing rows, blanking reasoning |
| Chunked (no-cache) generation for all NemotronH inference | selective_state_update KV cache diverges from training path at inference position ~138 | model.generate() — broken for NemotronH |
| Attn+MLP LoRA only (no Mamba) | Mamba layers: FP32/BF16 kernel mismatch causes inference divergence | Full-model LoRA including Mamba |
| Controlled same-eval-set comparison fixed at 102 rows from v8_vs_v8_1_same_eval_ids.json | V8 and V8.1 used different val files, making per-task comparison invalid | Accepting per-run eval as comparable |

## Constraints and Preferences
- **Do NOT train LoRA without explicit user approval per session**
- **Do NOT commit .jsonl, .json data files, .csv files, or .claude/ directory**
- **Do NOT use /v1/chat/completions for local Gemma — raw /completion endpoint only**
- **Do NOT push commits without explicit user instruction**
- **Do NOT start cloud/30B training without explicit user approval**
- **Do NOT run rclone sync — use rclone copy only**
- **Do NOT use model.generate() for NemotronH — chunked no-cache only**
- **Do NOT train on unclean files — always use *_clean.jsonl variants**
- **Do NOT merge or start cloud training after smoke eval — user review required**
- Training target modules: Attn+MLP only (q_proj, k_proj, v_proj, o_proj, up_proj, down_proj). Mamba excluded permanently.
- conda env: `/home/hareee234/miniconda3/envs/nemotron-train/bin/python3`
- llama.cpp server (Gemma generation): `/home/hareee234/Dev/tools/llamacpp/llama.cpp/build/bin/llama-server`; model at `~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/.../*.gguf`

## Do Not Do
- Do not run `model.generate()` for NemotronH — use chunked no-cache loop always
- Do not train on V8.2 files — verbose traces caused parse regression
- Do not overwrite V8, V8.1, V8.2, or V8.3-lite adapter directories or data files
- Do not use rclone sync — copy only
- Do not commit model artifacts, .jsonl/.json data outputs, or adapter weights
- Do not start cloud/30B training
- Do not evaluate with only max_new=250 for V8.3-lite — run both 250 and 500 to confirm parse recovery

## Open Questions / Risks
- **Will V8.3-lite solve gravity/UC accuracy?** — V8.2 proved the model learns the formula structure but hallucinates arithmetic values (model says `2×9.12/1.31²=9.8444`, correct is 10.6288). Compact traces show correct numerical answer in the reasoning; model may pattern-match rather than compute. Outcome unknown until training.
- **bit_manipulation regression risk** — V8.2 saw bit_manipulation drop 17.6%→11.8% due to longer gravity/UC traces shifting token distribution. V8.3-lite traces are shorter (45w vs 93w gravity), expected to reduce this risk. Unconfirmed.
- **unit_conversion sampling variance** — V8 had 11.8% UC accuracy; V8.1/V8.2 both dropped to 0%. V8.3-lite uses same UC solver, compact format. Accuracy outcome uncertain.
- **Controlled eval source** — The 102 fixed IDs in `phase03_local_smoke/outputs/evals/v8_vs_v8_1_same_eval_ids.json` come from `val_reasoning_v8_1_local_gemma_clean.jsonl`. Continue using that file as eval source for controlled comparisons, even for V8.3-lite, so results are directly comparable across all adapter versions.

## Next Action
Train V8.3-lite LoRA (requires explicit user approval per session). Command:

```bash
nohup /home/hareee234/miniconda3/envs/nemotron-train/bin/python3 \
    phase03_local_smoke/src/train_lora_v2.py \
    --train phase02_data_generation/data/v8/train_reasoning_v8_3_solver_lite_clean.jsonl \
    --val   phase02_data_generation/data/v8/val_reasoning_v8_3_solver_lite_clean.jsonl \
    --output phase03_local_smoke/outputs/adapters/v8_3_solver_lite_clean \
    --rank 32 --epochs 5 --lr 2e-4 --batch 1 --accum 4 --max-seq 2048 \
    --target-type haiku_reasoning \
    >> phase03_local_smoke/outputs/logs/v8_3_solver_lite_clean_train.log 2>&1 &
```

After training, run controlled 3-way eval by updating `eval_v8_2_controlled.py`:
- `V82_ADAPTER` → `phase03_local_smoke/outputs/adapters/v8_3_solver_lite_clean/final_adapter_haiku_reasoning`
- `MAX_NEW = 250` first, then re-run with `MAX_NEW = 500`
- Output filenames will auto-version via the `maxnew{MAX_NEW}` suffix already in the script

Target comparison: V8 (96.1%/22.5%), V8.1 (98.0%/21.6%), V8.2 (93.1%/19.6% at max_new=500).
