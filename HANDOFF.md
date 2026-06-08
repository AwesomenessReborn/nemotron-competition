# Handoff — 2026-06-08T21:38:00Z

## Mode
Repo (git-grounded)

## Goal
Generate all 9,500 V8 post-hoc rationale training rows for the Kaggle Nemotron competition using local Gemma 4 12B via llama.cpp raw `/completion`. The JSONL output (train + val split) becomes the dataset for NemotronH 4B LoRA fine-tuning. Phase 2 is not complete until all 9,500 rows are generated, validated (parse_success=True AND answer_correct=True or deterministic repair applied), and the final report is reviewed by the user.

## Current Status
**Waiting for user to approve the full 9,500-row run.** A 20-row workers=8 smoke test passed (20/20 GOOD, 0 failures, 0 truncations, VRAM flat at 9,306 MB / 16,303 MB, 1.71 rows/s → ~1.55h projected). The server was reconfigured to `--parallel 8` (8 slots × n_ctx=512). Staging/train/val files from the 20-row test are present in `data/v8/` and **must be deleted** before the full run to prevent the resume logic from skipping the first 20 rows. The full generation script `generate_v8_local_gemma.py` is written, tested, and ready.

## Repo State
- **Directory:** `/home/hareee234/Dev/kaggle/nemotron-competition-may/nemotron-competition`
- **Branch:** `feat/v8-data-generation`
- **Git status:**
  ```
  M HANDOFF.md
  M phase02_data_generation/src/generate_llm.py
  ?? phase02_data_generation/src/bench_concurrency.py
  ?? phase02_data_generation/src/generate_v8_local_gemma.py
  ?? phase02_data_generation/src/local_gemma_completion_pilot.py
  ?? phase02_data_generation/src/local_gemma_pilot.py
  ?? phase02_data_generation/src/pilot_fireworks_100.py
  ?? phase02_data_generation/src/prompt_repair_pilot.py
  ?? phase02_data_generation/src/recovery_1024.py
  ?? phase02_data_generation/src/smoke_test_fireworks.py
  ?? phase02_data_generation/data/v8/*.json  (reports — gitignored)
  ```
- **Recent commits:**
  ```
  78fbc82 chore: pre-V8 repo cleanup, conventions, and experiment records
  e47bbd9 feat: add V2 training variants and fix chunked generation for NemotronH
  a879ab4 docs: add screenshot proof of initial Kaggle submission (pending)
  59c1b29 feat: add placeholder submission.zip for initial Kaggle submission
  4160982 feat: add placeholder adapter generator for initial Kaggle submission
  ```
- **Changed files:**
  - `HANDOFF.md` — updated this session
  - `phase02_data_generation/src/generate_llm.py` — added `fireworks` and `local_openai` providers, switched system prompt to A++ post-hoc variant, updated `parse_response()` to return dict with gold comparison, `max_tokens` bumped 512→1024
  - New scripts (untracked): `generate_v8_local_gemma.py` (primary), `local_gemma_completion_pilot.py` (gate/sym-repair), `bench_concurrency.py` (benchmark)
- **Tests / build / lint:** not checked

## Key Decisions
| Decision | Rationale | Alternatives Rejected |
|---|---|---|
| Local Gemma 4 12B via raw `/completion` as primary provider | 100% parse, 96% copy on 100-row gate, $0 cost, avg 106 tok/row | Fireworks DeepSeek V4 Flash (90% good at max_tokens=1024, $7.28/9500 rows, persistent cipher/sym/unit failures) |
| Raw `/completion`, NOT `/v1/chat/completions` | llama.cpp `reasoning_format=none` strips all Gemma 4 output via chat endpoint; raw endpoint returns full token stream | `/v1/chat/completions` (100% empty content every row) |
| Deterministic symbol repair for symbol_transform copy failures | Model confuses puzzle-solving with copying on short special-char answers; deterministic copy of gold with fixed reasoning string keeps 100% usable rows | Sentinel prompt variants (tried `<ANSWER>` and `<<ANSWER>>` — both caused `<` tag-bleed into JSON values and 3 regressions vs general prompt) |
| workers=8 for full generation | 20-row test: 1.71 rows/s, VRAM flat, 0 failures. ~1.55h projected for 9,500 rows | workers=4 (1.56 rows/s, same quality, ~1.7h) |
| 95/5 train/val split, stratified by task_type | Maximizes training data; ensures all 6 task types appear in val | 90/10 split |

## Constraints and Preferences
- **Do NOT start full generation** without explicit user approval each session
- **Do NOT train LoRA** after generation — user must review dataset first
- **Do NOT commit** `.jsonl`, `.json` data files, `.csv` files, or `.claude/` directory
- **Do NOT use `/v1/chat/completions`** for local Gemma — raw `/completion` only
- **Do NOT use the sentinel symbol prompt** — causes tag-bleed regressions
- **workers=8 is approved** — user changed server to `--parallel 8` this session
- **Cost hard stop: $10** (moot for local run; applies if Fireworks used as fallback)
- **No modification** of `repair_pilot_30.csv`, `rejected_v8_pool.csv`, or v7 haiku outputs
- llama.cpp server must be running at `http://127.0.0.1:8080` before generation

## Do Not Do
- Do NOT run `generate_v8_local_gemma.py` without first deleting the 20-row test staging files (listed in Next Action)
- Do NOT use `generate_llm.py` for the local Gemma run — it uses `/v1/chat/completions`
- Do NOT retry content failures (parse/copy) — one attempt per row; symbol_transform copy fails get deterministic repair; everything else goes to failures log
- Do NOT merge/train LoRA automatically after generation completes
- Do NOT push commits without explicit user instruction

## Open Questions / Risks
- **20-row test staging files exist** — `local_gemma_staging.jsonl` (20 rows) and `train_reasoning_v8_local_gemma.jsonl` (20 rows) will cause resume logic to skip first 20 rows if not deleted before full run
- **n_ctx=512 per slot** — server now has 8 slots × 512 = 4096 total KV cache. Test showed `stopped_limit=False` on all rows including long gravity rows; gravity outputs compressed from ~487 to ~309 tokens but all correct. Watch for edge cases on very long gravity/unit prompts in the full run
- **symbol_transform failure rate** — gate showed 4/12 (33%) copy failures; all go to deterministic repair. Expect ~500 repair rows out of ~1,555 symbol_transform total. This is by design
- **Server must be running** — not verified at handoff time; confirm with `curl http://127.0.0.1:8080/health` before starting

## Next Action
Delete the 20-row test artifacts, confirm server is up, then start the full generation:

```bash
# 1. Delete test artifacts (prevents resume logic skipping first 20 rows)
rm phase02_data_generation/data/v8/local_gemma_staging.jsonl \
   phase02_data_generation/data/v8/train_reasoning_v8_local_gemma.jsonl \
   phase02_data_generation/data/v8/val_reasoning_v8_local_gemma.jsonl \
   phase02_data_generation/data/v8/local_gemma_full_report.json

# 2. Confirm server
curl -s http://127.0.0.1:8080/health

# 3. Full run (~1.55h, logs to file)
nohup /home/hareee234/miniconda3/envs/nemotron-train/bin/python3 \
  phase02_data_generation/src/generate_v8_local_gemma.py \
  --workers 8 --n-predict 512 --val-frac 0.05 \
  > /tmp/v8_generation.log 2>&1 &
echo "PID: $!"

# 4. Monitor (progress printed every 250 rows)
tail -f /tmp/v8_generation.log | grep -E "PROGRESS|FAIL|REPAIR|COMPLETE"
```

Expected: ~9,000+ model-accepted rows + ~500 symbol repairs → final `train_reasoning_v8_local_gemma.jsonl` and `val_reasoning_v8_local_gemma.jsonl` in `phase02_data_generation/data/v8/`. Stop and review `local_gemma_full_report.json` before LoRA training.
