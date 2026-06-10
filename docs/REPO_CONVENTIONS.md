# Repository Conventions

## Phase directory responsibilities

| Phase | Purpose | Owns |
|---|---|---|
| `phase01_teacher_benchmark` | Evaluate teacher model quality; produce trace correctness reports | `src/generate_traces.py`, `src/compare_teachers.py`, `src/build_teacher_test.py` |
| `phase02_data_generation` | Generate and version training datasets via LLM APIs | `src/generate_full_haiku.py`, `src/generate_full_gemini.py`, `src/generate_llm.py`, `src/build_splits.py`, `src/merge_datasets.py` |
| `phase03_local_smoke` | Local LoRA training, diagnostics, and eval on RTX 5070 Ti | `src/train_lora*.py`, `src/eval_*.py`, `src/diag_*.py` |
| `phase04_cloud_train` | Cloud-scale training (A100/H100), submission packaging | deployment scripts, `submission.zip` builder |
| `shared` | Raw competition data (`shared/data/raw/`), shared utilities (`shared/src/`) | Not modified after Phase 01 |

Each phase directory has the same internal layout:
```
<phase>/
  src/        scripts (committed)
  configs/    YAML/JSON hyperparams (committed)
  outputs/    generated artifacts (gitignored)
    adapters/
    evals/
    logs/
```

Never put source code under `outputs/`. Never put data or model weights under `src/`.

## Script naming rules

| Prefix | Meaning |
|---|---|
| `train_lora*.py` | LoRA training loop |
| `eval_*.py` | Evaluation / accuracy measurement |
| `diag_*.py` | Diagnostic / debugging (not part of production pipeline) |
| `generate_*.py` | LLM API calls to produce training data |
| `build_*.py` | Dataset construction, split generation |
| `compare_*.py` | Side-by-side comparison of two models or datasets |
| `validate_*.py` | Sanity-check a file, adapter, or split |
| `prompt_template.py` | Shared prompt formatting (imported, not run directly) |

Root-level `.sh` scripts are orchestration wrappers only — they call phase scripts in sequence. Do not put logic in shell scripts.

## When to modify vs create a new script

**Modify the existing script when:**
- Fixing a bug (generation mode, parsing regex, label masking, etc.)
- Adding a `--flag` that changes behavior at runtime (new `--target-type`, `--provider`, `--max-new`)
- Swapping an LLM provider while keeping the same data format
- Adding metrics to an eval script
- Refactoring internals that do not change the CLI or output format

**Create a new script when:**
- The input or output format is fundamentally different (e.g., a new data schema)
- The old script must remain runnable to reproduce a past experiment
- The new job is a genuinely distinct operation (e.g., adding a solver-based generator alongside the LLM generator)

When creating a new version of a training or eval script, suffix it with `_v<N>` (e.g., `train_lora_v2.py`) and note in a `docs/experiments/` record which script produced which adapter.

## Dataset versioning

Training datasets live under `phase02_data_generation/data/`. Each version is a named subdirectory:

```
phase02_data_generation/data/
  merged/        current production split (train.jsonl + val.jsonl)
  v8/            next version under construction
  <raw>/         provider-specific raw outputs (gitignored via *.jsonl)
```

Rules:
- Never overwrite `merged/` in place. When a new dataset version is ready, copy it to `merged/` and tag the commit with the version name.
- Version numbers (`v7`, `v8`, etc.) track teacher model or generation strategy changes, not minor fixes. Minor fixes within a version append `.1`, `.2`, etc. (e.g. `v8.1`).
- See `docs/dataset_versions.md` for the full version changelog — what each version changed, why, and what went wrong.
- Each version directory should contain a `README.md` or a `docs/experiments/` record explaining provenance.

## Experiment output naming

Adapter directories: `<run_slug>/final_adapter_<target_type>/`
Example: `full_haiku_9500/final_adapter_haiku_reasoning/`

Eval JSON files: `eval_<adapter_name>_<YYYYMMDD_HHMMSS>.json`
Example: `eval_final_adapter_haiku_reasoning_20260604_124340.json`

Log files: `<run_slug>.log`
Example: `full_haiku_9500.log`

Do not embed hyperparameter values in the run slug — those belong in the experiment record and in the eval JSON, not the directory name.

## Commit hygiene

- Never commit files matched by `.gitignore` (outputs, weights, data, caches, secrets).
- Commit messages follow `type: short description` (`feat:`, `fix:`, `docs:`, `chore:`).
- Each commit that trains an adapter or generates a dataset should reference the experiment record in the commit message.
- Docs commits (`docs/experiments/NNN_*.md`) should be in the same commit as the code change that produced the result, or immediately after.
- Do not batch unrelated changes. One commit per logical unit of work.
- Push to GitHub regularly — at minimum after each completed phase step.
