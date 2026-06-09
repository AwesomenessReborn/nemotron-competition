# Google Drive Backup — Nemotron Competition

## Canonical Drive organization

```
gdrive-hareee234:
├── data/
│   └── kaggle/
│       └── nemotron/
│           ├── source/                          # Original competition CSVs (read-only)
│           │   ├── train.csv
│           │   └── train_with_task_type.csv
│           └── phase02/                         # Generated training datasets
│               ├── v5/train_reasoning_v5.jsonl
│               ├── v6/train_reasoning_v6.jsonl
│               ├── v7/
│               │   ├── train_reasoning_v7_haiku.jsonl
│               │   └── train_reasoning_v7_fireworks.jsonl
│               ├── v8/
│               │   ├── train_reasoning_v8_local_gemma.jsonl   # FINAL train
│               │   ├── val_reasoning_v8_local_gemma.jsonl     # FINAL val
│               │   └── intermediates/                         # pilots, staging, failures
│               └── merged/
│                   ├── train.jsonl
│                   ├── val.jsonl
│                   └── smoke_50.jsonl
│
├── runs/
│   └── kaggle/
│       └── nemotron/
│           └── phase03/
│               ├── adapters/
│               │   ├── debug_overfit_adapter/   # overfit sanity check (15 checkpoints)
│               │   ├── full_haiku_9500/         # primary trained adapter (v7 full run)
│               │   ├── local_4b/                # local 4B model run, final only
│               │   └── variants_1k/             # answer_only / short_reasoning / haiku_reasoning
│               ├── evals/
│               └── logs/
│
└── backups/
    └── dev/
        └── kaggle/                              # FROZEN LEGACY ARCHIVE — see below
```

---

## Why `backups/dev/kaggle/` is a frozen legacy archive

This path was created by an earlier `rclone sync` run that mirrored the full repo
structure (including source code) to Drive. It contains several files that no longer
exist locally:

- `train_reasoning_v5.jsonl` (6 MB)
- `train_reasoning_v6.jsonl` (15 MB)
- `train.csv`, `train_with_task_type.csv`

**Do not run `rclone sync` pointed at this path.** `sync` is a mirror-and-delete
command — it would delete these Drive-only files the next time it ran against the
current local project.

These files have since been copied into the canonical `data/` structure above and
are safe there. The `backups/dev/kaggle/` path is kept as an audit trail and should
not be modified or synced to.

---

## Why `rclone copy` is preferred over `rclone sync`

| Command | Behaviour | Risk |
|---------|-----------|------|
| `rclone copy src dst` | Copies src → dst. Files on dst that aren't in src are left alone. | Safe — additive only |
| `rclone sync src dst` | Mirrors src → dst exactly. **Deletes files on dst that aren't in src.** | Dangerous — can silently delete Drive-only files |

For dataset backups and training artifact archiving, we never want Drive to lose
files just because they were reorganized or temporarily absent locally. Always use
`rclone copy`.

---

## How to run the backup script

```bash
# From project root:
./scripts/backup_nemotron_to_drive.sh
```

What it backs up:
1. Phase02 v7 datasets → `data/kaggle/nemotron/phase02/v7/`
2. Phase02 v8 final splits → `data/kaggle/nemotron/phase02/v8/`
3. Phase02 v8 intermediates → `data/kaggle/nemotron/phase02/v8/intermediates/`
4. Phase02 merged production splits → `data/kaggle/nemotron/phase02/merged/`
5. Phase03 adapters (all runs, checkpoints excluded from optimizer/rng) → `runs/kaggle/nemotron/phase03/adapters/`
6. Phase03 evals + logs → `runs/kaggle/nemotron/phase03/evals/` and `logs/`

---

## How to run dry-run mode

```bash
DRY_RUN=1 ./scripts/backup_nemotron_to_drive.sh
```

Prints exactly what would be transferred without writing anything to Drive.
Review the output, then run without `DRY_RUN=1` to apply.

---

## How to verify backups

```bash
./scripts/check_nemotron_drive_backup.sh
```

This script is read-only. It runs:
- `rclone size` on the canonical data and runs paths
- `rclone lsd` directory tree summaries
- `rclone check --one-way` for v7 files, v8 finals, and merged splits

---

## Training data locality

**Do not train directly from the GVFS mount or from Drive paths.**

The GVFS mount at `/run/user/1000/gvfs/google-drive:host=gmail.com,user=hareee234/`
is a GNOME remote filesystem. It is flaky under sustained I/O, has no OS-level
caching, and can stall or error mid-epoch. For training:

1. Copy the required dataset to local NVMe first:
   ```bash
   rclone copy gdrive-hareee234:data/kaggle/nemotron/phase02/v7/train_reasoning_v7_haiku.jsonl \
     ./phase02_data_generation/data/
   ```
2. Point the training script at the local path.
3. After training, back up outputs with the backup script.

---

## Excluded files: optimizer.pt and rng_state.pth

The backup script excludes `optimizer.pt` and `rng_state.pth` from all adapter
checkpoints by default. These files are:

- **`optimizer.pt`** — AdamW optimizer state (~261 MB per checkpoint). Only needed
  to resume training from an exact checkpoint. Not needed for inference or eval.
- **`rng_state.pth`** — PyTorch RNG state for reproducible resumption. Tiny but
  meaningless without `optimizer.pt`.

If you need to resume training from a specific checkpoint, back up `optimizer.pt`
manually for that checkpoint only:
```bash
rclone copy \
  ./phase03_local_smoke/outputs/adapters/<run>/checkpoint-<N>/optimizer.pt \
  gdrive-hareee234:runs/kaggle/nemotron/phase03/adapters/<run>/checkpoint-<N>/
```

---

## Phase04 placeholder adapter (deferred)

`phase04_cloud_train/outputs/adapters/cloud_30b/placeholder_adapter/adapter_model.safetensors`
(1.7 GB) is a **zero-initialized no-op LoRA adapter** for the 30B Nemotron model.
All LoRA weights are zero — it is mathematically identical to the base model and
was used only to pass Kaggle submission validation while real training was in flight.

It is not backed up by default because:
- It contains no trained weights and cannot be used for resumption or inference.
- It is fully reproducible by running `make_placeholder_adapter.py` (~5 minutes).
- The submission artifact (`submission.zip`) is already backed up on Drive.

To back it up manually if needed:
```bash
rclone copy \
  ./phase04_cloud_train/outputs/adapters/cloud_30b/ \
  gdrive-hareee234:runs/kaggle/nemotron/phase04/adapters/cloud_30b/ \
  --progress
```
