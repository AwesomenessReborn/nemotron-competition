#!/usr/bin/env bash
set -euo pipefail

# Read-only verification of Nemotron Drive backups.
# Does not write, move, delete, or sync anything.

REMOTE="gdrive-hareee234:"
DATA_PATH="${REMOTE}data/kaggle/nemotron/"
RUNS_PATH="${REMOTE}runs/kaggle/nemotron/phase03/"
ARCHIVE_PATH="${REMOTE}backups/dev/kaggle/"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_LOCAL="${PROJECT_ROOT}/phase02_data_generation/data"
PHASE03_LOCAL="${PROJECT_ROOT}/phase03_local_smoke/outputs"

check_ok()   { echo "  OK"; }
check_warn() { echo "  WARNING: $*"; }

echo "================================================"
echo " Nemotron Drive Backup -- Verification Report"
echo " $(date)"
echo "================================================"

echo ""
echo "[1] rclone size: canonical data/"
rclone size "${DATA_PATH}"

echo ""
echo "[2] rclone size: canonical runs/phase03/"
rclone size "${RUNS_PATH}"

echo ""
echo "[3] rclone size: legacy archive (frozen -- do not sync to this path)"
rclone size "${ARCHIVE_PATH}"

echo ""
echo "[4] data/ directory tree"
rclone lsd -R "${DATA_PATH}"

echo ""
echo "[5] runs/phase03/ adapter runs"
rclone lsd "${RUNS_PATH}adapters/" 2>/dev/null \
    || check_warn "adapters/ not found on Drive"

echo ""
echo "[6] runs/phase03/ evals and logs (file counts)"
printf "  evals : "
rclone lsf "${RUNS_PATH}evals/" 2>/dev/null | wc -l
printf "  logs  : "
rclone lsf "${RUNS_PATH}logs/"  2>/dev/null | wc -l

echo ""
echo "[7] rclone check: phase02 v7 files (local -> Drive, one-way)"
rclone check "${DATA_LOCAL}/" "${DATA_PATH}phase02/v7/" \
    --include "train_reasoning_v7_*.jsonl" \
    --one-way \
    && check_ok || check_warn "v7 check found differences"

echo ""
echo "[8] rclone check: phase02 v8 finals (local -> Drive, one-way)"
rclone check "${DATA_LOCAL}/v8/" "${DATA_PATH}phase02/v8/" \
    --include "train_reasoning_v8_local_gemma.jsonl" \
    --include "val_reasoning_v8_local_gemma.jsonl" \
    --one-way \
    && check_ok || check_warn "v8 finals check found differences"

echo ""
echo "[9] rclone check: phase02 merged (local -> Drive, one-way)"
rclone check "${DATA_LOCAL}/merged/" "${DATA_PATH}phase02/merged/" \
    --exclude ".gitkeep" \
    --one-way \
    && check_ok || check_warn "merged check found differences"

echo ""
echo "================================================"
echo " Verification complete."
echo "================================================"
