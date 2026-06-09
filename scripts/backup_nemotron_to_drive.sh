#!/usr/bin/env bash
set -euo pipefail

# Copies local project artifacts to canonical Google Drive paths.
# Uses rclone copy only — never rclone sync, move, or delete.
#
# Usage:
#   ./scripts/backup_nemotron_to_drive.sh            # real backup
#   DRY_RUN=1 ./scripts/backup_nemotron_to_drive.sh  # dry run, no writes

REMOTE="gdrive-hareee234:"
DATA_BASE="${REMOTE}data/kaggle/nemotron/phase02"
RUNS_BASE="${REMOTE}runs/kaggle/nemotron/phase03"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_LOCAL="${PROJECT_ROOT}/phase02_data_generation/data"
PHASE03_LOCAL="${PROJECT_ROOT}/phase03_local_smoke/outputs"

DRY_RUN="${DRY_RUN:-0}"

RCLONE_FLAGS=(--progress --transfers 4 --checkers 8)
[[ "${DRY_RUN}" == "1" ]] && RCLONE_FLAGS+=(--dry-run)

rcopy() { rclone copy "$@" "${RCLONE_FLAGS[@]}"; }

# -- guard: verify required local paths exist ---------------------
for path in \
    "${DATA_LOCAL}" \
    "${DATA_LOCAL}/v8" \
    "${DATA_LOCAL}/merged" \
    "${PHASE03_LOCAL}/adapters" \
    "${PHASE03_LOCAL}/evals" \
    "${PHASE03_LOCAL}/logs"; do
    [[ -d "${path}" ]] || { echo "ERROR: expected directory missing: ${path}"; exit 1; }
done

echo "================================================"
echo " Nemotron -> Google Drive backup"
echo " Project : ${PROJECT_ROOT}"
echo " DRY_RUN : ${DRY_RUN}"
echo "================================================"

# [1/6] v7 datasets
echo ""
echo "[1/6] phase02 v7 -> ${DATA_BASE}/v7/"
rcopy "${DATA_LOCAL}/train_reasoning_v7_haiku.jsonl"     "${DATA_BASE}/v7/"
rcopy "${DATA_LOCAL}/train_reasoning_v7_fireworks.jsonl" "${DATA_BASE}/v7/"

# [2/6] v8 final splits
echo ""
echo "[2/6] phase02 v8 finals -> ${DATA_BASE}/v8/"
rcopy "${DATA_LOCAL}/v8/train_reasoning_v8_local_gemma.jsonl" "${DATA_BASE}/v8/"
rcopy "${DATA_LOCAL}/v8/val_reasoning_v8_local_gemma.jsonl"   "${DATA_BASE}/v8/"

# [3/6] v8 intermediates (everything in v8/ except the two finals and .gitkeep)
echo ""
echo "[3/6] phase02 v8 intermediates -> ${DATA_BASE}/v8/intermediates/"
rcopy "${DATA_LOCAL}/v8/" "${DATA_BASE}/v8/intermediates/" \
    --exclude "train_reasoning_v8_local_gemma.jsonl" \
    --exclude "val_reasoning_v8_local_gemma.jsonl" \
    --exclude ".gitkeep"

# [4/6] merged production splits
echo ""
echo "[4/6] phase02 merged -> ${DATA_BASE}/merged/"
rcopy "${DATA_LOCAL}/merged/" "${DATA_BASE}/merged/" \
    --exclude ".gitkeep"

# [5/6] phase03 adapters (optimizer.pt + rng_state.pth excluded — resume-only, bulky)
echo ""
echo "[5/6] phase03 adapters -> ${RUNS_BASE}/adapters/"
echo "      (optimizer.pt and rng_state.pth excluded)"
rcopy "${PHASE03_LOCAL}/adapters/" "${RUNS_BASE}/adapters/" \
    --exclude "optimizer.pt" \
    --exclude "rng_state.pth" \
    --exclude ".gitkeep"

# [6/6] phase03 evals + logs
echo ""
echo "[6/6] phase03 evals + logs -> ${RUNS_BASE}/"
rcopy "${PHASE03_LOCAL}/evals/" "${RUNS_BASE}/evals/" --exclude ".gitkeep"
rcopy "${PHASE03_LOCAL}/logs/"  "${RUNS_BASE}/logs/"

echo ""
echo "================================================"
[[ "${DRY_RUN}" == "1" ]] \
    && echo " Dry run complete. Re-run without DRY_RUN=1 to apply." \
    || echo " Backup complete."
echo "================================================"
