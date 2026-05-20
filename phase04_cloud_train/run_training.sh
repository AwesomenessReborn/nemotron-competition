#!/usr/bin/env bash
# Phase 4 training launcher.
# Run from the repo root:  bash phase04_cloud_train/run_training.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

ENV_NAME="nemotron-train"
CONFIG="phase04_cloud_train/configs/cloud_30b.yaml"
TRAIN_DATA="phase02_data_generation/data/merged/train.jsonl"
VAL_DATA="phase02_data_generation/data/merged/val.jsonl"
ADAPTER_DIR="phase04_cloud_train/outputs/adapters/cloud_30b/final_adapter"
LOG_DIR="phase04_cloud_train/outputs/logs"

echo "=================================================="
echo " Nemotron Phase 4 — 30B LoRA Training"
echo " Config:     $CONFIG"
echo " Train data: $TRAIN_DATA"
echo " Val data:   $VAL_DATA"
echo "=================================================="

# ── Pre-flight checks ────────────────────────────────────
echo "[check] Conda env '$ENV_NAME'..."
conda env list | grep -q "^$ENV_NAME " || { echo "ERROR: Run setup.sh first."; exit 1; }

echo "[check] Training data..."
[ -f "$TRAIN_DATA" ] || { echo "ERROR: $TRAIN_DATA not found. Upload it first (see DEPLOY.md)."; exit 1; }
[ -f "$VAL_DATA" ]   || { echo "ERROR: $VAL_DATA not found."; exit 1; }

TRAIN_ROWS=$(wc -l < "$TRAIN_DATA")
VAL_ROWS=$(wc -l < "$VAL_DATA")
echo "[check] Train: $TRAIN_ROWS rows | Val: $VAL_ROWS rows"

echo "[check] HuggingFace auth..."
conda run -n "$ENV_NAME" python -c "
from huggingface_hub import whoami
try:
    u = whoami()
    print(f'  Logged in as: {u[\"name\"]}')
except Exception:
    print('  WARNING: Not logged in to HuggingFace.')
    print('  Run: huggingface-cli login')
    print('  Then accept the model license at:')
    print('  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
    import sys; sys.exit(1)
"

echo "[check] GPU memory..."
conda run -n "$ENV_NAME" python -c "
import torch
gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
name = torch.cuda.get_device_name(0)
print(f'  {name}  {gb:.0f} GB')
if gb < 70:
    print('  WARNING: <70 GB VRAM. 30B BF16 needs ~75 GB. Consider load_in_4bit: true.')
"

# ── Create output dirs ───────────────────────────────────
mkdir -p "$LOG_DIR" "phase04_cloud_train/outputs/adapters/cloud_30b"

# ── Run training ─────────────────────────────────────────
echo ""
echo "[train] Starting 30B LoRA training..."
echo "        Logs: $LOG_DIR/train.log"
echo "        (tail -f $LOG_DIR/train.log to monitor)"
echo ""

conda run -n "$ENV_NAME" python -u phase03_local_smoke/src/train_lora.py \
    --config "$CONFIG" \
    --train  "$TRAIN_DATA" \
    --val    "$VAL_DATA" \
    2>&1 | tee "$LOG_DIR/train.log"

# ── Validate ─────────────────────────────────────────────
echo ""
echo "[validate] Checking final adapter..."
conda run -n "$ENV_NAME" python phase03_local_smoke/src/validate_adapter.py \
    --adapter-dir "$ADAPTER_DIR"

echo ""
echo "=================================================="
echo " Training complete."
echo " Adapter: $ADAPTER_DIR"
echo " Next: download the adapter and run /smart-commit"
echo "=================================================="
