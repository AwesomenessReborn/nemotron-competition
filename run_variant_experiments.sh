#!/usr/bin/env bash
# Run three small V2 training experiments comparing assistant target formats.
# Each: 1000 train rows, 100 val rows, rank 32, 3 epochs.
# Outputs go to phase03_local_smoke/outputs/adapters/variants_1k/

set -euo pipefail

PYTHON=/home/hareee234/miniconda3/envs/nemotron-train/bin/python3.11
SCRIPT=phase03_local_smoke/src/train_lora_v2.py
TRAIN=phase02_data_generation/data/merged/train.jsonl
VAL=phase02_data_generation/data/merged/val.jsonl
OUTBASE=phase03_local_smoke/outputs/adapters/variants_1k
LOGDIR=phase03_local_smoke/outputs/logs

mkdir -p "$LOGDIR"

cd /home/hareee234/Dev/kaggle/nemotron-competition-may/nemotron-competition

for TARGET in answer_only short_reasoning haiku_reasoning; do
  LOG="$LOGDIR/variant_${TARGET}.log"
  echo "======================================================"
  echo "  Starting experiment: $TARGET"
  echo "  Log: $LOG"
  echo "======================================================"

  $PYTHON "$SCRIPT" \
    --train "$TRAIN" \
    --val "$VAL" \
    --output "$OUTBASE/$TARGET" \
    --rank 32 \
    --epochs 3 \
    --n-train 1000 \
    --n-val 100 \
    --batch 1 \
    --accum 4 \
    --lr 2e-4 \
    --target-type "$TARGET" \
    2>&1 | tee "$LOG"

  echo ""
  echo "  Experiment $TARGET complete."
  echo ""
done

echo "======================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "======================================================"

# Print comparison table
$PYTHON -c "
import json, os, glob
base = 'phase03_local_smoke/outputs/adapters/variants_1k'
files = sorted(glob.glob(f'{base}/*/eval_summary_*.json'))
if not files:
    print('No summary files found yet.')
else:
    print(f\"{'Target':<22} {'Train box%':>10} {'Train acc%':>10} {'Val box%':>10} {'Val acc%':>10}\")
    print('-'*64)
    for f in files:
        d = json.load(open(f))
        print(f\"{d['target_type']:<22} {d['train_boxed_pct']:>10} {d['train_acc_pct']:>10} {d['val_boxed_pct']:>10} {d['val_acc_pct']:>10}\")
"
