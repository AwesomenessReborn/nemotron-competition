#!/usr/bin/env bash
# Re-evaluate all three variant adapters with chunked (no-SSM-cache) generation.
set -euo pipefail

PYTHON=/home/hareee234/miniconda3/envs/nemotron-train/bin/python3.11
SCRIPT=phase03_local_smoke/src/diag_chunked_gen.py
VAL=phase02_data_generation/data/merged/val.jsonl
LOGDIR=phase03_local_smoke/outputs/logs

cd /home/hareee234/Dev/kaggle/nemotron-competition-may/nemotron-competition

for TARGET in answer_only short_reasoning haiku_reasoning; do
  ADAPTER="phase03_local_smoke/outputs/adapters/variants_1k/${TARGET}/final_adapter_${TARGET}"
  LOG="$LOGDIR/chunked_eval_${TARGET}.log"
  echo "======================================================"
  echo "  Chunked eval: $TARGET"
  echo "======================================================"

  $PYTHON "$SCRIPT" \
    --adapter-dir "$ADAPTER" \
    --val "$VAL" \
    --n 100 \
    --max-new 80 \
    2>&1 | tee "$LOG"

  echo ""
done

echo "======================================================"
echo "  FINAL COMPARISON (chunked generation)"
echo "======================================================"

$PYTHON -c "
import re, glob

for target in ['answer_only', 'short_reasoning', 'haiku_reasoning']:
    log = f'phase03_local_smoke/outputs/logs/chunked_eval_{target}.log'
    try:
        txt = open(log).read()
        # Get CHUNKED summary line
        m = re.search(r'SUMMARY \[CHUNKED.*?\]: boxed=(\d+)/(\d+)\s+correct=(\d+)/(\d+)', txt)
        if m:
            b, n, c, _ = m.groups()
            print(f'{target:<22}  boxed={b}/{n} ({100*int(b)/int(n):.0f}%)  correct={c}/{n} ({100*int(c)/int(n):.0f}%)')
        else:
            print(f'{target:<22}  (no summary found)')
    except Exception as e:
        print(f'{target:<22}  ERROR: {e}')
"
