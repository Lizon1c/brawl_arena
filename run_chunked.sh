#!/bin/bash
# Continue training the omni student on the 80k merged dataset in 10-epoch
# chunks; after each chunk, evaluate 20 gem_grab games. Stop early when
# winrate reaches >= 50% (10 wins). Logs to runs/student_vision/chunked.log
set -e
cd "$(dirname "$0")"
PY=.venv/Scripts/python
DATA=runs/student_vision/bc_data_merged.npz
ACTS=runs/student_vision/bc_acts_merged.npz
CKPT=runs/student_vision/omni_ep37.pt
EPOCH0=37
LOG=runs/student_vision/chunked.log

for round in 1 2 3 4 5 6; do
    echo "=== round $round: train 10 epochs from $CKPT ===" | tee -a "$LOG"
    $PY student_omni.py --train 10 --batch-size 8 --init "$CKPT" \
        --data "$DATA" --acts "$ACTS" --tag omni2 --epoch0 "$EPOCH0" \
        2>&1 | grep '^\[epoch' | tee -a "$LOG"
    CKPT=runs/student_vision/omni2_ep$((EPOCH0 + 10)).pt
    EPOCH0=$((EPOCH0 + 10))
    echo "=== eval $CKPT ===" | tee -a "$LOG"
    RES=$($PY eval_student_omni.py "$CKPT" 30 gem_grab 2>&1 | tail -1)
    echo "$RES" | tee -a "$LOG"
    W=$(echo "$RES" | grep -oP 'W\d+' | grep -oP '\d+')
    if [ "$W" -ge 15 ]; then
        echo "=== reached ${W}/30 wins, stopping ===" | tee -a "$LOG"
        break
    fi
done
echo "=== chunked training finished, last ckpt: $CKPT ===" | tee -a "$LOG"
