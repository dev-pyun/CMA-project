#!/bin/bash
# Run validation confusion matrix evaluation in background (SSH-safe).
# Output: src/val_confusion/  +  logs/val_confusion.log

cd /home/pyuncb/src
PYTHON=/home/pyuncb/.conda/envs/cloud/bin/python
LOG=logs/val_confusion.log

echo "Starting validation confusion matrix evaluation..."
echo "Log: $LOG"

nohup $PYTHON compute_val_confusion.py > "$LOG" 2>&1 &
PID=$!
echo "PID: $PID"
echo "$PID" > logs/val_confusion.pid
echo "Run 'tail -f $LOG' to monitor progress."
