#!/bin/bash
# cron wrapper — runs analyze_frames.py every hour
# Logs to world-ocr/cron.log

SCRIPT_DIR="/home/z/my-project/world-ocr"
LOG="$SCRIPT_DIR/cron.log"

echo "=== $(date -Iseconds) ===" >> "$LOG"
cd "$SCRIPT_DIR" && /home/z/.venv/bin/python3 analyze_frames.py >> "$LOG" 2>&1
echo "" >> "$LOG"