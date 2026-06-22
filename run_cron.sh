#!/bin/bash
# Cron wrapper — runs analyze_frames.py with PaddleOCR.
SCRIPT_DIR="/home/z/my-project/world-ocr"
LOG="$SCRIPT_DIR/cron.log"

{
    echo "=== $(date -Iseconds) ==="
    cd "$SCRIPT_DIR"
    OCR_ENGINE=paddle /home/z/.venv/bin/python3 analyze_frames.py
    echo ""
} >> "$LOG" 2>&1