#!/bin/bash
# run_cron.sh — picks the next untested OCR engine, installs it, runs the pipeline.
# Engine registry lives in engines.json.
# Already-tested engines tracked in engines_done.json.
# Logs to cron.log.

set -euo pipefail
SCRIPT_DIR="/home/z/my-project/world-ocr"
LOG="$SCRIPT_DIR/cron.log"
PYTHON="/home/z/.venv/bin/python3"

{
    echo "=== $(date -Iseconds) ==="
    cd "$SCRIPT_DIR"
    "$PYTHON" analyze_frames.py
    echo ""
} >> "$LOG" 2>&1