#!/bin/bash
# run_cron.sh — runs analyze_frames.py with the NEXT engine in the rotation.
# Tracks which engine to use next via .engine_index file.
# Each run produces:
#   - output_<engine>.mp4    (video with boxes)
#   - results_<engine>.json  (per-frame detection data)
# Logs everything to cron.log.

set -euo pipefail
SCRIPT_DIR="/home/z/my-project/world-ocr"
LOG="$SCRIPT_DIR/cron.log"
PYTHON="/home/z/.venv/bin/python3"
ENGINES=("tesseract" "paddle" "easyocr")
INDEX_FILE="$SCRIPT_DIR/.engine_index"

# Read or init engine index
if [ -f "$INDEX_FILE" ]; then
    IDX=$(cat "$INDEX_FILE")
else
    IDX=0
fi

ENGINE="${ENGINES[$IDX]}"
NEXT_IDX=$(( (IDX + 1) % ${#ENGINES[@]} ))
echo "$NEXT_IDX" > "$INDEX_FILE"

{
    echo "=== $(date -Iseconds) — engine: $ENGINE (index $IDX) ==="
    cd "$SCRIPT_DIR"
    OCR_ENGINE="$ENGINE" OUTPUT_VIDEO="output_${ENGINE}.mp4" "$PYTHON" analyze_frames.py
    echo ""
} >> "$LOG" 2>&1