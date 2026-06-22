#!/usr/bin/env python3
"""
analyze_frames.py — Dissect video, OCR each frame for "World", mark it, reassemble.

Supported engines (via OCR_ENGINE env var):
  tesseract  — Tesseract 5 (fast, decent accuracy)
  paddle    — PaddleOCR PP-OCRv4 (slower, good on complex scenes)
  easyocr   — EasyOCR (PyTorch-based, strong multilingual)

Outputs:
  output.mp4          — reassembled video with boxes drawn
  results_<engine>.json — per-frame detection summary for this engine run
"""

import subprocess
import sys
import os
import re
import shutil
import json
import time
from pathlib import Path

import cv2
import numpy as np

# ── Config ───────────────────────────────────────────────────────
INPUT_VIDEO  = os.environ.get("INPUT_VIDEO", "input.mp4")
OUTPUT_VIDEO = os.environ.get("OUTPUT_VIDEO", "output.mp4")
TARGET_WORD  = "World"
FRAMES_DIR   = "frames"
MARKED_DIR   = "marked"
BOX_COLOR    = (0, 255, 0)       # BGR green
BOX_THICKNESS = 2
LABEL_FONT   = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE  = 0.55
LABEL_THICKNESS = 1


def get_video_info(path):
    """Return (fps, width, height) using ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True
    )
    parts = r.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    fps_num, fps_den = map(int, parts[2].split("/"))
    return fps_num / fps_den, w, h


def dissect_video(video_path, out_dir, fps):
    """Extract every frame to out_dir/frame_NNNN.png."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(out_dir, "frame_%04d.png")
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    frames = sorted(Path(out_dir).glob("frame_*.png"))
    print(f"  Extracted {len(frames)} frames")
    return frames


def find_word_tesseract(image_bgr, target):
    """Use Tesseract 5 to find word bounding boxes. Returns list of (x1,y1,x2,y2,text,conf)."""
    import pytesseract
    data = pytesseract.image_to_data(image_bgr, output_type=pytesseract.Output.DICT,
                                      config="--psm 11 --oem 3")
    matches = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        if re.search(re.escape(target), text, re.IGNORECASE):
            conf = int(data["conf"][i])
            if conf < 10:
                continue
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            matches.append((x, y, x + w, y + h, text, conf / 100.0))
    return matches


def find_word_paddle(image_bgr, target):
    """Use PaddleOCR to find word bounding boxes. Returns list of (x1,y1,x2,y2,text,conf)."""
    from paddleocr import PaddleOCR
    if not hasattr(find_word_paddle, "_ocr"):
        find_word_paddle._ocr = PaddleOCR(
            use_angle_cls=False, lang="en", show_log=False,
            det_db_thresh=0.3, det_db_box_thresh=0.5
        )
    ocr = find_word_paddle._ocr
    result = ocr.ocr(image_bgr, cls=False, det_db_thresh=0.3, det_db_box_thresh=0.5)
    if not result or not result[0]:
        return []
    matches = []
    for line in result[0]:
        bbox_pts, (text, conf) = line[0], line[1]
        if re.search(re.escape(target), text, re.IGNORECASE):
            pts = np.array(bbox_pts).astype(int)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            matches.append((int(x1), int(y1), int(x2), int(y2), text, conf))
    return matches


def find_word_easyocr(image_bgr, target):
    """Use EasyOCR to find word bounding boxes. Returns list of (x1,y1,x2,y2,text,conf)."""
    import easyocr
    if not hasattr(find_word_easyocr, "_reader"):
        find_word_easyocr._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    reader = find_word_easyocr._reader
    # EasyOCR expects RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(image_rgb)
    matches = []
    for (bbox_pts, text, conf) in results:
        if re.search(re.escape(target), text, re.IGNORECASE):
            pts = np.array(bbox_pts).astype(int)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            matches.append((int(x1), int(y1), int(x2), int(y2), text, conf))
    return matches


def draw_box(image, x1, y1, x2, y2, text, conf):
    """Draw bounding box and label."""
    pad = 3
    cv2.rectangle(image, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad),
                  BOX_COLOR, BOX_THICKNESS)
    label = f"{text} ({conf:.0%})"
    (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    cv2.rectangle(image, (x1 - pad, y1 - th - pad * 3),
                  (x1 + tw + pad, y1 - pad),
                  BOX_COLOR, -1)
    cv2.putText(image, label, (x1, y1 - pad * 2),
                LABEL_FONT, LABEL_SCALE, (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA)
    return image


def reassemble_video(marked_dir, output_path, fps, width, height):
    """Combine marked frames back into MP4 via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(marked_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output: {output_path} ({size_mb:.1f} MB)")


def main():
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    engine = os.environ.get("OCR_ENGINE", "tesseract")
    engine_map = {
        "tesseract": find_word_tesseract,
        "paddle": find_word_paddle,
        "easyocr": find_word_easyocr,
    }
    if engine not in engine_map:
        print(f"  ERROR: unknown engine '{engine}'. Choose from: {', '.join(engine_map)}")
        sys.exit(1)
    find_fn = engine_map[engine]
    print(f"  OCR engine: {engine}")

    print(f"[1/4] Reading {INPUT_VIDEO}")
    fps, w, h = get_video_info(INPUT_VIDEO)
    print(f"  {w}x{h} @ {fps:.1f} fps")

    print(f"[2/4] Extracting frames to {FRAMES_DIR}/")
    frames = dissect_video(INPUT_VIDEO, FRAMES_DIR, fps)

    print(f"[3/4] Scanning {len(frames)} frames for \"{TARGET_WORD}\"")
    os.makedirs(MARKED_DIR, exist_ok=True)

    found = 0
    per_frame = []
    total_conf = 0.0
    conf_count = 0

    for idx, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue

        t0 = time.time()
        matches = find_fn(img, TARGET_WORD)
        elapsed = time.time() - t0

        frame_matches = []
        if matches:
            found += 1
            for x1, y1, x2, y2, text, conf in matches:
                img = draw_box(img, x1, y1, x2, y2, text, conf)
                total_conf += conf
                conf_count += 1
                frame_matches.append({
                    "text": text, "conf": round(conf, 4),
                    "box": [x1, y1, x2, y2]
                })

        cv2.imwrite(str(Path(MARKED_DIR) / fp.name), img)
        per_frame.append({
            "frame": fp.name, "found": len(matches) > 0,
            "matches": frame_matches, "time_s": round(elapsed, 3)
        })

        if (idx + 1) % 30 == 0 or idx == len(frames) - 1:
            print(f"  ... {idx + 1}/{len(frames)} frames processed, {found} with \"{TARGET_WORD}\"")

    avg_conf = (total_conf / conf_count) if conf_count else 0.0
    print(f"  Found \"{TARGET_WORD}\" in {found}/{len(frames)} frames")
    print(f"  Avg confidence: {avg_conf:.1%}")

    print(f"[4/4] Reassembling into {OUTPUT_VIDEO}")
    reassemble_video(MARKED_DIR, OUTPUT_VIDEO, fps, w, h)

    # Write results.json for this engine
    results = {
        "engine": engine, "target": TARGET_WORD,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frames_total": len(frames), "frames_found": found,
        "avg_confidence": round(avg_conf, 4),
        "per_frame": per_frame,
    }
    results_path = f"results_{engine}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    # Cleanup frame dirs
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(MARKED_DIR, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()