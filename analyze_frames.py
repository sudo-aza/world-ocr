#!/usr/bin/env python3
"""
analyze_frames.py — Dissect video, OCR each frame for "World", mark it, reassemble.

Default engine: Tesseract 5 (fast, hourly-cron friendly).
Set OCR_ENGINE=paddle to use PaddleOCR (slower, higher accuracy on complex scenes).
"""

import subprocess
import sys
import os
import re
import shutil
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
    # PaddleOCR expects BGR
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
    find_fn = find_word_paddle if engine == "paddle" else find_word_tesseract
    print(f"  OCR engine: {engine}")

    print(f"[1/4] Reading {INPUT_VIDEO}")
    fps, w, h = get_video_info(INPUT_VIDEO)
    print(f"  {w}x{h} @ {fps:.1f} fps")

    print(f"[2/4] Extracting frames to {FRAMES_DIR}/")
    frames = dissect_video(INPUT_VIDEO, FRAMES_DIR, fps)

    print(f"[3/4] Scanning {len(frames)} frames for \"{TARGET_WORD}\"")
    os.makedirs(MARKED_DIR, exist_ok=True)

    found = 0
    for idx, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue

        matches = find_fn(img, TARGET_WORD)
        if matches:
            found += 1
            for x1, y1, x2, y2, text, conf in matches:
                img = draw_box(img, x1, y1, x2, y2, text, conf)

        cv2.imwrite(str(Path(MARKED_DIR) / fp.name), img)

        if (idx + 1) % 30 == 0 or idx == len(frames) - 1:
            print(f"  ... {idx + 1}/{len(frames)} frames processed, {found} with \"{TARGET_WORD}\"")

    print(f"  Found \"{TARGET_WORD}\" in {found}/{len(frames)} frames")

    print(f"[4/4] Reassembling into {OUTPUT_VIDEO}")
    reassemble_video(MARKED_DIR, OUTPUT_VIDEO, fps, w, h)

    # Cleanup frame dirs
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(MARKED_DIR, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()