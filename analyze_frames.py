#!/usr/bin/env python3
"""
analyze_frames.py — Dissect video, OCR each frame for "World", mark it, reassemble.
The find_word() function is rewritten by the agent before each run.
"""

import subprocess, os, re, shutil
from pathlib import Path
import cv2, numpy as np

INPUT_VIDEO  = os.environ.get("INPUT_VIDEO", "input.mp4")
OUTPUT_VIDEO = os.environ.get("OUTPUT_VIDEO", "output.mp4")
TARGET_WORD  = "World"
FRAMES_DIR   = "frames"
MARKED_DIR   = "marked"
BOX_COLOR    = (0, 255, 0)
BOX_THICKNESS = 2
LABEL_FONT   = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE  = 0.55
LABEL_THICKNESS = 1


import warnings
warnings.filterwarnings("ignore")

# --- Engine #18: HSV color-space text segmentation + morphological closing + CRNN ---
# Novel approach: Converts frame to HSV color space, thresholds for bright
# low-saturation pixels (white/light text on dark background), applies
# morphological closing to connect character regions into text lines,
# then finds contours as bounding boxes. Crops are fed to RapidOCR's
# CRNN recognizer. Distinct from all prior engines:
#   - Engine 17 (Canny + dilation): uses edge-based detection, not color
#   - Engines 1,5,7,8,11,13,16 (DBNet/RapidOCR): neural network detectors
#   - Engines 2,3,6,9,10,12,15 (Tesseract): built-in detection
#   - Engine 14 (template tracking): score-map template matching
#   - Engine 4 (OCR.space): cloud API
# The novelty is color-space based pixel classification for text detection
# (hue/saturation/value thresholding) — no gradient/edge computation
# and no neural network for the detection stage.
from rapidocr_onnxruntime import RapidOCR

_rapid = RapidOCR()
_recognizer = _rapid.text_rec

# Pre-build morphological kernel for closing (connect character blobs)
_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))


def _fuzzy_match(text, target, max_dist=3):
    text_lower = text.lower().strip()
    target_lower = target.lower()
    tlen = len(target_lower)
    if len(text_lower) < tlen:
        return False
    for i in range(len(text_lower) - tlen + 1):
        window = text_lower[i:i + tlen]
        diffs = sum(a != b for a, b in zip(window, target_lower))
        if diffs <= max_dist:
            return True
    return False


def find_word(img_bgr, target):
    """Engine #18: HSV text segmentation + morphological closing + CRNN.
    Return list of (x1, y1, x2, y2, text, conf).
    """
    h, w = img_bgr.shape[:2]

    # Convert to HSV and isolate bright, low-saturation pixels (white text)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 150)).astype(np.uint8) * 255

    # Morphological closing to connect nearby character blobs into lines
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _close_kernel)

    # Find contours of connected text regions
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Filter for text-line-like regions and sort by area descending
    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 15 or bh < 5 or bh > 100:
            continue
        aspect = bw / max(bh, 1)
        if 0.5 < aspect < 50:
            candidates.append((x, y, bw, bh, bw * bh))

    candidates.sort(key=lambda r: r[4], reverse=True)

    # Recognize each candidate region
    for (rx, ry, rw, rh, _) in candidates[:10]:
        pad = 5
        x1 = max(0, rx - pad)
        y1 = max(0, ry - pad)
        x2 = min(w, rx + rw + pad)
        y2 = min(h, ry + rh + pad)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        rec_results, _ = _recognizer(crop)
        for text, conf in rec_results:
            if conf < 0.2 or not text.strip():
                continue
            if _fuzzy_match(text, target, max_dist=3):
                return [(x1, y1, x2, y2, text, float(conf))]

    return []


def get_video_info(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    parts = r.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    fps_num, fps_den = map(int, parts[2].split("/"))
    return fps_num / fps_den, w, h


def extract_frames(video_path, out_dir, fps):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}",
                     os.path.join(out_dir, "frame_%04d.png")],
                    capture_output=True, text=True, check=True)
    return sorted(Path(out_dir).glob("frame_*.png"))


def draw_box(image, x1, y1, x2, y2, text, conf):
    pad = 3
    cv2.rectangle(image, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), BOX_COLOR, BOX_THICKNESS)
    label = f"{text} ({conf:.0%})"
    (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    cv2.rectangle(image, (x1 - pad, y1 - th - pad * 3), (x1 + tw + pad, y1 - pad), BOX_COLOR, -1)
    cv2.putText(image, label, (x1, y1 - pad * 2), LABEL_FONT, LABEL_SCALE, (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA)
    return image


def reassemble(marked_dir, output_path, fps):
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps),
                     "-i", os.path.join(marked_dir, "frame_%04d.png"),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                     "-pix_fmt", "yuv420p", output_path],
                    capture_output=True, text=True, check=True)
    print(f"  Output: {output_path} ({os.path.getsize(output_path) / 1048576:.1f} MB)")


def main():
    os.chdir(Path(__file__).parent)

    print("[1/4] Reading video")
    fps, w, h = get_video_info(INPUT_VIDEO)
    print(f"  {w}x{h} @ {fps:.1f} fps")

    print("[2/4] Extracting frames")
    frames = extract_frames(INPUT_VIDEO, FRAMES_DIR, fps)
    print(f"  {len(frames)} frames")

    print(f'[3/4] Scanning for "{TARGET_WORD}"')
    os.makedirs(MARKED_DIR, exist_ok=True)
    found = 0
    import time
    t0 = time.time()
    for idx, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        matches = find_word(img, TARGET_WORD)
        if matches:
            found += 1
            for x1, y1, x2, y2, text, conf in matches:
                img = draw_box(img, x1, y1, x2, y2, text, conf)
        cv2.imwrite(str(Path(MARKED_DIR) / fp.name), img)
        if (idx + 1) % 30 == 0 or idx == len(frames) - 1:
            elapsed = time.time() - t0
            print(f"  ... {idx + 1}/{len(frames)}, {found} with \"{TARGET_WORD}\" ({elapsed:.1f}s)")
    print(f"  Found in {found}/{len(frames)} frames")

    print(f"[4/4] Reassembling into {OUTPUT_VIDEO}")
    reassemble(MARKED_DIR, OUTPUT_VIDEO, fps)

    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(MARKED_DIR, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()