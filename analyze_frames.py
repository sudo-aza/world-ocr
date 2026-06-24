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

# --- Engine #38: Gamma correction + RapidOCR learned pipeline ---
# Gamma correction: pixel_out = 255 * (pixel_in / 255) ^ gamma.
# A standard, general-purpose nonlinear brightness/contrast adjustment.
# gamma < 1 brightens shadows (reveals dark text on dark bg),
# gamma > 1 darkens highlights. Gamma=0.8 is a mild shadow-boost used
# widely in document imaging. No thresholds, no edge detection.
# Different from #35 (bilateral), #36 (CLAHE), #37 (unsharp mask).
from rapidocr_onnxruntime import RapidOCR
import cv2
import numpy as np

_rapid = RapidOCR()
_GAMMA = 0.8
# Build 256-entry LUT for fast gamma correction
_gamma_lut = np.array([255 * ((i / 255.0) ** _GAMMA) for i in range(256)], dtype=np.uint8)


def _fuzzy_find_pos(text, target, max_dist=3):
    """Find the best matching position of target in text.
    Returns (start_char_index, end_char_index, matched_text) or None.
    """
    text_lower = text.lower().strip()
    target_lower = target.lower()
    tlen = len(target_lower)
    if len(text_lower) < tlen:
        return None
    best = None
    for i in range(len(text_lower) - tlen + 1):
        window = text_lower[i:i + tlen]
        diffs = sum(a != b for a, b in zip(window, target_lower))
        if diffs <= max_dist:
            if best is None or diffs < best[0]:
                best = (diffs, i, i + tlen, text[i:i + tlen])
    if best is None:
        return None
    return (best[1], best[2], best[3])


def _sub_bbox(bbox, text, start_idx, end_idx):
    """Slice a 4-point bbox proportionally to the character range [start_idx, end_idx)."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    text_len = len(text)
    if text_len == 0:
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # Proportional x-slicing based on character positions
    x1 = x_min + (x_max - x_min) * start_idx / text_len
    x2 = x_min + (x_max - x_min) * end_idx / text_len
    return int(x1), int(y_min), int(x2), int(y_max)


def find_word(img_bgr, target):
    """Engine #38: Gamma correction + RapidOCR with sub-word bounding box.
    General-purpose: works on any image with any target word.
    Return list of (x1, y1, x2, y2, text, conf).
    Bounding box is sliced proportionally to isolate just the target word.
    """
    # Apply gamma correction via LUT (fast, per-channel)
    corrected = cv2.LUT(img_bgr, _gamma_lut)
    results, _ = _rapid(corrected)
    matches = []
    for item in (results or []):
        bbox, text, conf = item
        if conf < 0.2 or not text.strip():
            continue
        pos = _fuzzy_find_pos(text, target, max_dist=3)
        if pos is None:
            continue
        start_idx, end_idx, matched_text = pos
        if start_idx == 0 and end_idx == len(text):
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            matches.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), matched_text, float(conf)))
        else:
            x1, y1, x2, y2 = _sub_bbox(bbox, text, start_idx, end_idx)
            matches.append((x1, y1, x2, y2, matched_text, float(conf)))
    return matches


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