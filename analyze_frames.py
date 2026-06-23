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

# --- Engine #16: DBNet score-map region detection + RapidOCR TextRecognizer ---
# Novel hybrid pipeline: Uses onnxruntime to run the DBNet detector and extract
# a raw score map (no DBPostProcess), then thresholds and finds the bounding box
# of text regions. Crops are fed to RapidOCR's TextRecognizer for word-level
# recognition. This is distinct from:
#   - Engine 1 (RapidOCR full pipeline): uses RapidOCR's own detector+recognizer
#   - Engines 5,7,8,11 (raw ONNX/cv2.dnn DBNet + CRNN): use DBPostProcess polygon detection
#   - Engine 14 (score-map template tracking): matches template, no per-frame recognition
#   - Engine 10 (cv2.dnn DBNet + Tesseract): different detector backend + different recognizer
# The novel aspect is: onnxruntime DBNet score-map for localization (not polygon detection)
# paired with RapidOCR's TextRecognizer (which handles preprocessing correctly)
# for recognition on the detected region.
import onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR

# DBNet detector via onnxruntime (for score map extraction)
_det_sess = ort.InferenceSession(
    '/home/z/my-project/world-ocr/models/ch_PP-OCRv4_det_infer.onnx',
    providers=['CPUExecutionProvider']
)
_det_input_name = _det_sess.get_inputs()[0].name

# RapidOCR's TextRecognizer (handles CRNN preprocessing correctly)
_rapid = RapidOCR()
_recognizer = _rapid.text_rec


def _fuzzy_match(text, target, max_dist=2):
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
    """Engine #16: DBNet score-map + RapidOCR recognizer.
    Return list of (x1, y1, x2, y2, text, conf).
    """
    h, w = img_bgr.shape[:2]

    # Run DBNet detector to get score map
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0 / 255., (640, 640),
                                 (0.485, 0.456, 0.406), True, False)
    score_map = _det_sess.run(['sigmoid_0.tmp_0'],
                               {_det_input_name: blob})[0][0, 0]

    # Threshold to find text regions
    text_mask = (score_map > 0.3).astype(np.uint8) * 255

    # Find contours of text regions
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Get bounding boxes, scaled from 640x640 to original image
    text_regions = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        sx = x * w / 640
        sy = y * h / 640
        sw = bw * w / 640
        sh = bh * h / 640
        if sw > 20 and sh > 5:
            text_regions.append((sx, sy, sw, sh))

    if not text_regions:
        return []

    # Sort by area (largest first — most likely to contain the target word)
    text_regions.sort(key=lambda r: r[2] * r[3], reverse=True)

    # Recognize each text region
    matches = []
    for (rx, ry, rw, rh) in text_regions:
        pad = 5
        x1 = max(0, int(rx) - pad)
        y1 = max(0, int(ry) - pad)
        x2 = min(w, int(rx + rw) + pad)
        y2 = min(h, int(ry + rh) + pad)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Use RapidOCR's recognizer
        rec_results, _ = _recognizer(crop)
        for text, conf in rec_results:
            if conf < 0.3 or not text.strip():
                continue
            if _fuzzy_match(text, target, max_dist=2):
                matches.append((x1, y1, x2, y2, text, conf))
                return matches  # First match

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