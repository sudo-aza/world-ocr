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

# --- Engine #13: DBNet score-map row-scanning + EasyOCR attention-LSTM recognition ---
# Novel two-stage approach:
# Stage 1 (fast, ~30ms): DBNet produces a text probability heatmap via cv2.dnn.
#   Row-wise score aggregation locates the dominant text line's Y-position.
# Stage 2 (novel): Crop a wide strip from the original image at that Y-position
#   and feed to EasyOCR's ResNet+BiLSTM+Attention decoder — a fundamentally
#   different recognition architecture from the CRNN+CTC used in engines 7/8/11.
import easyocr

_det_net = None
_recognizer = None

def _init_models():
    global _det_net, _recognizer
    if _det_net is not None:
        return
    import time
    t0 = time.time()
    _det_net = cv2.dnn.readNetFromONNX("models/ch_PP-OCRv4_det_infer.onnx")
    _det_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    _recognizer = easyocr.Reader(['en'], gpu=False, verbose=False)
    print(f"  Models loaded in {time.time()-t0:.1f}s (DBNet scorer + EasyOCR AttentionLSTM)")

def _fuzzy_match(text, target, max_dist=2):
    """Check if target appears in text with up to max_dist character edits."""
    text_lower = text.lower()
    target_lower = target.lower()
    tlen = len(target_lower)
    for i in range(len(text_lower) - tlen + 1):
        window = text_lower[i:i + tlen]
        diffs = sum(a != b for a, b in zip(window, target_lower))
        if diffs <= max_dist:
            return True
    return False

def _locate_text_strips(score_map, img_h, img_w, strip_half=35, top_n=3):
    """Find top-N widest text regions from DBNet score map.
    Returns list of (y1, y2, x1, x2) sorted by width descending.
    """
    binary = (score_map > 0.3).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)
    num_labels, labels = cv2.connectedComponents(binary)
    components = []
    for lid in range(1, num_labels):
        py, px = np.where(labels == lid)
        if len(px) < 10:
            continue
        x1, y1 = int(px.min()), int(py.min())
        x2, y2 = int(px.max()), int(py.max())
        components.append((x2 - x1, x1, y1, x2, y2))
    components.sort(reverse=True)
    strips = []
    for (bw, x1, y1, x2, y2) in components[:top_n]:
        strips.append((y1, y2, x1, x2))
    return strips

def find_word(img, target):
    """Engine #13: DBNet score-map + EasyOCR attention-LSTM.
    Return list of (x1, y1, x2, y2, text, conf).
    """
    _init_models()
    h, w = img.shape[:2]

    # Stage 1: DBNet score map
    blob = cv2.dnn.blobFromImage(img, 1.0/255., (1280, 736),
                                 (0.485, 0.456, 0.406), swapRB=True)
    _det_net.setInput(blob)
    score_map = _det_net.forward("sigmoid_0.tmp_0")[0, 0]

    # Locate text regions (widest first)
    strips = _locate_text_strips(score_map, h, w, top_n=3)
    if not strips:
        return []

    # Stage 2: Try each region with EasyOCR
    matches = []
    for (y1, y2, x1, x2) in strips:
        pad = 10
        crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        try:
            results = _recognizer.recognize(rgb_crop)
        except Exception:
            continue
        for (pts, text, conf) in results:
            if _fuzzy_match(text, target, max_dist=2):
                matches.append((x1, y1, x2, y2, text, float(conf)))
                break
        if matches:
            break
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
            print(f"  ... {idx + 1}/{len(frames)}, {found} with \"{TARGET_WORD}\"")
    print(f"  Found in {found}/{len(frames)} frames")

    print(f"[4/4] Reassembling into {OUTPUT_VIDEO}")
    reassemble(MARKED_DIR, OUTPUT_VIDEO, fps)

    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(MARKED_DIR, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()