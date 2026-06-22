#!/usr/bin/env python3
"""
analyze_frames.py — Dissect video, OCR each frame for "World", mark it, reassemble.
The find_word() function is rewritten by the agent before each run.
"""

import subprocess, os, re, shutil, tempfile
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

# --- Engine #14: DBNet score-map template tracking ---
# Two-phase approach:
# Phase 1 (frame 1): DBNet score map → Tesseract finds "World" position →
#   extract score-map template around "World".
# Phase 2 (frames 2+): DBNet score map → cv2.matchTemplate to track "World"
#   position in the score map. No Tesseract needed after initialization.
# This is ~35ms/frame (DBNet + template match) vs ~500ms/frame with Tesseract.
# Novel: score-map based template tracking for OCR word localization.
import onnxruntime as ort

_det_session = None
_template = None       # Score-map crop (numpy float32)
_tmpl_box_det = None   # (sx1, sy1, sx2, sy2) in score map coords
_tmpl_box_img = None   # (x1, y1, x2, y2) in original image coords
_tmpl_text = ""
_tmpl_conf = 0.0
_tmpl_center_det = None  # (cx, cy) in score map coords for search window

def _init_detector():
    global _det_session
    if _det_session is not None:
        return
    import time
    t0 = time.time()
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 4
    _det_session = ort.InferenceSession(
        "models/ch_PP-OCRv4_det_infer.onnx", so,
        providers=['CPUExecutionProvider'])
    print(f"  DBNet (onnxruntime) loaded in {time.time()-t0:.1f}s")


def _run_dbnet(img_bgr):
    """Run DBNet and return score map (det_h, det_w)."""
    det_h, det_w = 736, 1280
    img_resized = cv2.resize(img_bgr, (det_w, det_h))
    blob = (img_resized.astype(np.float32) / 255.0
            - np.array([0.485, 0.456, 0.406], dtype=np.float32)) \
           / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    inputs = {_det_session.get_inputs()[0].name: blob}
    score_map = _det_session.run(None, inputs)[0][0, 0]
    return score_map, det_h, det_w


def _tesseract_tsv(img_crop):
    """Run Tesseract PSM 7 TSV. Return list of (left,top,w,h,conf,text)."""
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        cv2.imwrite(tmp_path, img_crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        result = subprocess.run(
            ['tesseract', tmp_path, '-', 'tsv',
             '--psm', '7', '-l', 'eng', '--oem', '3'],
            capture_output=True, text=True, timeout=10)
        words = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) < 12 or parts[0] != '5':
                continue
            try:
                words.append((int(parts[6]), int(parts[7]),
                              int(parts[8]), int(parts[9]),
                              float(parts[10]) / 100.0,
                              parts[11]))
            except (ValueError, IndexError):
                continue
        return words
    except Exception:
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _fuzzy_match(text, target, max_dist=2):
    text_lower = text.lower().strip()
    target_lower = target.lower()
    tlen = len(target_lower)
    for i in range(len(text_lower) - tlen + 1):
        window = text_lower[i:i + tlen]
        diffs = sum(a != b for a, b in zip(window, target_lower))
        if diffs <= max_dist:
            return True
    return False


def find_word(img_bgr, target):
    """Engine #14: DBNet score-map template tracking.
    Return list of (x1, y1, x2, y2, text, conf).
    """
    global _template, _tmpl_box_det, _tmpl_box_img, _tmpl_text, _tmpl_conf
    global _tmpl_center_det

    _init_detector()
    h, w = img_bgr.shape[:2]
    score_map, det_h, det_w = _run_dbnet(img_bgr)
    scale_x = w / det_w
    scale_y = h / det_h

    # Phase 1: Initialize template from first detection
    if _template is None:
        # Row projection to find text line
        row_scores = score_map.sum(axis=1)
        if row_scores.max() == 0:
            return []
        text_y_det = int(np.argmax(row_scores))

        # Extract full-width strip from original image
        strip_half = 35
        y1_img = max(0, int(text_y_det * scale_y) - strip_half)
        y2_img = min(h, int(text_y_det * scale_y) + strip_half)
        strip = img_bgr[y1_img:y2_img, :]

        words = _tesseract_tsv(strip)
        for (left, top, bw, bh, conf, text) in words:
            if _fuzzy_match(text, target, max_dist=2):
                # Convert strip coordinates to score map coordinates
                sx1 = int(left / scale_x)
                sy1 = int((y1_img + top) / scale_y)
                sx2 = int((left + bw) / scale_x)
                sy2 = int((y1_img + top + bh) / scale_y)

                # Clamp to score map bounds
                sx1 = max(0, min(sx1, det_w - 1))
                sy1 = max(0, min(sy1, det_h - 1))
                sx2 = max(sx1 + 1, min(sx2, det_w))
                sy2 = max(sy1 + 1, min(sy2, det_h))

                _template = score_map[sy1:sy2, sx1:sx2].astype(np.float32).copy()
                _tmpl_box_det = (sx1, sy1, sx2, sy2)
                _tmpl_box_img = (left, y1_img + top, left + bw, y1_img + top + bh)
                _tmpl_text = text
                _tmpl_conf = conf
                _tmpl_center_det = ((sx1 + sx2) / 2, (sy1 + sy2) / 2)

                return [(left, y1_img + top, left + bw, y1_img + top + bh,
                          text, conf)]
        return []

    # Phase 2: Template match in score map
    th, tw = _template.shape
    tcx, tcy = _tmpl_center_det

    # Search window: template size + 100px margin in each direction
    margin = 100
    sw_x1 = max(0, int(tcx - tw / 2 - margin))
    sw_y1 = max(0, int(tcy - th / 2 - margin))
    sw_x2 = min(det_w, int(tcx + tw / 2 + margin))
    sw_y2 = min(det_h, int(tcy + th / 2 + margin))

    search_region = score_map[sw_y1:sw_y2, sw_x1:sw_x2].astype(np.float32)
    result = cv2.matchTemplate(search_region, _template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.3:
        return []

    # Map back to score map then to original image coordinates
    match_sx1 = sw_x1 + max_loc[0]
    match_sy1 = sw_y1 + max_loc[1]
    match_sx2 = match_sx1 + tw
    match_sy2 = match_sy1 + th

    ox1 = int(match_sx1 * scale_x)
    oy1 = int(match_sy1 * scale_y)
    ox2 = int(match_sx2 * scale_x)
    oy2 = int(match_sy2 * scale_y)

    # Update template center for next frame
    _tmpl_center_det = ((match_sx1 + match_sx2) / 2, (match_sy1 + match_sy2) / 2)

    return [(ox1, oy1, ox2, oy2, _tmpl_text, float(max_val))]


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