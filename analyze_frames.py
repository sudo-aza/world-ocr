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


def find_word(img, target):
    """Engine #8: PP-OCRv4 cv2.dnn DBNet + dilated score-mask box filtering.
    Variant of engine #7: uses 3x3 dilation on probability map before thresholding,
    DBBox-style mean-score filtering per contour (reject boxes below mean score 0.4),
    and greedy horizontal NMS merge before recognition.
    Return list of (x1, y1, x2, y2, text, conf).
    """
    import onnxruntime as ort
    import math

    if not hasattr(find_word, "_det_net"):
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        find_word._det_net = cv2.dnn.readNet(os.path.join(model_dir, "ch_PP-OCRv4_det_infer.onnx"))
        opts = ort.SessionOptions()
        opts.log_severity_level = 4
        opts.enable_cpu_mem_arena = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        find_word._rec_sess = ort.InferenceSession(
            os.path.join(model_dir, "ch_PP-OCRv4_rec_infer.onnx"),
            sess_options=opts, providers=["CPUExecutionProvider"])
        meta = find_word._rec_sess.get_modelmeta().custom_metadata_map
        find_word._charset = ["blank"] + meta["character"].splitlines() + [" "]

    det_net = find_word._det_net
    rec_sess = find_word._rec_sess
    charset = find_word._charset

    def _recognize(crop):
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None, 0
        crop_h, crop_w = crop.shape[:2]
        wh_ratio = crop_w / float(crop_h)
        img_width = max(int(48 * wh_ratio), 320)
        resized_w = img_width if math.ceil(48 * wh_ratio) > img_width else int(math.ceil(48 * wh_ratio))
        resized_crop = cv2.resize(crop, (resized_w, 48))
        normed = (resized_crop.astype(np.float32) / 255.0 - 0.5) / 0.5
        padded = np.zeros((3, 48, img_width), dtype=np.float32)
        padded[:, :, :resized_w] = normed.transpose((2, 0, 1))
        rec_out = rec_sess.run(None, {"x": padded[np.newaxis].astype(np.float32)})[0]
        preds_idx = rec_out[0].argmax(axis=1)
        preds_prob = rec_out[0].max(axis=1)
        text_chars, conf_vals, prev_idx = [], [], -1
        for t in range(len(preds_idx)):
            idx = int(preds_idx[t])
            if idx == 0 or idx == prev_idx:
                continue
            prev_idx = idx
            if idx < len(charset):
                text_chars.append(charset[idx])
                conf_vals.append(float(preds_prob[t]))
        text = "".join(text_chars).strip()
        return (text, float(np.mean(conf_vals))) if text else (None, 0)

    def _try_box(ox1, oy1, ox2, oy2):
        ox1, oy1 = max(0, ox1), max(0, oy1)
        ox2, oy2 = min(w, ox2), min(h, oy2)
        if ox2 - ox1 < 5 or oy2 - oy1 < 3:
            return None
        text, conf = _recognize(img[oy1:oy2, ox1:ox2])
        if text and re.search(re.escape(target), text, re.IGNORECASE):
            return (ox1, oy1, ox2, oy2, text, conf)
        return None

    h, w = img.shape[:2]
    max_side = 960
    ratio = min(max_side / max(h, w), 1.0)
    resize_h = int(round(h * ratio / 32) * 32)
    resize_w = int(round(w * ratio / 32) * 32)
    resized = cv2.resize(img, (resize_w, resize_h))
    det_input = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    det_input = det_input.transpose((2, 0, 1))[np.newaxis].astype(np.float32)
    det_net.setInput(det_input)
    pred = det_net.forward()[0, 0]

    pred_h, pred_w = pred.shape
    sx, sy = w / pred_w, h / pred_h

    # Post-processing: dilate probability map, then threshold with score filtering
    kernel = np.ones((3, 3), dtype=np.float32)
    dilated = cv2.dilate(pred, kernel, iterations=1)
    bitmap = (dilated > 0.3).astype(np.uint8) * 255
    if cv2.countNonZero(bitmap) == 0:
        return []
    outs = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[1] if len(outs) == 3 else outs[0]

    boxes = []
    for c in contours:
        c = c.reshape(-1, 2).astype(np.float32)
        if len(c) < 4:
            continue
        # DBBox-style: compute mean score inside contour
        mask = np.zeros((pred_h, pred_w), dtype=np.uint8)
        cv2.fillPoly(mask, [c.astype(np.int32)], 1)
        mean_score = pred[mask > 0].mean() if np.any(mask) else 0
        if mean_score < 0.4:
            continue
        rect = cv2.minAreaRect(c)
        pts = cv2.boxPoints(rect)
        x1 = max(0, int(min(p[0] for p in pts) * sx))
        y1 = max(0, int(min(p[1] for p in pts) * sy))
        x2 = min(w, int(max(p[0] for p in pts) * sx))
        y2 = min(h, int(max(p[1] for p in pts) * sy))
        if x2 - x1 < 10 or y2 - y1 < 5:
            continue
        boxes.append((x1, y1, x2, y2))
    boxes.sort(key=lambda b: b[0])
    if not boxes:
        return []

    # Greedy horizontal NMS merge
    merged = [boxes[0]]
    for b in boxes[1:]:
        last = merged[-1]
        if min(last[3], b[3]) - max(last[1], b[1]) > 0:
            merged[-1] = (last[0], min(last[1], b[1]), b[2], max(last[3], b[3]))
        else:
            merged.append(b)

    # Try merged boxes, then individual, then global merge
    for b in merged:
        r = _try_box(*b)
        if r:
            return [r]
    for b in boxes:
        r = _try_box(*b)
        if r:
            return [r]
    m = (min(b[0] for b in boxes), min(b[1] for b in boxes),
         max(b[2] for b in boxes), max(b[3] for b in boxes))
    r = _try_box(*m)
    if r:
        return [r]
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