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
    """Find target word using OCR.space cloud API (Engine 2). Return list of (x1,y1,x2,y2,text,conf)."""
    import requests, base64, time, io, hashlib
    from PIL import Image
    if not hasattr(find_word, "_s"):
        find_word._s = requests.Session()
        find_word._s.headers["apikey"] = "helloworld"
        find_word._t = 0
        find_word._cache = {}
        find_word._call_count = 0
    # Frame skip cache: only call API every 5th unique frame
    small = cv2.resize(img, (64, 36))
    h = hashlib.md5(small.tobytes()).hexdigest()
    if h in find_word._cache:
        return find_word._cache[h]
    find_word._call_count += 1
    if find_word._call_count % 5 != 0 and find_word._cache:
        # Reuse last result for skipped frames
        find_word._cache[h] = list(find_word._cache.values())[-1] if find_word._cache else []
        return find_word._cache[h]
    elapsed = time.time() - find_word._t
    if elapsed < 0.55:
        time.sleep(0.55 - elapsed)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=75)
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        r = find_word._s.post(
            "https://api.ocr.space/parse/image",
            data={"base64Image": f"data:image/jpeg;base64,{b64}",
                  "language": "eng", "isOverlayRequired": "true",
                  "OCREngine": "2"},
            timeout=10)
        r.raise_for_status()
    except Exception:
        find_word._t = time.time()
        find_word._cache[h] = []
        return []
    find_word._t = time.time()
    data = r.json()
    matches = []
    if data.get("IsErroredOnProcessing"):
        find_word._cache[h] = []
        return []
    for res in data.get("ParsedResults", []):
        for line in res.get("TextOverlay", {}).get("Lines", []):
            for w in line.get("Words", []):
                text = w.get("WordText", "")
                if re.search(re.escape(target), text, re.IGNORECASE):
                    x1 = int(w["Left"])
                    y1 = int(w["Top"])
                    x2 = x1 + int(w["Width"])
                    y2 = y1 + int(w["Height"])
                    conf = float(w.get("Confidence") or 0) / 100.0
                    matches.append((x1, y1, x2, y2, text, conf))
    find_word._cache[h] = matches
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