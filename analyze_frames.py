#!/usr/bin/env python3
"""
analyze_frames.py — Dissect video, OCR each frame for "World", mark it, reassemble.

Engine system:
  Engines are defined in engines.json (pip package + detection wrapper).
  The script tracks which engines have been tested in engines_done.json.
  Each run picks the NEXT untested engine, installs it, runs the pipeline,
  and writes results_<engine>.json + output_<engine>.mp4.

  If OCR_ENGINE is set, it forces that specific engine (bypassing rotation).

Outputs:
  output_<engine>.mp4     — video with bounding boxes
  results_<engine>.json   — per-frame detection data
  engines_done.json       — list of engines already tested
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

SCRIPT_DIR = Path(__file__).parent
ENGINES_JSON = SCRIPT_DIR / "engines.json"
ENGINES_DONE_JSON = SCRIPT_DIR / "engines_done.json"
PYTHON_BIN = "/home/z/.venv/bin/python3"
PIP_BIN = f"{PYTHON_BIN} -m pip"

# ── Config ───────────────────────────────────────────────────────
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


# ── Engine registry ──────────────────────────────────────────────

def get_default_engines():
    """Default engine definitions. Each has: name, pip, setup (optional pre-install shell),
    detect (Python code string that defines find_word_<name>(image_bgr, target) -> list of tuples)."""
    return [
        {
            "name": "tesseract",
            "pip": "pytesseract",
            "detect": """
import pytesseract
import re
def find_word_tesseract(image_bgr, target):
    data = pytesseract.image_to_data(image_bgr, output_type=pytesseract.Output.DICT,
                                      config="--psm 11 --oem 3")
    matches = []
    for i in range(len(data["text"])):
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
""",
        },
        {
            "name": "paddleocr",
            "pip": "paddleocr paddlepaddle",
            "detect": """
import numpy as np
import re
def find_word_paddleocr(image_bgr, target):
    from paddleocr import PaddleOCR
    if not hasattr(find_word_paddleocr, "_ocr"):
        find_word_paddleocr._ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False,
                                              det_db_thresh=0.3, det_db_box_thresh=0.5)
    result = find_word_paddleocr._ocr.ocr(image_bgr, cls=False, det_db_thresh=0.3, det_db_box_thresh=0.5)
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
""",
        },
        {
            "name": "easyocr",
            "pip": "easyocr torch torchvision --index-url https://download.pytorch.org/whl/cpu",
            "detect": """
import numpy as np
import re
def find_word_easyocr(image_bgr, target):
    import easyocr
    if not hasattr(find_word_easyocr, "_reader"):
        find_word_easyocr._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    reader = find_word_easyocr._reader
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
""",
        },
        {
            "name": "doctr",
            "pip": "python-doctr",
            "detect": """
import numpy as np
import re
def find_word_doctr(image_bgr, target):
    from doctr.models import ocr_predictor
    if not hasattr(find_word_doctr, "_model"):
        find_word_doctr._model = ocr_predictor(pretrained=True)
    model = find_word_doctr._model
    # doctr expects RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    import doctr
    doc = doctr.io.Document.from_images(image_rgb)
    result = model(doc)
    matches = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                text = "".join(w.value for w in line.words)
                if re.search(re.escape(target), text, re.IGNORECASE):
                    # Get bounding box from geometry
                    geom = block.geometry  # ((x0, y0), (x1, y1))
                    (x0, y0), (x1, y1) = geom
                    x0, y0, x1, y1 = int(x0 * image_rgb.shape[1]), int(y0 * image_rgb.shape[0]), int(x1 * image_rgb.shape[1]), int(y1 * image_rgb.shape[0])
                    conf = min(w.confidence for w in line.words) if line.words else 0.0
                    matches.append((x0, y0, x1, y1, text, conf))
    return matches
""",
        },
        {
            "name": "keras_ocr",
            "pip": "keras-ocr",
            "detect": """
import numpy as np
import re
def find_word_keras_ocr(image_bgr, target):
    import keras_ocr
    if not hasattr(find_word_keras_ocr, "_pipeline"):
        find_word_keras_ocr._pipeline = keras_ocr.pipelines.Recognizer()
    pipeline = find_word_keras_ocr._pipeline
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # keras_ocr expects RGB float
    predictions = pipeline.recognize([image_rgb])
    matches = []
    for pred in predictions[0]:
        text, box = pred
        if re.search(re.escape(target), text, re.IGNORECASE):
            pts = np.array(box).astype(int)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            matches.append((int(x1), int(y1), int(x2), int(y2), text, 0.0))  # keras_ocr doesn't expose per-word conf easily
    return matches
""",
        },
        {
            "name": "rapidocr",
            "pip": "rapidocr-onnxruntime",
            "detect": """
import numpy as np
import re
def find_word_rapidocr(image_bgr, target):
    from rapidocr_onnxruntime import RapidOCR
    if not hasattr(find_word_rapidocr, "_engine"):
        find_word_rapidocr._engine = RapidOCR()
    engine = find_word_rapidocr._engine
    result, elapse = engine(image_bgr)
    if not result:
        return []
    matches = []
    for item in result:
        bbox_pts, text, conf = item[0], item[1], item[2]
        if re.search(re.escape(target), text, re.IGNORECASE):
            pts = np.array(bbox_pts).astype(int)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            matches.append((int(x1), int(y1), int(x2), int(y2), text, float(conf)))
    return matches
""",
        },
        {
            "name": "surya",
            "pip": "surya-ocr",
            "detect": """
import numpy as np
import re
def find_word_surya(image_bgr, target):
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
    from PIL import Image
    if not hasattr(find_word_surya, "_models"):
        det_model = load_det_model()
        det_processor = load_det_processor()
        rec_model = load_rec_model()
        rec_processor = load_rec_processor()
        find_word_surya._models = (det_model, det_processor, rec_model, rec_processor)
    det_model, det_processor, rec_model, rec_processor = find_word_surya._models
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    predictions = run_ocr([pil_img], [{}], det_model, det_processor, rec_model, rec_processor)[0]
    matches = []
    for pred in predictions:
        text = pred.text
        if re.search(re.escape(target), text, re.IGNORECASE):
            bbox = pred.bbox  # [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
            pts = np.array(bbox).astype(int)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            matches.append((int(x1), int(y1), int(x2), int(y2), text, pred.confidence if hasattr(pred, 'confidence') else 0.0))
    return matches
""",
        },
    ]


def load_engines():
    """Load engine list. If engines.json exists, use it. Otherwise create from defaults."""
    if ENGINES_JSON.exists():
        with open(ENGINES_JSON) as f:
            return json.load(f)
    engines = get_default_engines()
    with open(ENGINES_JSON, "w") as f:
        json.dump(engines, f, indent=2)
    return engines


def load_done():
    """Load set of engine names already tested."""
    if ENGINES_DONE_JSON.exists():
        with open(ENGINES_DONE_JSON) as f:
            return set(json.load(f))
    return set()


def mark_done(name):
    """Record that an engine has been tested."""
    done = load_done()
    done.add(name)
    with open(ENGINES_DONE_JSON, "w") as f:
        json.dump(sorted(done), f, indent=2)


def pick_engine(force=None):
    """Pick the next untested engine. If force is set, use that specific one."""
    engines = load_engines()
    if force:
        for e in engines:
            if e["name"] == force:
                return e
        print(f"  ERROR: engine '{force}' not found in registry")
        sys.exit(1)

    done = load_done()
    for e in engines:
        if e["name"] not in done:
            return e
    print("  All registered engines have been tested. Add more to engines.json or delete engines_done.json to restart.")
    sys.exit(0)


def install_engine(engine):
    """Install the engine's pip package. Returns True on success."""
    pip_cmd = engine.get("pip", "")
    if not pip_cmd:
        return True  # no install needed
    print(f"  Installing: pip install {pip_cmd}")
    # Handle --index-url flag
    parts = pip_cmd.split()
    cmd = [PYTHON_BIN, "-m", "pip", "install", "--quiet"]
    i = 0
    while i < len(parts):
        if parts[i] == "--index-url" and i + 1 < len(parts):
            cmd.extend(["--index-url", parts[i + 1]])
            i += 2
        elif parts[i].startswith("-"):
            cmd.append(parts[i])
            i += 1
        else:
            cmd.append(parts[i])
            i += 1
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  Install failed: {r.stderr.strip()}")
        return False
    print(f"  Installed successfully")
    return True


def load_detect_fn(engine):
    """Execute the engine's detect code and return the find_word function."""
    code = engine["detect"]
    ns = {"cv2": cv2, "np": np, "re": re}
    exec(code, ns)
    fn_name = f"find_word_{engine['name']}"
    if fn_name not in ns:
        print(f"  ERROR: detect code must define {fn_name}()")
        sys.exit(1)
    return ns[fn_name]


# ── Video pipeline ───────────────────────────────────────────────

def get_video_info(path):
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
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}",
           os.path.join(out_dir, "frame_%04d.png")]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    frames = sorted(Path(out_dir).glob("frame_*.png"))
    print(f"  Extracted {len(frames)} frames")
    return frames


def draw_box(image, x1, y1, x2, y2, text, conf):
    pad = 3
    cv2.rectangle(image, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad),
                  BOX_COLOR, BOX_THICKNESS)
    label = f"{text} ({conf:.0%})"
    (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    cv2.rectangle(image, (x1 - pad, y1 - th - pad * 3), (x1 + tw + pad, y1 - pad),
                  BOX_COLOR, -1)
    cv2.putText(image, label, (x1, y1 - pad * 2),
                LABEL_FONT, LABEL_SCALE, (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA)
    return image


def reassemble_video(marked_dir, output_path, fps, width, height):
    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", os.path.join(marked_dir, "frame_%04d.png"),
           "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-pix_fmt", "yuv420p", output_path]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output: {output_path} ({size_mb:.1f} MB)")


def main():
    os.chdir(SCRIPT_DIR)

    force_engine = os.environ.get("OCR_ENGINE", None)
    engine = pick_engine(force=force_engine)
    name = engine["name"]
    print(f"  OCR engine: {name}")

    # Install if needed
    if not install_engine(engine):
        print(f"  Aborting: could not install {name}")
        sys.exit(1)

    # Load detection function
    find_fn = load_detect_fn(engine)

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
        try:
            matches = find_fn(img, TARGET_WORD)
        except Exception as e:
            print(f"  ERROR on {fp.name}: {e}")
            matches = []
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
            print(f"  ... {idx + 1}/{len(frames)} frames, {found} with \"{TARGET_WORD}\"")

    avg_conf = (total_conf / conf_count) if conf_count else 0.0
    print(f"  Found \"{TARGET_WORD}\" in {found}/{len(frames)} frames")
    print(f"  Avg confidence: {avg_conf:.1%}")

    output_file = os.environ.get("OUTPUT_VIDEO", f"output_{name}.mp4")
    print(f"[4/4] Reassembling into {output_file}")
    reassemble_video(MARKED_DIR, output_file, fps, w, h)

    # Write results
    results = {
        "engine": name, "target": TARGET_WORD,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frames_total": len(frames), "frames_found": found,
        "avg_confidence": round(avg_conf, 4),
        "per_frame": per_frame,
    }
    results_path = f"results_{name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    # Mark engine as done
    mark_done(name)

    # Cleanup
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(MARKED_DIR, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()