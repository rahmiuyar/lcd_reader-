#!/usr/bin/env python3
"""
Lightweight preview: ring-detection panel localization + TrackerVit
(deep-learning, ONNX-based) tracking, instead of TrackerMIL. No digit
cropping, no OCR/segment decode, no Excel output - just draws the tracked
green box on each sampled frame and saves it, for fast visual comparison
against the MIL-tracker preview already generated in
nestle_syringie/test_ring_detection/.

Usage (from inside nestle_syringie/deep_learning_tracker/):
    ../../.venv/bin/python vit_tracker_preview_only.py <video1> <video2> ...

Output: test_vit_tracker/<video_key>/annotated_frames/frame_NNNNNN.png
Reuses the SAME initial panel box already found for each video in
../roi_cache_ring_test.json (ring detection is unchanged - only the
per-frame tracker differs), so this isolates the tracker as the only
variable in the comparison.
"""
import sys
from pathlib import Path
sys.path.append('/home/bee/projeler/lcd_reader/.venv/lib/python3.14/site-packages')
sys.path.append(str(Path(__file__).parent.parent))

import json
import cv2

from new_template_method_ring_detect import (
    detect_panel_via_ring, grab_preview_frames, video_key,
)

RING_CACHE_PATH = Path(__file__).parent.parent / "roi_cache_ring_test.json"
VIT_MODEL_PATH = Path(__file__).parent / "models" / "vittrack.onnx"
OUTDIR = Path(__file__).parent / "test_vit_tracker"


def load_cache():
    if RING_CACHE_PATH.exists():
        return json.loads(RING_CACHE_PATH.read_text())
    return {}


def get_roi(video_path, cache):
    """Reuse the already-detected ring box from roi_cache_ring_test.json if
    present (keeps the comparison to MIL fair - same starting box), else
    detect fresh via ring detection."""
    key = video_key(video_path)
    if key in cache and "roi" in cache[key]:
        return tuple(cache[key]["roi"])

    frames = grab_preview_frames(video_path)
    boxes = [detect_panel_via_ring(f) for f in frames]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    import numpy as np
    return tuple(int(v) for v in np.median(np.array(boxes), axis=0))


def make_vit_tracker():
    params = cv2.TrackerVit_Params()
    params.net = str(VIT_MODEL_PATH)
    return cv2.TrackerVit_create(params)


def process_video_preview(video_path, roi, out_dir, fps_target=10):
    annotated_folder = out_dir / "annotated_frames"
    annotated_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}  FPS: {fps}  Total frames: {total_frames}  ROI: {roi}")

    frame_interval = max(1, fps // fps_target)
    tracker = None
    current_box = roi
    saved = 0

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if tracker is None:
            tracker = make_vit_tracker()
            tracker.init(frame, current_box)
        else:
            ok, box = tracker.update(frame)
            if ok:
                current_box = tuple(int(v) for v in box)

        if frame_count % frame_interval == 0:
            x, y, w, h = current_box
            annotated = frame.copy()
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(str(annotated_folder / f"frame_{frame_count:06d}.png"), annotated)
            saved += 1

        frame_count += 1

    cap.release()
    print(f"  saved {saved} annotated frames -> {annotated_folder}")


def main():
    videos = sys.argv[1:]
    if not videos:
        print("usage: vit_tracker_preview_only.py <video1> [video2 ...]")
        return
    cache = load_cache()
    for video_path in videos:
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"skip: {video_path} not found")
            continue
        roi = get_roi(video_path, cache)
        if roi is None:
            print(f"skip: {video_path.name} (ring not found on any preview frame)")
            continue
        key = video_key(video_path)
        out_dir = OUTDIR / key
        process_video_preview(video_path, roi, out_dir)


if __name__ == "__main__":
    main()
