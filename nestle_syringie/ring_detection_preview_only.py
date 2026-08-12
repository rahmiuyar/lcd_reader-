#!/usr/bin/env python3
"""
Lightweight preview: ring-detection panel localization + tracking only.
No digit cropping, no OCR/segment decode, no Excel output - just draws the
tracked green box on each sampled frame and saves it, for fast visual
inspection of how well ring detection + tracking follows the LCD panel.

Usage (from inside nestle_syringie/):
    ../.venv/bin/python ring_detection_preview_only.py <video1> <video2> ...

Output: test_ring_detection/<video_key>/annotated_frames/frame_NNNNNN.png
Reuses the same roi_cache_ring_test.json as new_template_method_ring_detect.py
so a video already calibrated there doesn't need re-detection here.
"""
import sys
sys.path.append('/home/bee/projeler/lcd_reader/.venv/lib/python3.14/site-packages')

import json
from pathlib import Path
import cv2

from new_template_method_ring_detect import (
    detect_panel_via_ring, grab_preview_frames, video_key,
)

ROI_CACHE_PATH = Path(__file__).parent / "roi_cache_ring_test.json"
OUTDIR = Path(__file__).parent / "test_ring_detection"


def load_cache():
    if ROI_CACHE_PATH.exists():
        return json.loads(ROI_CACHE_PATH.read_text())
    return {}


def save_cache(cache):
    ROI_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def get_roi(video_path, cache):
    key = video_key(video_path)
    if key in cache and "roi" in cache[key]:
        return tuple(cache[key]["roi"])

    frames = grab_preview_frames(video_path)
    boxes = [detect_panel_via_ring(f) for f in frames]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    import numpy as np
    roi = tuple(int(v) for v in np.median(np.array(boxes), axis=0))
    cache.setdefault(key, {})["roi"] = list(roi)
    save_cache(cache)
    return roi


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
    lost_count = 0
    saved = 0

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if tracker is None:
            tracker = cv2.TrackerMIL_create()
            tracker.init(frame, current_box)
        else:
            ok, box = tracker.update(frame)
            if ok:
                current_box = tuple(int(v) for v in box)
                lost_count = 0
            else:
                lost_count += 1

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
        print("usage: ring_detection_preview_only.py <video1> [video2 ...]")
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
