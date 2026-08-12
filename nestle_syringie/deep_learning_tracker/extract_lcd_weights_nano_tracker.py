#!/usr/bin/env python3
"""
Full pipeline test: ring-detection panel localization (unchanged) +
TrackerNano (deep-learning, lightweight ONNX Siamese tracker) instead of
TrackerMIL for per-frame tracking. Same digit decode / Excel / ocr_debug /
annotated_frames output as new_template_method_ring_detect.py - only the
tracker differs, so this isolates the tracker as the only variable versus
that script's results.

Reuses roi_cache_ring_test.json (read-only expectation for the 8 videos
already calibrated there) and all panel-detection/digit-decode functions
from new_template_method_ring_detect.py unchanged - only process_video()
is overridden here to use TrackerNano instead of TrackerMIL.

Usage (from inside nestle_syringie/deep_learning_tracker/):
    ../../.venv/bin/python extract_lcd_weights_nano_tracker.py <video1> <video2> ...

Output: test_nano_full/<video_key>/{annotated_frames,ocr_debug}/,
        test_nano_full/<video_key>/<video_key>_weights.xlsx
"""
import sys
from pathlib import Path
sys.path.append('/home/bee/projeler/lcd_reader/.venv/lib/python3.14/site-packages')
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import pandas as pd

from new_template_method_ring_detect import (
    load_template, load_roi_cache, get_video_calibration, video_key,
    read_weight_from_frame, TEMPLATE_MIN_SCORE, TRACKER_RECOVERY_LOST_FRAMES,
)

NANO_BACKBONE_PATH = Path(__file__).parent / "models" / "nanotrack_backbone.onnx"
NANO_HEAD_PATH = Path(__file__).parent / "models" / "nanotrack_head.onnx"
OUTDIR = Path(__file__).parent / "test_nano_full"


def make_nano_tracker():
    params = cv2.TrackerNano_Params()
    params.backbone = str(NANO_BACKBONE_PATH)
    params.neckhead = str(NANO_HEAD_PATH)
    return cv2.TrackerNano_create(params)


def process_video(video_path, initial_roi, rotation_deg, crop_fractions, output_excel, annotated_folder,
                   debug_folder=None, fps_target=10):
    Path(annotated_folder).mkdir(parents=True, exist_ok=True)
    if debug_folder is not None:
        Path(debug_folder).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"FPS: {fps}  Total frames: {total_frames}  ROI: {initial_roi}  Rotation: {rotation_deg:.2f} deg")

    results = []
    frame_interval = max(1, fps // fps_target)

    tracker = None
    current_box = initial_roi
    last_good_box = initial_roi
    lost_count = 0

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if tracker is None:
            tracker = make_nano_tracker()
            tracker.init(frame, current_box)
        else:
            ok, box = tracker.update(frame)
            if ok:
                current_box = tuple(int(v) for v in box)
                last_good_box = current_box
                lost_count = 0
            else:
                lost_count += 1
                if lost_count >= TRACKER_RECOVERY_LOST_FRAMES:
                    current_box = last_good_box
                    tracker = make_nano_tracker()
                    tracker.init(frame, current_box)
                    lost_count = 0

        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / fps

            raw_text, weight = read_weight_from_frame(
                frame, current_box, rotation_deg, crop_fractions, debug_folder, frame_count
            )

            results.append({
                'frame_no': frame_count,
                'timestamp_sec': timestamp_sec,
                'raw_digits': raw_text,
                'weight': weight
            })

            x, y, w, h = current_box
            annotated_frame = frame.copy()
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if weight is not None:
                cv2.putText(annotated_frame, f'Weight: {weight:.2f}',
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            output_path = f"{annotated_folder}/frame_{frame_count:06d}.png"
            cv2.imwrite(output_path, annotated_frame)

            print(f"Frame {frame_count}: raw='{raw_text}' Weight = {weight}, box={current_box}, lost={lost_count}")

        frame_count += 1

    cap.release()

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_excel(output_excel, index=False)
        print(f"Results saved to {output_excel}")
    else:
        print("No valid weight data found.")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", help="Video files to process")
    parser.add_argument("--rescan", action="store_true", help="Ignore cached panel detection and re-run detection for every video")
    parser.add_argument("--no-debug", action="store_true", help="Skip writing ocr_debug crop images (faster, less disk)")
    parser.add_argument("--fps-target", type=int, default=10, help="Sampled frames per second")
    args = parser.parse_args()

    template_gray = load_template()
    cache = load_roi_cache()
    summary = []

    for video_path in args.videos:
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"skip: {video_path} not found")
            continue

        roi, rotation_deg, crop_fractions, score = get_video_calibration(
            video_path, cache, template_gray, force_rescan=args.rescan
        )

        key = video_key(video_path)
        video_outdir = OUTDIR / key
        annotated_folder = video_outdir / "annotated_frames"
        debug_folder = None if args.no_debug else video_outdir / "ocr_debug"
        output_excel = video_outdir / f"{key}_weights.xlsx"

        results = process_video(
            video_path, roi, rotation_deg, crop_fractions, output_excel, annotated_folder,
            debug_folder, fps_target=args.fps_target
        )
        decoded = sum(1 for r in results if r['weight'] is not None)
        summary.append((key, len(results), decoded, score))

    print("\n=== Batch summary (TrackerNano) ===")
    for name, total, decoded, score in summary:
        pct = f"{100 * decoded / total:.0f}%" if total else "n/a"
        flag = "" if score >= TEMPLATE_MIN_SCORE else "  <-- low match confidence"
        print(f"{name:20s} sampled={total:4d} decoded={decoded:4d} ({pct})  match_score={score:.2f}{flag}")


if __name__ == "__main__":
    main()
