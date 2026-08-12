#!/usr/bin/env python3
"""
v2: same as extract_lcd_weights_nano_tracker.py (ring detection + TrackerNano
+ full pipeline), plus a fix for a digit-merge bug found by inspecting this
run's ocr_debug images.

Bug found: the ring-detected panel box is systematically ~10-16% smaller
(in pixels) than the old template-matched box used previously (e.g.
lvl3_60mL-3: w=373 vs w=442; lvl2_60mL-1: w=351 vs w=416 - see roi_cache.json
vs roi_cache_ring_test.json). Since CROP_FRACTIONS and the 4x upscale in
preprocess_digits_area are applied identically regardless of the source
box's absolute size, the resulting digit crop is also proportionally
smaller for ring-detected boxes - and some digit *pairs* on this LCD sit
naturally close together (measured as little as ~2-5px apart at 4x scale on
failing frames, e.g. the "0" before a "9"/"0"). The existing close kernel
(25x25, a FIXED pixel size, not scaled to the crop) was tuned against the
old, larger box and reliably bridges that gap now, fusing two adjacent
digits into one wide blob that classify_digit can't recognize (or that
python's classify_digit spits back as a nearest-neighbor guess against the
wrong pattern) - producing "?0" for what should read "000", "?8" for "098",
etc. Confirmed on frame 18 of lvl3_60mL-3 and frame 30 of lvl2_60mL-1: the
merged blob's width is ~2.0x a normal single-digit width in both cases
(347px vs 179px, and 322px vs 164px respectively) - a very consistent
signature, not a coincidence.

Fix: after finding boxes (post fragment-merge, same as before), check each
box's width against the MEDIAN box width in that frame. Any box more than
1.6x the median is very likely N merged digits, not one. Split it into N =
round(width / median) pieces by finding the locally darkest-column valley
near each expected split point in the pre-close ("opened") mask - which
still preserves the real (if narrow) gap between the two digits, since only
the wide CLOSE step (not the 3x3 open) erases it. This is scale-invariant
(works whatever the box's absolute pixel size is) rather than needing the
kernel size retuned per video/box-detection-method.
"""
import sys
from pathlib import Path
sys.path.append('/home/bee/projeler/lcd_reader/.venv/lib/python3.14/site-packages')
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np
import pandas as pd

from new_template_method_ring_detect import (
    load_template, load_roi_cache, get_video_calibration, video_key,
    preprocess_digits_area, classify_digit,
    TEMPLATE_MIN_SCORE, TRACKER_RECOVERY_LOST_FRAMES,
)

NANO_BACKBONE_PATH = Path(__file__).parent / "models" / "nanotrack_backbone.onnx"
NANO_HEAD_PATH = Path(__file__).parent / "models" / "nanotrack_head.onnx"
OUTDIR = Path(__file__).parent / "test_nano_full_v2"

WIDE_BOX_RATIO = 1.6  # a box wider than this x the frame's median box width is treated as N merged digits


def make_nano_tracker():
    params = cv2.TrackerNano_Params()
    params.backbone = str(NANO_BACKBONE_PATH)
    params.neckhead = str(NANO_HEAD_PATH)
    return cv2.TrackerNano_create(params)


def split_wide_boxes(boxes, opened):
    """
    Split any box that's a multiple of the frame's typical single-digit
    width into that many sub-boxes, cutting at the locally sparsest column
    (fewest foreground pixels) near each expected split point - the
    surviving trace of the real inter-digit gap that the closing kernel
    bridged over. No-op if there's nothing to split against (<2 boxes) or
    every box is already digit-sized.
    """
    if len(boxes) < 2:
        return boxes

    median_w = float(np.median([b[2] for b in boxes]))
    if median_w <= 0:
        return boxes

    result = []
    for (bx, by, bw, bh) in boxes:
        n = round(bw / median_w)
        if n < 2 or bw < median_w * WIDE_BOX_RATIO:
            result.append((bx, by, bw, bh))
            continue

        region = opened[by:by + bh, bx:bx + bw]
        col_density = (region > 0).sum(axis=0).astype(np.int32)
        seg_w = bw / n

        splits = [0]
        for i in range(1, n):
            center = int(i * seg_w)
            lo = max(splits[-1] + 1, int(center - seg_w * 0.4))
            hi = min(bw - 1, int(center + seg_w * 0.4))
            if hi <= lo:
                splits.append(center)
                continue
            local_min = lo + int(np.argmin(col_density[lo:hi]))
            splits.append(local_min)
        splits.append(bw)

        for i in range(len(splits) - 1):
            x0, x1 = splits[i], splits[i + 1]
            if x1 > x0:
                result.append((bx + x0, by, x1 - x0, bh))

    result.sort(key=lambda b: b[0])
    return result


def read_weight_from_frame(frame, roi_coords, rotation_deg, crop_fractions, debug_folder=None, frame_count=None):
    x, y, w, h = roi_coords
    roi = frame[y:y + h, x:x + w]

    processed = preprocess_digits_area(roi, rotation_deg, crop_fractions)

    if debug_folder is not None:
        cv2.imwrite(f"{debug_folder}/frame_{frame_count:06d}_roi.png", roi)
        cv2.imwrite(f"{debug_folder}/frame_{frame_count:06d}_proc.png", processed)

    digit_mask = cv2.bitwise_not(processed)
    opened = cv2.morphologyEx(digit_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    kernel = np.ones((25, 25), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = processed.shape
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[3] > H * 0.25 and b[2] > 3 and b[2] < W * 0.5]
    boxes.sort(key=lambda b: b[0])

    merged = []
    for b in boxes:
        bx, by, bw, bh = b
        if merged:
            mx, my, mw, mh = merged[-1]
            overlap = min(mx + mw, bx + bw) - max(mx, bx)
            if overlap > 0.5 * min(mw, bw):
                nx0, ny0 = min(mx, bx), min(my, by)
                nx1, ny1 = max(mx + mw, bx + bw), max(my + mh, by + bh)
                merged[-1] = (nx0, ny0, nx1 - nx0, ny1 - ny0)
                continue
        merged.append(b)
    boxes = merged

    boxes = split_wide_boxes(boxes, opened)

    if debug_folder is not None:
        vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
        cv2.imwrite(f"{debug_folder}/frame_{frame_count:06d}_boxes.png", vis)

    digits = []
    for (bx, by, bw, bh) in boxes:
        digit_img = digit_mask[by:by + bh, bx:bx + bw]
        digits.append(classify_digit(digit_img))

    raw_text = ''.join(str(d) if d is not None else '?' for d in digits)

    if not digits or any(d is None for d in digits):
        return raw_text, None

    try:
        weight = int(''.join(str(d) for d in digits)) / 100.0
    except ValueError:
        return raw_text, None

    if weight < 0 or weight > 3000:
        return raw_text, None

    return raw_text, weight


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

    print("\n=== Batch summary (TrackerNano + split-wide-box fix) ===")
    for name, total, decoded, score in summary:
        pct = f"{100 * decoded / total:.0f}%" if total else "n/a"
        flag = "" if score >= TEMPLATE_MIN_SCORE else "  <-- low match confidence"
        print(f"{name:20s} sampled={total:4d} decoded={decoded:4d} ({pct})  match_score={score:.2f}{flag}")


if __name__ == "__main__":
    main()
