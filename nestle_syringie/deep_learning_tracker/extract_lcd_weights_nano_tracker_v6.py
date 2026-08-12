#!/usr/bin/env python3
"""
v6: same as v5 (ring detection + TrackerNano + full pipeline + per-box
ratio split fix + per-video wide-right crop for the two Nestle-dataset
videos it was diagnosed on), plus a rewritten, GENERAL panel-localization
step - needed to make this pipeline work on a second, visually different
dataset (kivampro_10_20_60mL_deneyler/iddsi1/, phone-recorded, portrait
with pillarboxing, more camera tilt) without any per-video tuning.

Bug found running v5 as-is on iddsi1: 3 of 9 videos decoded at 2-7%. Not a
tracker problem - TrackerNano faithfully tracks whatever box it's given;
the box itself was wrong from frame 0. Root cause (confirmed visually via
the HSV border mask on iddsi1-20ml-1): the bezel-border ring IS a complete,
correctly-shaped rectangle in the mask, but glare on this dataset's
lighting breaks its color continuity at one edge, so
find_ring_component_wide's connectedComponentsWithStats (close kernel
7x7) sees it as two disconnected blobs and keeps only the smaller one -
confirmed literally half the true panel width (188px vs the true ~335px).

This needed a genuinely more robust localization step, not a bigger
constant for this one case - a single fixed kernel size traded one failure
for another (kernel=20 fixed iddsi1-20ml-1 but produced a wrong box on
iddsi1-60ml-2/60ml-3 elsewhere; see the exploration in this session).
Rewritten detect_panel_via_ring to:
  1. Try a RANGE of closing kernel sizes (7 to 31) instead of one fixed
     size, so it can bridge whatever size gap the glare left, on whichever
     video needs it, without needing to know that size in advance.
  2. Tighten the aspect-ratio acceptance window from the old (1.5, 5.0) to
     (2.3, 3.2) - measured across both datasets' known-correct boxes,
     which all cluster in this range; the old lower bound of 1.5 was loose
     enough to accept the broken half-box (aspect 1.62) as "plausible".
  3. Sample far more preview frames (15, spread 0.03-0.5 of the video
     instead of 5 frames over 0.05-0.35) and take the LARGEST valid
     candidate per frame (small stray objects - e.g. a power-cord clip
     that happens to match the border's hue - occasionally pass the
     shape filter too, but the real panel is reliably the largest
     candidate in whichever frame finds it cleanly).
  4. Cluster the resulting per-frame boxes by position (60px grid) and
     take the majority cluster's median, instead of blindly medianing
     every frame's answer - so a couple of frames hitting a stray object
     can't drag the result off the real panel.
Verified this doesn't regress the original Nestle dataset: spot-checked
4 already-validated boxes (lvl4_10mL, lvl1_20mL-2, lvl2_60mL-2,
lvl3_60mL-3) - all reproduce within a few pixels of their v5 values.

Second bug found after this fix (iddsi1-20ml-3 and iddsi1-60ml-1 still low
at 49%/15%): a stray thin/short blob between two real digits (visible in
ocr_debug/*_boxes.png as a noticeably shorter box sitting right next to a
tall '1' or between two digits - decimal-point glare or a compression
artifact) was passing the box height filter (H*0.25) and getting treated
as an unclassifiable 4th/5th "digit", poisoning the whole read (e.g.
"1.00" -> "1?00"). Measured real digit strokes at h/H=0.70-0.82 on BOTH
datasets, these stray blobs at 0.05-0.48 - raised the height floor to
H*0.55 (see read_weight_from_frame) to reject them everywhere, not just on
the videos where it was diagnosed.
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
    load_template, load_roi_cache, save_roi_cache, video_key,
    preprocess_digits_area, classify_digit, detect_panel_in_frame,
    calibrate_rotation_and_crop,
    BORDER_HSV_LOWER, BORDER_HSV_UPPER,
    WIDE_BORDER_MIN_FRAC, WIDE_BORDER_MAX_FRAC, WIDE_BORDER_MAX_FILL, WIDE_SEARCH_TOP_FRAC,
    TEMPLATE_MIN_SCORE, TRACKER_RECOVERY_LOST_FRAMES,
)

NANO_BACKBONE_PATH = Path(__file__).parent / "models" / "nanotrack_backbone.onnx"
NANO_HEAD_PATH = Path(__file__).parent / "models" / "nanotrack_head.onnx"
OUTDIR = Path(__file__).parent / "test_nano_full_v6"

# Per-video override of CROP_FRACTIONS' right boundary (base: 0.85, see
# module docstring). Only the two videos it was diagnosed/verified on get
# the wider 0.97 - applying it to every video pulled a stray "g" unit-label
# stroke into lvl2_60mL-2's crop and regressed it hard. This is specific to
# the original Nestle dataset; the new iddsi1 set gets no per-video crops
# at all - see module docstring for why that's a deliberate choice here.
CROP_FRACTIONS_WIDE_RIGHT_VIDEOS = {"lvl1_20mL-2", "lcd_reader_20mL-2"}

# Measured on this dataset (see module docstring): real single digits sit
# at width/height <= ~0.73 (95th pct 0.57); confirmed 2-digit merges sit at
# 0.87-0.99. 0.78 splits the gap between them.
WIDE_BOX_WH_THRESHOLD = 0.78
SINGLE_DIGIT_WH_RATIO = 0.55  # typical single-digit ratio, used to size N for >2-digit merges

# --- General panel-localization rewrite (see module docstring) ---
RING_VOTE_FRAMES = 15
RING_VOTE_FRAME_RANGE = (0.03, 0.5)
RING_CLOSE_KERNELS = (7, 11, 15, 19, 23, 27, 31)
RING_GOOD_ASPECT_RANGE = (2.3, 3.2)
RING_CLUSTER_GRID = 60  # px bucket size for grouping per-frame candidates
MAX_PLAUSIBLE_ROTATION_DEG = 10.0


def make_nano_tracker():
    params = cv2.TrackerNano_Params()
    params.backbone = str(NANO_BACKBONE_PATH)
    params.neckhead = str(NANO_HEAD_PATH)
    return cv2.TrackerNano_create(params)


def grab_vote_frames(video_path, n=RING_VOTE_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    lo, hi = RING_VOTE_FRAME_RANGE
    positions = [int(total * f) for f in np.linspace(lo, hi, n)]
    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def best_ring_candidate_in_frame(search_img):
    """
    Try a range of closing-kernel sizes (bridges whatever gap glare left in
    the border color, without needing to know its size ahead of time) and
    keep the LARGEST candidate that passes the shape/aspect filters - small
    stray objects occasionally pass the filters too, but the real panel is
    reliably the largest true positive when any kernel finds it cleanly.
    """
    hsv = cv2.cvtColor(search_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BORDER_HSV_LOWER, BORDER_HSV_UPPER)
    H, W = mask.shape
    best = None
    for close_k in RING_CLOSE_KERNELS:
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            bbox_area = w * h
            fill = area / bbox_area if bbox_area else 0
            frac = bbox_area / (H * W)
            aspect = w / h if h else 0
            if (WIDE_BORDER_MIN_FRAC < frac < WIDE_BORDER_MAX_FRAC and fill < WIDE_BORDER_MAX_FILL
                    and RING_GOOD_ASPECT_RANGE[0] < aspect < RING_GOOD_ASPECT_RANGE[1]):
                if best is None or frac > best[0]:
                    best = (frac, (x, y, w, h))
    return best


def detect_panel_via_ring_v2(frame):
    H, W = frame.shape[:2]
    yoff = int(H * WIDE_SEARCH_TOP_FRAC)
    search = frame[yoff:, :]
    found = best_ring_candidate_in_frame(search)
    if found is None:
        return None
    _, (x, y, w, h) = found
    return (x, y + yoff, w, h)


def cluster_pick(boxes):
    buckets = {}
    for b in boxes:
        x, y, w, h = b
        key = (round(x / RING_CLUSTER_GRID), round(y / RING_CLUSTER_GRID))
        buckets.setdefault(key, []).append(b)
    best_key = max(buckets, key=lambda k: len(buckets[k]))
    members = np.array(buckets[best_key])
    return tuple(int(v) for v in np.median(members, axis=0)), len(members), len(boxes)


def detect_panel_v2(video_path, template_gray):
    """
    Replaces new_template_method_ring_detect.detect_panel(): same overall
    strategy (ring detection first, template matching fallback) but with
    the more robust multi-kernel / multi-frame / majority-cluster ring
    detection described in the module docstring. Falls back to template
    matching only if not even one preview frame yields a plausible ring
    candidate.
    """
    frames = grab_vote_frames(video_path)
    per_frame_boxes = [detect_panel_via_ring_v2(f) for f in frames]
    per_frame_boxes = [b for b in per_frame_boxes if b is not None]

    if per_frame_boxes:
        box, votes, total = cluster_pick(per_frame_boxes)
        print(f"    ring: {votes}/{total} frames agree on {box}")
        return 1.0, box

    results = [detect_panel_in_frame(f, template_gray) for f in frames]
    best = max(results, key=lambda r: r[0])
    print(f"    ring found nothing - template matching fallback, score={best[0]:.2f}")
    return best


def get_video_calibration_v2(video_path, cache, template_gray, force_rescan=False):
    key = video_key(video_path)
    if not force_rescan and key in cache:
        entry = cache[key]
        return (tuple(entry["roi"]), entry["rotation_deg"],
                tuple(entry["crop_fractions"]), entry.get("score", 1.0))

    score, roi = detect_panel_v2(video_path, template_gray)
    rotation_deg, crop_fractions, used_ring = calibrate_rotation_and_crop(video_path, roi)
    if abs(rotation_deg) > MAX_PLAUSIBLE_ROTATION_DEG:
        # Every rotation measured across both datasets so far sits within a
        # couple degrees of level (up to ~2 deg on a tripod, a bit more on
        # handheld phone footage) - a scale sitting on a table doesn't tilt
        # 30+ degrees relative to the camera. When this happens the ring
        # couldn't be found on any preview frame either (same root cause as
        # a failed panel detection - glare/low score for this video), so
        # the estimate fell through to the Hough-line fallback and picked
        # up some other diagonal edge (a glare streak, an object edge) as
        # if it were the panel's own tilt. Treat it as unreliable and
        # assume level rather than apply a wild rotation to an
        # already-level image.
        print(f"    rotation {rotation_deg:.1f} deg implausible - resetting to 0.0")
        rotation_deg = 0.0
    method = "ring-detected" if used_ring else "FALLBACK: Hough+default crop, check this one"
    flag = "" if score >= TEMPLATE_MIN_SCORE else "  <-- LOW MATCH CONFIDENCE, check this one"
    print(f"[{key}] roi={roi} rotation={rotation_deg:.2f} deg  crop={tuple(round(c,2) for c in crop_fractions)}"
          f"  ({method})  match_score={score:.2f}{flag}")

    cache[key] = {
        "roi": list(roi), "rotation_deg": rotation_deg,
        "crop_fractions": list(crop_fractions), "score": score,
    }
    save_roi_cache(cache)
    return roi, rotation_deg, crop_fractions, score


def split_wide_boxes(boxes, opened):
    """
    Judge each box independently by its own width/height ratio (see module
    docstring for the measured thresholds) rather than comparing against
    sibling boxes in the same frame - a merged blob of N digits is split
    into N sub-boxes, cutting at the locally sparsest column (fewest
    foreground pixels) near each expected split point in the pre-close
    ("opened") mask, which still preserves the real (if narrow) inter-digit
    gap that only the wide CLOSE step bridged over.
    """
    result = []
    for (bx, by, bw, bh) in boxes:
        ratio = bw / bh if bh else 0
        if ratio <= WIDE_BOX_WH_THRESHOLD:
            result.append((bx, by, bw, bh))
            continue

        n = max(2, round(ratio / SINGLE_DIGIT_WH_RATIO))
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
    # Height floor raised from H*0.25 to H*0.55 - measured real digit strokes
    # (both datasets) cluster at h/H = 0.70-0.82, while stray noise specks
    # between digits (decimal-point glare, compression artifacts) measured
    # at 0.05-0.48. The old 0.25 floor caught the smallest specks but let
    # moderate ones on iddsi1 through as a spurious extra "digit" that fails
    # classify_digit and poisons the whole read (e.g. "1.00" -> "1?00").
    # 0.55 sits safely in the gap between both clusters on both datasets.
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[3] > H * 0.55 and b[2] > 3 and b[2] < W * 0.5]
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
    parser.add_argument("--outdir", default=None, help="Override the default OUTDIR (test_nano_full_v5/ next to this script)")
    args = parser.parse_args()
    outdir = Path(args.outdir) if args.outdir else OUTDIR

    template_gray = load_template()
    cache = load_roi_cache()
    summary = []

    for video_path in args.videos:
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"skip: {video_path} not found")
            continue

        roi, rotation_deg, crop_fractions, score = get_video_calibration_v2(
            video_path, cache, template_gray, force_rescan=args.rescan
        )
        key = video_key(video_path)
        if key in CROP_FRACTIONS_WIDE_RIGHT_VIDEOS:
            crop_fractions = (0.20, 0.93, 0.28, 0.97)
        video_outdir = outdir / key
        annotated_folder = video_outdir / "annotated_frames"
        debug_folder = None if args.no_debug else video_outdir / "ocr_debug"
        output_excel = video_outdir / f"{key}_weights.xlsx"

        results = process_video(
            video_path, roi, rotation_deg, crop_fractions, output_excel, annotated_folder,
            debug_folder, fps_target=args.fps_target
        )
        decoded = sum(1 for r in results if r['weight'] is not None)
        summary.append((key, len(results), decoded, score))

    print("\n=== Batch summary (TrackerNano + split fix + general multi-kernel ring detection) ===")
    for name, total, decoded, score in summary:
        pct = f"{100 * decoded / total:.0f}%" if total else "n/a"
        flag = "" if score >= TEMPLATE_MIN_SCORE else "  <-- low match confidence"
        print(f"{name:20s} sampled={total:4d} decoded={decoded:4d} ({pct})  match_score={score:.2f}{flag}")


if __name__ == "__main__":
    main()
