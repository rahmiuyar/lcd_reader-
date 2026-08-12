#!/usr/bin/env python3
"""
LCD weight reader - batch version - v1.1

Only change from the base script: the 7-segment fill threshold in
classify_digit() is 0.35 instead of 0.40. Found on lvl3/20mL-3.mp4, where a
sustained run of frames misread "9" as "5" (segment b/top-right measured
0.385 fill - just under the 0.40 cutoff) causing a physically-impossible
sustained drop in the decoded weight (0.77 -> 0.65g over ~13 frames, since
liquid is being dispensed and true weight can only increase). Confirmed no
regression on the original reference video (1.mp4: 284/285 -> 285/285) and
an improvement on the video that prompted it (89% -> 91%), but NOT yet
validated across the rest of the ~38-video dataset, so kept as a separate
version rather than folded into the base script - promote it only after
broader validation.

LCD weight reader - batch version.

Fully automatic, no clicking. The things that had to be hand-measured and
hardcoded per video are now all derived from one reference image
(panel_template.png, a known-good crop of the display from 1.mp4) and cached
per video in roi_cache.json so re-runs are instant:
  1. Panel box: located per video via multi-scale template matching against
     panel_template.png (tried at many scale factors since camera zoom/
     distance differs per recording; picked by best normalized correlation,
     voted across several frames for robustness against a hand or reflection
     briefly covering the display in any single frame).
  2. Rotation: auto-measured from that box via Hough line detection on the
     bezel border.
  3. Digit crop: a single fixed fraction of the panel box (top, bottom,
     left, right), also measured once from the reference template. This
     works reliably now (it did NOT for manually-drawn boxes) because every
     detected box is a uniformly-scaled copy of the exact same reference
     rectangle - so the digits always sit at the same relative position
     within it, unlike hand-drawn boxes whose margins vary video to video.

If template matching can't find a confident match on a video (very different
framing/lighting than the reference), it's reported in the batch summary
rather than silently producing a bad crop - re-save panel_template.png from
a clean frame of that video and re-run with --rescan if that happens.

Usage (run from inside nestle_syringie/, videos live one level up):
    ../.venv/bin/python extract_lcd_weights_batch.py ../1.mp4 ../2.mp4 ../20mL-*.mp4
    ../.venv/bin/python extract_lcd_weights_batch.py ../*.mp4 --rescan
    ../.venv/bin/python extract_lcd_weights_batch.py ../*.mp4 --no-debug

roi_cache.json and batch_output/ are created next to this script, inside
nestle_syringie/, regardless of what directory you run it from.
"""

import sys
sys.path.append('/home/bee/projeler/lcd_reader/.venv/lib/python3.14/site-packages')

import argparse
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

ROI_CACHE_PATH = Path(__file__).parent / "roi_cache.json"
TEMPLATE_PATH = Path(__file__).parent / "panel_template.png"

DIGIT_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 0, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}

FONT_SHEAR = 0.21  # measured italic slant: x shifts by ~0.21px per y px

# Digit-only crop as a fraction of the panel template (top, bottom, left,
# right) - excludes the bezel border, "WL-3002L..." label, icons, "g" unit.
# Valid for any template-matched box since matching only ever scales this
# same reference rectangle uniformly, never distorts its proportions.
CROP_FRACTIONS = (0.20, 0.93, 0.28, 0.85)

# Multi-scale template search range and how many preview frames to vote
# across when locating the panel per video.
TEMPLATE_SCALE_RANGE = (0.5, 2.0)
TEMPLATE_SCALE_STEP = 0.05
TEMPLATE_VOTE_FRAMES = 5
TEMPLATE_MIN_SCORE = 0.15  # below this, treat the match as unreliable

# If the tracker loses the target for this many consecutive sampled frames,
# snap back to the last known-good box and keep going instead of drifting
# off for the rest of the video.
TRACKER_RECOVERY_LOST_FRAMES = 30


# --------------------------------------------------------------------------
# ROI + rotation acquisition
# --------------------------------------------------------------------------

def load_roi_cache():
    if ROI_CACHE_PATH.exists():
        return json.loads(ROI_CACHE_PATH.read_text())
    return {}


def save_roi_cache(cache):
    ROI_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def grab_preview_frame(video_path, seconds=2.0):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    target_frame = int(fps * seconds)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read any frame from {video_path}")
    return frame


def grab_preview_frames(video_path, n=TEMPLATE_VOTE_FRAMES):
    """Spread n sample frames across the first third of the video (avoids
    likely intro/hand-adjusting-the-cup moments at the very start or end)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    positions = [int(total * f) for f in np.linspace(0.05, 0.35, n)]
    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    if not frames:
        frames = [grab_preview_frame(video_path)]
    return frames


def load_template():
    if not TEMPLATE_PATH.exists():
        raise RuntimeError(
            f"{TEMPLATE_PATH} not found - this is the one-time reference crop "
            f"of the LCD panel that every video is matched against."
        )
    template = cv2.imread(str(TEMPLATE_PATH))
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def detect_panel_in_frame(frame, template_gray):
    """
    Multi-scale template match: resize the reference template through a
    range of scale factors (camera zoom/distance differs per video) and
    keep the best-scoring position and scale.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    H, W = gray.shape
    th, tw = template_gray.shape

    best = None
    scale = TEMPLATE_SCALE_RANGE[0]
    while scale <= TEMPLATE_SCALE_RANGE[1] + 1e-9:
        rw, rh = int(tw * scale), int(th * scale)
        if 20 <= rw <= W and 20 <= rh <= H:
            resized = cv2.resize(template_gray, (rw, rh))
            res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)
            if best is None or maxval > best[0]:
                best = (maxval, maxloc[0], maxloc[1], rw, rh)
        scale += TEMPLATE_SCALE_STEP

    score, x, y, w, h = best
    return score, (x, y, w, h)


def detect_panel(video_path, template_gray):
    """
    Try several frames of the video and keep the single BEST-scoring match,
    not a coordinate-wise blend across frames. A hand or reflection
    occluding the display in even one or two of the sampled frames pulls a
    positional median away from where any real frame actually matched well;
    picking the most confident individual detection avoids that failure
    mode instead of averaging a good detection with a bad one.
    """
    frames = grab_preview_frames(video_path)
    results = [detect_panel_in_frame(f, template_gray) for f in frames]
    best = max(results, key=lambda r: r[0])
    return best


def estimate_rotation_deg(frame, roi):
    """
    Auto-measure the panel's tilt in-frame from near-horizontal edges
    (bezel border, digit tops) inside the ROI, via Hough line detection.
    Falls back to 0 degrees if no confident line cluster is found.
    """
    x, y, w, h = roi
    crop = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 360, threshold=40,
        minLineLength=max(20, int(w * 0.3)), maxLineGap=10
    )
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) < 20:  # keep near-horizontal lines only
            angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_image(image, rotation_deg):
    """
    Rotate around the center onto an EXPANDED canvas sized to fit the whole
    rotated content. Rotating onto the original (w, h) canvas clips corners
    (bottom of the digits, in practice) whenever they swing outside the
    original box during rotation.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


BORDER_HSV_LOWER = np.array([80, 100, 0])
BORDER_HSV_UPPER = np.array([140, 255, 130])


def find_ring_component(img):
    """
    Find the dark navy bezel-border ring around the LCD glass: high
    saturation, low value, in the same blue-ish hue range as the glass
    itself (the glass is the same hue but low saturation/high value - this
    is what actually distinguishes them). A raw color mask also picks up
    dark digit segments and icons, so pick out specifically the component
    whose bounding box covers most of the image but is mostly hollow (a
    thin frame, not a filled blob) - that combination is unique to the ring.
    Returns (ring_mask, stats_row) or None if nothing matches.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BORDER_HSV_LOWER, BORDER_HSV_UPPER)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    H, W = mask.shape
    best, best_frac = None, -1
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        bbox_area = w * h
        fill = area / bbox_area if bbox_area else 0
        frac = bbox_area / (H * W)
        if frac > 0.4 and fill < 0.35 and frac > best_frac:
            best, best_frac = i, frac
    if best is None:
        return None
    return np.uint8(labels == best) * 255, stats[best]


def longest_edge_angle(rect):
    """The panel's true tilt is the angle of the ring's longest edge (its
    width, not its ~3x-shorter height) relative to horizontal."""
    box = cv2.boxPoints(rect)
    best_len, best_ang = -1, 0.0
    for i in range(4):
        p1, p2 = box[i], box[(i + 1) % 4]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length > best_len:
            best_len = length
            ang = np.degrees(np.arctan2(dy, dx))
            if ang > 90:
                ang -= 180
            if ang < -90:
                ang += 180
            best_ang = ang
    return best_ang


def detect_rotation_via_ring(panel):
    """
    Measure the panel's tilt from the bezel-border ring's longest edge.
    More precise than the Hough-line estimate since it fits the ring's
    actual geometry rather than voting over many candidate edge segments.

    Only used for rotation. An earlier version also derived per-video crop
    boundaries from the ring's interior (rotate level, re-find the ring,
    inset past its thickness) - in principle more accurate than a fixed
    crop fraction, but it wasn't robust in practice: which preview frame
    the ring got detected on materially changed the resulting crop bounds
    (occasional detections included an icon or the border edge itself,
    sweeping in false digits), and a single frame's early calibration
    doesn't necessarily hold as the tracked box drifts pixel by pixel
    across the rest of the video. Rotation alone doesn't have that
    failure mode - it's a single angle, medianed across frames, applied
    uniformly - so only that part is kept; cropping stays on the fixed,
    per-installation-tuned CROP_FRACTIONS.

    Returns rotation_deg or None if no ring can be found (e.g. unusual
    lighting) - caller should fall back to the Hough-line estimate.
    """
    found = find_ring_component(panel)
    if found is None:
        return None
    ring, _ = found
    rect = cv2.minAreaRect(cv2.findNonZero(ring))
    return float(longest_edge_angle(rect))


def video_key(video_path):
    """
    Unique, readable identifier for a video: parent-folder + filename stem,
    e.g. "lvl2_10mL-1". Filename alone collides whenever the same recording
    name shows up under different source folders (this dataset has
    Nestle_10-20-60mL_syringe_deneyler/lvl1..lvl4, each with its own
    10mL-1.mp4/20mL-1.mp4/etc) - using just the name as the cache key or
    output folder name would silently reuse another folder's ROI/rotation,
    or overwrite its results.
    """
    video_path = Path(video_path)
    return f"{video_path.resolve().parent.name}_{video_path.stem}"


def calibrate_rotation_and_crop(video_path, roi):
    """
    Rotation: median of the ring-based estimate across several preview
    frames (falls back to the Hough-line estimate on frames/videos where
    the ring can't be found; if none succeed, falls back entirely to a
    single Hough estimate on one frame). Crop: always the fixed, tuned
    CROP_FRACTIONS - see detect_rotation_via_ring for why that part isn't
    also derived from the ring.
    """
    x, y, w, h = roi
    rotations = []
    for frame in grab_preview_frames(video_path):
        panel = frame[y:y + h, x:x + w]
        rotation_deg = detect_rotation_via_ring(panel)
        if rotation_deg is None:
            rotation_deg = estimate_rotation_deg(frame, roi)
        rotations.append(rotation_deg)

    rotation_deg = float(np.median(rotations)) if rotations else estimate_rotation_deg(
        grab_preview_frame(video_path), roi
    )
    return rotation_deg, CROP_FRACTIONS, True


def get_video_calibration(video_path, cache, template_gray, force_rescan=False):
    """
    Returns (roi, rotation_deg, crop_fractions, score) for a video - fully
    automatic via template matching + bezel-ring detection, cached per
    video so repeat runs are instant.
    """
    key = video_key(video_path)
    if not force_rescan and key in cache:
        entry = cache[key]
        return (tuple(entry["roi"]), entry["rotation_deg"],
                tuple(entry["crop_fractions"]), entry.get("score", 1.0))

    score, roi = detect_panel(video_path, template_gray)
    rotation_deg, crop_fractions, used_ring = calibrate_rotation_and_crop(video_path, roi)
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


# --------------------------------------------------------------------------
# Digit extraction / decoding (same math as v4)
# --------------------------------------------------------------------------

def preprocess_digits_area(image, rotation_deg, crop_fractions=CROP_FRACTIONS):
    rotated = rotate_image(image, rotation_deg)
    h, w = rotated.shape[:2]

    top, bottom, left, right = crop_fractions
    digits_only = rotated[int(h * top):int(h * bottom), int(w * left):int(w * right)]

    gray = cv2.cvtColor(digits_only, cv2.COLOR_BGR2GRAY) if len(digits_only.shape) == 3 else digits_only.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive (local) threshold instead of one global Otsu cutoff: some
    # videos have a brightness gradient across the digit crop (one side of
    # the panel lit less evenly than the other), which makes a single
    # global threshold correctly binarize one side while fragmenting the
    # strokes on the dimmer side. A local threshold tracks that gradient.
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 75, 10)

    scale_factor = 4
    upscaled = cv2.resize(thresh, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    H, W = upscaled.shape
    extra_w = int(FONT_SHEAR * H) + 10
    M_shear = np.float32([[1, FONT_SHEAR, 0], [0, 1, 0]])
    deskewed = cv2.warpAffine(upscaled, M_shear, (W + extra_w, H), flags=cv2.INTER_NEAREST, borderValue=255)

    return deskewed


def classify_digit(digit_img):
    h, w = digit_img.shape[:2]
    if h == 0 or w == 0:
        return None

    if w < h * 0.30:
        return 1

    def filled(fx0, fy0, fx1, fy1):
        x0, y0 = int(fx0 * w), int(fy0 * h)
        x1, y1 = int(fx1 * w), int(fy1 * h)
        region = digit_img[y0:y1, x0:x1]
        if region.size == 0:
            return 0
        return 1 if (np.count_nonzero(region) / region.size) > 0.35 else 0

    a = filled(0.15, 0.02, 0.85, 0.11)
    b = filled(0.65, 0.10, 1.00, 0.46)
    c = filled(0.65, 0.54, 1.00, 0.90)
    d = filled(0.15, 0.89, 0.85, 0.98)
    e = filled(0.00, 0.54, 0.35, 0.90)
    f = filled(0.00, 0.10, 0.35, 0.46)
    g = filled(0.15, 0.44, 0.85, 0.56)

    pattern = (a, b, c, d, e, f, g)

    digit = DIGIT_LOOKUP.get(pattern)
    if digit is not None:
        return digit

    best_digit, best_dist = None, 99
    for ref_pattern, ref_digit in DIGIT_LOOKUP.items():
        dist = sum(a1 != a2 for a1, a2 in zip(pattern, ref_pattern))
        if dist < best_dist:
            best_dist, best_digit = dist, ref_digit
    return best_digit if best_dist <= 1 else None


def read_weight_from_frame(frame, roi_coords, rotation_deg, crop_fractions, debug_folder=None, frame_count=None):
    x, y, w, h = roi_coords
    roi = frame[y:y + h, x:x + w]

    processed = preprocess_digits_area(roi, rotation_deg, crop_fractions)

    if debug_folder is not None:
        cv2.imwrite(f"{debug_folder}/frame_{frame_count:06d}_roi.png", roi)
        cv2.imwrite(f"{debug_folder}/frame_{frame_count:06d}_proc.png", processed)

    digit_mask = cv2.bitwise_not(processed)
    # Wide but SHORT kernel: needs ~25px horizontally to rejoin a digit's own
    # segments (e.g. the diagonal joints of a 7-segment digit), but there's
    # no legitimate reason to bridge that far vertically. A square 25x25
    # kernel could bridge a digit up into the residual bezel-border sliver
    # at the top of the crop when they happened to be close enough - that
    # merged blob then fails the max-width filter below and gets discarded
    # whole, silently dropping the digit it absorbed (seen on 20mL-1: digit
    # '3' vanishing from a '1379' reading, producing '179' instead).
    kernel = np.ones((9, 25), np.uint8)
    closed = cv2.morphologyEx(digit_mask, cv2.MORPH_CLOSE, kernel)

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


# --------------------------------------------------------------------------
# Per-video pipeline
# --------------------------------------------------------------------------

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
            tracker = cv2.TrackerMIL_create()
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
                    tracker = cv2.TrackerMIL_create()
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


# --------------------------------------------------------------------------
# Batch driver
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", help="Video files to process")
    parser.add_argument("--rescan", action="store_true", help="Ignore cached panel detection and re-run template matching for every video")
    parser.add_argument("--no-debug", action="store_true", help="Skip writing ocr_debug crop images (faster, less disk)")
    parser.add_argument("--outdir", default=str(Path(__file__).parent / "batch_output"),
                         help="Root folder for per-video outputs (default: batch_output/ next to this script)")
    parser.add_argument("--fps-target", type=int, default=10, help="Sampled frames per second")
    args = parser.parse_args()

    template_gray = load_template()
    cache = load_roi_cache()
    outdir = Path(args.outdir)
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

    print("\n=== Batch summary ===")
    for name, total, decoded, score in summary:
        pct = f"{100 * decoded / total:.0f}%" if total else "n/a"
        flag = "" if score >= TEMPLATE_MIN_SCORE else "  <-- low match confidence"
        print(f"{name:20s} sampled={total:4d} decoded={decoded:4d} ({pct})  match_score={score:.2f}{flag}")


if __name__ == "__main__":
    main()
