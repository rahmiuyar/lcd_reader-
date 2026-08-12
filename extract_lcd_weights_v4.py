#!/usr/bin/env python3
"""
LCD weight reader v4: adds a whole-ROI rotation correction on top of v3.

Two separate distortions were identified:
1. The whole LCD panel appears rotated ~2.61 degrees in frame (camera
   viewing angle vs the physical panel) - this is what made the bezel
   border line and the "g" unit label impossible to crop out cleanly.
2. The digit font itself is italic-slanted (~0.21 dx/dy) - independent of
   the panel rotation, and still needs the shear correction from v3.

Fixing #1 first lets the digits-only crop be much tighter (no border
line, no unit label, no icons - just the 3 digits), then #2 (shear) is
applied same as v3, then the same 7-segment decoder classifies digits.
"""

import sys
sys.path.append('/home/bee/projeler/lcd_reader/.venv/lib/python3.14/site-packages')

import cv2
import numpy as np
import pandas as pd
from pathlib import Path


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

PANEL_ROTATION_DEG = 2.61  # measured via Hough line detection on the bezel border
FONT_SHEAR = 0.21          # measured italic slant: x shifts by ~0.21px per y px


def preprocess_digits_area(image):
    """
    Straighten the panel rotation, crop tightly to just the 3 digits
    (excludes the bezel border, unit label, and icons), then threshold,
    upscale, and deskew the italic font slant.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, PANEL_ROTATION_DEG, 1.0)
    rotated = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))

    # Tight digit-only crop measured on the now-level panel: excludes the
    # "WL-3002L Max.3000g d:0.01g" label, battery/zero icons, "g" unit,
    # and top/bottom bezel border entirely.
    digits_only = rotated[int(h * 0.10):int(h * 0.82), int(w * 0.33):int(w * 0.80)]

    gray = cv2.cvtColor(digits_only, cv2.COLOR_BGR2GRAY) if len(digits_only.shape) == 3 else digits_only.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scale_factor = 4
    upscaled = cv2.resize(thresh, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    H, W = upscaled.shape
    extra_w = int(FONT_SHEAR * H) + 10
    M_shear = np.float32([[1, FONT_SHEAR, 0], [0, 1, 0]])
    deskewed = cv2.warpAffine(upscaled, M_shear, (W + extra_w, H), flags=cv2.INTER_NEAREST, borderValue=255)

    return deskewed


def classify_digit(digit_img):
    """
    digit_img: binary crop of a single (upright) digit, 255 = lit segment,
    0 = background. Returns the recognized digit (0-9) or None.
    """
    h, w = digit_img.shape[:2]
    if h == 0 or w == 0:
        return None

    # A '1' is much narrower than the other digits (only 2 vertical
    # strokes). '7' is also narrower than full-width digits (only a/b/c,
    # no left strokes) but consistently wider than '1' (0.27 vs 0.34).
    if w < h * 0.30:
        return 1

    def filled(fx0, fy0, fx1, fy1):
        x0, y0 = int(fx0 * w), int(fy0 * h)
        x1, y1 = int(fx1 * w), int(fy1 * h)
        region = digit_img[y0:y1, x0:x1]
        if region.size == 0:
            return 0
        return 1 if (np.count_nonzero(region) / region.size) > 0.40 else 0

    a = filled(0.15, 0.02, 0.85, 0.11)   # top
    b = filled(0.65, 0.10, 1.00, 0.46)   # top-right
    c = filled(0.65, 0.54, 1.00, 0.90)   # bottom-right
    d = filled(0.15, 0.89, 0.85, 0.98)   # bottom
    e = filled(0.00, 0.54, 0.35, 0.90)   # bottom-left
    f = filled(0.00, 0.10, 0.35, 0.46)   # top-left
    g = filled(0.15, 0.44, 0.85, 0.56)   # middle

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


def read_weight_from_frame(frame, roi_coords, debug_folder=None, frame_count=None):
    """
    Locate the individual digits inside the ROI via contours, classify each
    one with the 7-segment decoder, and combine them into a weight value.
    Display always shows 2 decimal places (resolution d:0.01g), so the
    decoded digit string is interpreted as e.g. "958" -> 9.58.
    """
    x, y, w, h = roi_coords
    roi = frame[y:y+h, x:x+w]

    processed = preprocess_digits_area(roi)

    if debug_folder is not None:
        cv2.imwrite(f"{debug_folder}/frame_{frame_count:06d}_roi.png", roi)
        cv2.imwrite(f"{debug_folder}/frame_{frame_count:06d}_proc.png", processed)

    digit_mask = cv2.bitwise_not(processed)
    kernel = np.ones((25, 25), np.uint8)
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
            overlap = min(mx+mw, bx+bw) - max(mx, bx)
            if overlap > 0.5 * min(mw, bw):
                nx0, ny0 = min(mx, bx), min(my, by)
                nx1, ny1 = max(mx+mw, bx+bw), max(my+mh, by+bh)
                merged[-1] = (nx0, ny0, nx1-nx0, ny1-ny0)
                continue
        merged.append(b)
    boxes = merged

    if debug_folder is not None:
        vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 0, 255), 3)
        cv2.imwrite(f"{debug_folder}/frame_{frame_count:06d}_boxes.png", vis)

    digits = []
    for (bx, by, bw, bh) in boxes:
        digit_img = digit_mask[by:by+bh, bx:bx+bw]
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


def process_video(video_path, output_excel, annotated_folder, debug_folder=None,
                   initial_roi=(270, 1400, 520, 210)):
    """
    Process video and extract weights from LCD display.
    The camera drifts over time, so a fixed ROI is not enough - a tracker
    follows the LCD region frame by frame starting from an initial box.
    initial_roi (x, y, width, height) must be re-measured per video since
    camera framing/distance differs between recordings.
    """
    Path(annotated_folder).mkdir(parents=True, exist_ok=True)
    if debug_folder is not None:
        Path(debug_folder).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    results = []
    frame_interval = max(1, fps // 10)

    tracker = None
    current_box = initial_roi
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
                lost_count = 0
            else:
                lost_count += 1

        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / fps

            raw_text, weight = read_weight_from_frame(frame, current_box, debug_folder, frame_count)

            results.append({
                'frame_no': frame_count,
                'timestamp_sec': timestamp_sec,
                'raw_digits': raw_text,
                'weight': weight
            })

            x, y, w, h = current_box
            annotated_frame = frame.copy()
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if weight is not None:
                cv2.putText(annotated_frame, f'Weight: {weight:.2f}',
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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


if __name__ == "__main__":
    video_file = "1.mp4"
    output_file = "weight_readings_v4.xlsx"
    annotated_folder = "annotated_frames_v4"
    debug_folder = "ocr_debug_v4"

    results = process_video(video_file, output_file, annotated_folder, debug_folder)

    print(f"\nProcessed {len(results)} frames.")
