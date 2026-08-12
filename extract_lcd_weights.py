#!/usr/bin/env python3

import sys
sys.path.append('/home/bee/projeler/lcd_reader/.venv/lib/python3.14/site-packages')

import cv2
import pytesseract
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os
import time
from pathlib import Path


def preprocess_image_for_ocr(image):
    """
    Preprocess image for OCR - crop out the label text/icons/unit that
    surround the digits, convert to grayscale, threshold, and upscale.
    """
    h, w = image.shape[:2]
    # Keep only the digit area: skip the "WL-3002L Max.3000g d:0.01g" label
    # above, the battery/zero icons on the left, and the "g" unit on the right.
    digits_only = image[int(h * 0.20):int(h * 0.82), int(w * 0.17):int(w * 0.83)]

    gray = cv2.cvtColor(digits_only, cv2.COLOR_BGR2GRAY) if len(digits_only.shape) == 3 else digits_only.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply threshold to get binary image (black and white)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Upscale the image for better OCR recognition
    scale_factor = 4
    upscaled = cv2.resize(thresh, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    return upscaled


def extract_weight_from_frame(frame, roi_coords, debug_folder=None, frame_count=None):
    """
    Extract weight from a frame using OCR.

    The scale's decimal point is too small/faint for reliable OCR, and the
    display always shows exactly 2 decimal digits (resolution d:0.01g), so
    we OCR digits only (no '.') and insert the decimal point programmatically
    instead of trying to detect it.
    """
    # Extract Region of Interest (ROI)
    x, y, w, h = roi_coords
    roi = frame[y:y+h, x:x+w]

    # Preprocess for OCR
    processed_roi = preprocess_image_for_ocr(roi)

    if debug_folder is not None:
        cv2.imwrite(f"{debug_folder}/frame_{frame_count:06d}_roi.png", roi)
        cv2.imwrite(f"{debug_folder}/frame_{frame_count:06d}_proc.png", processed_roi)

    # Try several psm modes and keep whichever finds the most digits - on some
    # frames one mode only catches part of the number, so this ensemble
    # recovers cases a single fixed mode would miss.
    best_digits = ''
    for psm in (8, 7, 13, 6):
        custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(processed_roi, config=custom_config).strip()
        digits = ''.join(c for c in text if c.isdigit())
        if len(digits) > len(best_digits):
            best_digits = digits

    digits = best_digits

    if not digits:
        return None, None

    # Display always shows 2 decimal places (e.g. "958" -> 9.58)
    try:
        weight = int(digits) / 100.0
    except ValueError:
        return digits, None

    # Sanity check against the scale's rated range (Max. 3000g)
    if weight < 0 or weight > 3000:
        return digits, None

    return digits, weight


def process_video(video_path, output_excel, annotated_folder, debug_folder=None):
    """
    Process video and extract weights from LCD display.
    The camera drifts over time, so a fixed ROI is not enough - a tracker
    follows the LCD region frame by frame starting from an initial box.
    """
    # Create annotated folder if it doesn't exist
    Path(annotated_folder).mkdir(parents=True, exist_ok=True)
    if debug_folder is not None:
        Path(debug_folder).mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Initial ROI for the LCD display, measured on frame 0. The tracker
    # updates this box every frame to follow camera drift.
    initial_roi = (270, 1400, 520, 210)  # (x, y, width, height)

    # Initialize result list
    results = []

    # Sample every 6th frame to get at ~10fps (60fps/10fps = 6)
    frame_interval = max(1, fps // 10)

    tracker = None
    current_box = initial_roi
    lost_count = 0

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run the tracker on every frame so it doesn't lose the target
        # between sampled frames.
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

        # Process only every nth frame to match target FPS
        if frame_count % frame_interval == 0:
            # Get timestamp in seconds
            timestamp_sec = frame_count / fps

            # Extract weight from the LCD display at the tracked position
            raw_ocr_text, weight = extract_weight_from_frame(frame, current_box, debug_folder, frame_count)

            # Add to results - keep both the raw OCR digit string and the
            # converted gram value so misreads can be spotted/corrected later.
            result = {
                'frame_no': frame_count,
                'timestamp_sec': timestamp_sec,
                'raw_ocr_text': raw_ocr_text,
                'weight': weight
            }
            results.append(result)

            # Draw bounding box and text on frame for annotation
            x, y, w, h = current_box
            annotated_frame = frame.copy()
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if weight is not None:
                cv2.putText(annotated_frame, f'Weight: {weight:.2f}',
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save annotated frame
            output_path = f"{annotated_folder}/frame_{frame_count:06d}.png"
            cv2.imwrite(output_path, annotated_frame)

            print(f"Frame {frame_count}: raw='{raw_ocr_text}' Weight = {weight}, box={current_box}, lost={lost_count}")

        frame_count += 1

    cap.release()
    
    # Save results to Excel file. raw_ocr_text is written as text so
    # leading-zero digit counts (e.g. "009" vs "9") stay distinguishable
    # when the file is opened in Excel/LibreOffice.
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_excel(output_excel, index=False)
        print(f"Results saved to {output_excel}")
    else:
        print("No valid weight data found.")
        
    return results


if __name__ == "__main__":
    # File paths
    video_file = "1.mp4"
    output_file = "weight_readings.xlsx"
    annotated_folder = "annotated_frames"
    debug_folder = "ocr_debug"

    # Process the video and extract weights
    results = process_video(video_file, output_file, annotated_folder, debug_folder)
    
    print(f"\nProcessed {len(results)} frames.")