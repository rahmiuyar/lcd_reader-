#!/usr/bin/env python3
"""
Runs the v4 (rotation+shear corrected 7-segment decoder) pipeline on
3.mp4 - same experiment repeat, different camera framing than 1.mp4, so
the initial ROI was re-measured for this video.
"""

from extract_lcd_weights_v4 import process_video

if __name__ == "__main__":
    results = process_video(
        video_path="3.mp4",
        output_excel="weight_readings_v4_video3.xlsx",
        annotated_folder="annotated_frames_v4_video3",
        debug_folder="ocr_debug_v4_video3",
        initial_roi=(355, 1390, 420, 150),
    )
    print(f"\nProcessed {len(results)} frames for 3.mp4.")
