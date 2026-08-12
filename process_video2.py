#!/usr/bin/env python3
"""
Runs the v4 (rotation+shear corrected 7-segment decoder) pipeline on
2.mp4 - same experiment repeat, different camera framing than 1.mp4, so
the initial ROI was re-measured for this video.
"""

from extract_lcd_weights_v4 import process_video

if __name__ == "__main__":
    results = process_video(
        video_path="2.mp4",
        output_excel="weight_readings_v4_video2.xlsx",
        annotated_folder="annotated_frames_v4_video2",
        debug_folder="ocr_debug_v4_video2",
        initial_roi=(380, 1370, 410, 160),
    )
    print(f"\nProcessed {len(results)} frames for 2.mp4.")
