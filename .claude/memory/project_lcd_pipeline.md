---
name: project-lcd-pipeline
description: "What the lcd_reader project does, where the working pipeline script lives, and the current dataset layout"
metadata: 
  node_type: memory
  type: project
  originSessionId: a5761d75-d215-4b22-ad92-2af68d69d458
  modified: 2026-08-07T15:31:32.200Z
---

The project reads the weight value off a digital scale's LCD display (WL-3002L, 4-digit 7-segment, max 3000g, resolution 0.01g) from video recordings of syringe-dispensing experiments, and outputs frame-by-frame `weight` readings to Excel.

**The working pipeline is `nestle_syringie/extract_lcd_weights_batch.py`** — fully automatic (no manual ROI clicking), run per video or with a glob:
```
cd nestle_syringie && ../.venv/bin/python extract_lcd_weights_batch.py "<path/to/video.mp4>"
```
Earlier scripts in the project root (`extract_lcd_weights.py`, `_v3.py`, `_v4.py`, `_segment.py`) are superseded/abandoned — they used hardcoded per-video ROI coordinates and were never made generic. Don't build on them.

**Pipeline architecture** (see also [[feedback-lcd-reader-workflow]] for why it's shaped this way):
- Panel location: multi-scale template match against `nestle_syringie/panel_template.png` (a saved reference crop from `1.mp4`), voted across several preview frames, picking the single most-confident match (not a positional average — averaging let one occluded frame drag the box off target).
- Rotation: measured from the dark bezel-border ring's longest edge (HSV-thresholded, connected-component isolated from digit/icon noise by its hollow-ring shape), medianed across preview frames. Falls back to a Hough-line estimate if no ring is found.
- Digit crop: fixed `CROP_FRACTIONS` constant (top, bottom, left, right) applied to the leveled panel — NOT derived per-video from the ring's interior; see [[feedback-lcd-reader-workflow]] for why that was tried and reverted.
- Per-video results cached in `nestle_syringie/roi_cache.json`, keyed by `<parent-folder-name>_<video-stem>` (via `video_key()`) — not by filename alone, because the dataset has multiple folders with identically-named videos.
- Output per video: `nestle_syringie/batch_output/<key>/` containing `<key>_weights.xlsx`, `annotated_frames/`, `ocr_debug/` (debug crops saved by default).

**Companion scripts** (also in `nestle_syringie/`), safe to re-run any time after processing videos:
- `add_charts.py` — adds a weight-vs-time point/scatter chart to every `batch_output/*/*_weights.xlsx`.
- `add_remaining_volume.py` — copies `batch_output/*/*_weights.xlsx` into `nestle_syringie/deney_verileri/`, adds a `remaining_mL` column (`initial_mL - weight_g / density`, density fixed at 1.0 g/mL = water, clipped to `[0, initial_mL]`) plus its own chart. Initial volume is parsed from the output-folder key by regex for a `NNmL` marker, with `NO_MARKER_INITIAL_ML` special-casing the root-level `lcd_reader_1/2/3` recordings (confirmed via md5sum to be the same files as `10mL-1/2/3.mp4`).

**Dataset layout**: recordings live in several places with duplicate/overlapping names —
- Project root: `1.mp4`/`2.mp4`/`3.mp4` (== 10mL trials 1-3), `20mL-1/2/3.mp4`, `60mL-1/2/3.mp4`.
- `Nestle_10-20-60mL_syringe_deneyler/lvl1/`..`lvl4/` — each folder has its own `10mL-1/2/3.mp4`, `20mL-1/2/3.mp4`, `60mL-1/2/3.mp4` (tripod-mounted camera, consistent framing). Referred to below as "the Nestle dataset."
- An older `lvl1-20260803T143309Z-1-001/lvl1/` Google-Drive-export folder also exists; its `20mL-*`/`60mL-*` files are byte-identical duplicates of the root-level ones (confirmed via md5sum) — only its `10mL-1/2/3.mp4` are unprocessed/unique.
- **`kıvampro_10_20_60mL_deneyler/iddsi1/` `iddsi2/` `iddsi3/`** — a second, later dataset (note the `ı` in the folder name). Each subfolder has 9 videos: `iddsiN-10ml-1/2/3.mp4`, `iddsiN-20ml-1/2/3.mp4`, `iddsiN-60ml-1/2/3.mp4`. Phone-recorded (handheld), NOT tripod-mounted like the Nestle set — noticeably more camera tilt/portrait-pillarboxing, occasional real perspective/keystone distortion (uneven digit widths across the same display, confirmed by direct measurement), and heavier glare on some clips. This is why it needed its own, more robust pipeline (see below) rather than reusing `extract_lcd_weights_batch.py` as-is.

**Second pipeline for the kıvampro dataset — `nestle_syringie/extract_lcd_weights_nano_tracker_v7.py`** (moved here from `deep_learning_tracker/` once it graduated from experiment to daily-driver). Same overall shape as the original pipeline (auto panel detection → rotation → digit crop → 7-segment decode → Excel) but with three swapped-out pieces, each added because the original pipeline's assumptions broke on this newer, phone-recorded dataset:
- **Panel position**: bezel-border ring detection (HSV mask, connected-component) instead of template matching as the primary method — tries a RANGE of morphological closing kernel sizes (7-31px) and keeps the largest valid candidate per preview frame, then takes the majority-position cluster across 15 preview frames (not a plain median) — needed because glare intermittently breaks the border's color continuity into disconnected pieces, and a single fixed kernel size that fixes one video's gap can wrongly merge a different video's display with its button row below. Falls back to template matching only if ring detection finds nothing at all.
- **Tracker**: `cv2.TrackerNano` (lightweight Siamese-network tracker, ONNX models in `nestle_syringie/deep_learning_tracker/models/`) instead of `cv2.TrackerMIL` — MIL drifted/oscillated over long videos; a heavier ONNX tracker (`TrackerVit`) was also tried and rejected because it catastrophically lost tracking (box exploded to full-frame) on 2 of 8 test videos, a failure mode NanoTrack didn't show on the same set.
- **Digit segmentation/classification**: raised the box height floor (`H*0.25` → `H*0.55`) to reject stray glare/compression-artifact blobs that were being counted as unclassifiable extra "digits"; added a per-box width/height-ratio split (catches two digits merged into one blob by the closing kernel, splits at the locally sparsest column); narrowed `classify_digit`'s middle-segment ('g') sample band (`x=0.15-0.85` → `x=0.25-0.75`) because on digit boxes that are narrower/taller than the reference proportions (the perspective-distortion effect above), the italic-shear-tapered tips of the left/right strokes were poking into the wide sample band and misreading a clean "0" as "8".
- Panel/rotation results cache in the SAME `nestle_syringie/roi_cache_ring_test.json` used by the ring-detection experiments (not the production `roi_cache.json`) — video keys collide-safely across both datasets via the same `video_key()` scheme.
- Output per video: `<--outdir>/<key>/` (pass `--outdir` explicitly; default is `nestle_syringie/test_nano_full_v7/`) containing `<key>_weights.xlsx`, `annotated_frames/`, `ocr_debug/`.
- For the kıvampro dataset specifically, outputs were written to `kivampro_iddsi/iddsi1/`, `kivampro_iddsi/iddsi2/`, `kivampro_iddsi/iddsi3/` (mirroring the source subfolder names), with a flat `kivampro_iddsi/results/` holding just a copy of every `*_weights.xlsx` for easy handoff.
- Chart script for this pipeline's outputs: `nestle_syringie/deep_learning_tracker/add_charts_nano.py <folder_name>` (same chart style as `add_charts.py`, but takes the target output folder as a CLI arg since there are several versioned ones: `test_nano_full`, `test_nano_full_v2` .. `_v7`, or an arbitrary `--outdir`).
- **Excel post-processing companion — `nestle_syringie/excel_postprocess.py`** (also moved here from `deep_learning_tracker/`): adds a spike-exclusion filter, dispense-start-time correction, and a "Hacim" (remaining syringe volume, mL, water density) sheet+chart to an already-produced `*_weights.xlsx`, entirely as live Excel formulas (not baked-in Python values) so the thresholds stay visible/adjustable in the spreadsheet itself. See [[feedback-lcd-reader-workflow]] for the design history (why it isn't a carry-forward filter, why excluded cells use `NA()` not `""`, why the median window and reset threshold ended up at their current values).

Decode rates on the kıvampro dataset run noticeably lower than the Nestle dataset's best cases (roughly 60-95% depending on the clip, vs 95-100%) even after all of the above — the remaining gap is real per-frame OCR ambiguity from this dataset's harder recording conditions (perspective distortion, glare), not a known/fixable bug. `excel_postprocess.py`'s spike filter cleans up most of the resulting visible noise in the chart but can't recover data the OCR never read correctly in the first place.
