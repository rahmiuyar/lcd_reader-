# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Reads the weight value off a digital scale's LCD display (WL-3002L, 4-digit
7-segment, max 3000g, resolution 0.01g) from video recordings of
syringe-dispensing experiments, and outputs frame-by-frame `weight`
readings to Excel (`*_weights.xlsx`).

There is no build/lint/test suite — this is a research/data-processing
pipeline run interactively against video files, verified by visual
inspection of debug crops and rendered charts (see "Verifying changes"
below).

## Environment

Python virtualenv at `.venv/`. Always invoke scripts with the venv
interpreter, not system python:
```
.venv/bin/python <script.py> ...
```
Key packages: `opencv-python`, `numpy`, `openpyxl`, `pillow`, `pytesseract`.
No `requirements.txt`/`pyproject.toml` exists — the venv is the source of truth.

## Two parallel pipelines

There are two independent extraction pipelines, each tuned for a different
dataset. **Do not conflate them or "fix" one by porting logic from the
other without checking it generalizes** — see the file-history comments in
each script for why specific constants/thresholds were chosen.

### 1. `nestle_syringie/extract_lcd_weights_batch.py` — Nestle dataset
For tripod-mounted recordings (consistent framing, minimal tilt).
```
cd nestle_syringie && ../.venv/bin/python extract_lcd_weights_batch.py "<path/to/video.mp4>" [--rescan] [--no-debug] [--outdir DIR] [--fps-target N]
```
- Panel location: multi-scale template match against `panel_template.png`,
  voted across several preview frames (most-confident single match, not an
  average — averaging lets one occluded frame drag the box off target).
- Rotation: measured from the dark bezel-border ring's longest edge
  (HSV-thresholded, connected-component isolated), medianed across preview
  frames; falls back to Hough-line estimate if no ring is found.
- Digit crop: fixed `CROP_FRACTIONS` constant applied to the leveled panel
  — deliberately NOT derived per-video from the ring interior (tried and
  reverted; per-video geometric crop was fragile across a full video even
  though it looked fine on single frames). If crop quality regresses on a
  new dataset, recalibrate `CROP_FRACTIONS`, don't re-attempt fully
  automatic per-video crop-bound derivation.
- Per-video results cached in `roi_cache.json`, keyed by
  `<parent-folder-name>_<video-stem>` (`video_key()`) — not filename alone,
  because the dataset has multiple folders with identically-named videos.
- Output: `nestle_syringie/batch_output/<key>/` containing
  `<key>_weights.xlsx`, `annotated_frames/`, `ocr_debug/` (written by
  default — pass `--no-debug` only for quick internal iteration, not for
  runs meant for review).
- Earlier scripts in the project root (`extract_lcd_weights.py`, `_v3.py`,
  `_v4.py`, `_segment.py`) and `nestle_syringie/extract_lcd_weights_batch_v1.1.py`/
  `_v1.2.py`/`_latest.py` are superseded/abandoned (hardcoded per-video ROI
  coordinates, never made generic) — don't build on them.

### 2. `nestle_syringie/extract_lcd_weights_nano_tracker_v7.py` — kıvampro dataset
For phone-recorded (handheld) videos: more tilt/portrait-pillarboxing,
real perspective/keystone distortion, heavier glare. Same overall shape as
pipeline 1 (auto panel detection → rotation → digit crop → 7-segment
decode → Excel) but with three pieces swapped out because pipeline 1's
assumptions broke on this dataset:
- **Panel position**: bezel-border ring detection (HSV mask,
  connected-component) as primary method instead of template matching —
  tries a range of morphological closing kernel sizes (7-31px), keeps the
  largest valid candidate per preview frame, takes the majority-position
  cluster across 15 preview frames. Falls back to template matching only
  if ring detection finds nothing.
- **Tracker**: `cv2.TrackerNano` (ONNX models in
  `nestle_syringie/deep_learning_tracker/models/`) instead of
  `cv2.TrackerMIL` (MIL drifted over long videos). `TrackerVit` was also
  tried and rejected — it catastrophically lost tracking on 2/8 test videos.
- **Digit segmentation/classification**: raised box-height floor
  (`H*0.25` → `H*0.55`) to reject glare/artifact blobs; added a
  width/height-ratio split for merged-digit blobs; narrowed
  `classify_digit`'s middle-segment ('g') sample band (`x=0.15-0.85` →
  `x=0.25-0.75`) to stop italic-shear stroke tips on narrower/taller digit
  boxes from misreading "0" as "8".
```
cd nestle_syringie && ../.venv/bin/python extract_lcd_weights_nano_tracker_v7.py "<path/to/video.mp4>" [--rescan] [--no-debug] [--outdir DIR] [--fps-target N]
```
- Panel/rotation cache: `nestle_syringie/roi_cache_ring_test.json`
  (separate from pipeline 1's `roi_cache.json`; same `video_key()` scheme).
- Default output: `nestle_syringie/test_nano_full_v7/` (override with
  `--outdir`); contains `<key>_weights.xlsx`, `annotated_frames/`, `ocr_debug/`.
- Older versions (`extract_lcd_weights_nano_tracker.py` .. `_v6.py`,
  `deep_learning_tracker/test_nano_full` .. `_v5`, `test_vit_tracker`) are
  experiment history, not maintained.
- Decode rates on this dataset run lower than pipeline 1's best cases
  (~60-95% vs ~95-100%) even after all the fixes above — the remaining gap
  is genuine per-frame OCR ambiguity from harder recording conditions, not
  a known/fixable bug.

## Companion post-processing scripts

Safe to re-run any time after a pipeline produces `*_weights.xlsx`:

- **`nestle_syringie/add_charts.py`** — adds a weight-vs-time point/scatter
  chart to every `batch_output/*/*_weights.xlsx` (pipeline 1 output).
- **`nestle_syringie/deep_learning_tracker/add_charts_nano.py <folder_name>`**
  — same chart, for pipeline 2 output; takes the target output folder as a
  CLI arg since there are several versioned ones.
- **Always add a chart to any new `*_weights.xlsx`** as part of the same
  turn that produces it, before reporting file paths — this is a standing
  requirement, not something to wait to be asked for again.
- **`nestle_syringie/excel_postprocess.py <weights.xlsx> [...]`** — adds a
  spike-exclusion filter, dispense-start-time correction, and a "Hacim"
  (remaining syringe volume, mL, water density) sheet+chart, entirely as
  live Excel formulas referencing a labeled, editable parameter block (not
  baked-in Python values). Excluded rows are left as `#N/A` (never deleted
  or substituted) — Excel/LibreOffice scatter charts plot `#N/A` as a gap
  but plot `""` as `y=0`, which is why `NA()` is used instead of blanking
  the cell. See the module docstring for the full history of why the
  filter is shaped the way it is (median-window local-outlier detection +
  separate largest-sustained-drop reset detector, thresholds tuned across
  several false-positive/false-negative iterations).
- **`nestle_syringie/add_remaining_volume.py`** — copies
  `batch_output/*/*_weights.xlsx` into `nestle_syringie/deney_verileri/`,
  adds a `remaining_mL` column (density fixed at 1.0 g/mL = water) plus
  its own chart. Initial volume is parsed from the output-folder key via a
  `NNmL` regex marker, with `NO_MARKER_INITIAL_ML` special-casing the
  root-level `lcd_reader_1/2/3` recordings (confirmed identical to
  `10mL-1/2/3.mp4` via md5sum).

**Verifying changes to any of the above**: `openpyxl` only writes formula
*text*, it never evaluates formulas — reading a cell back right after
writing (even with `data_only=True`) proves nothing. To actually verify:
```
soffice --headless -env:UserInstallation=file:///tmp/some_profile_dir \
  --convert-to xlsx:"Calc MS Excel 2007 XML" --outdir <tmp_dir> <file.xlsx>
```
then re-open the converted copy with `openpyxl.load_workbook(..., data_only=True)`
to check specific cells. For anything chart-rendering-specific, go one step
further: narrow `ws.print_area` to just the chart's cells in a throwaway
copy (delete other sheets to avoid a huge PDF), `soffice --convert-to pdf`,
then `pdftoppm` the page to PNG and actually look at it — several
regressions here were only visible in the rendered chart, not in formula
text or recalculated cell values. Use a fresh `-env:UserInstallation=file:///tmp/...`
profile dir per conversion to avoid LibreOffice lock/profile contention
when converting repeatedly.

## Dataset layout

Recordings live in several places with duplicate/overlapping names:
- Project root: `1.mp4`/`2.mp4`/`3.mp4` (== 10mL trials 1-3),
  `20mL-1/2/3.mp4`, `60mL-1/2/3.mp4`.
- `Nestle_10-20-60mL_syringe_deneyler/lvl1/`..`lvl4/` — each has its own
  `10mL-1/2/3.mp4`, `20mL-1/2/3.mp4`, `60mL-1/2/3.mp4` (tripod-mounted,
  consistent framing) — "the Nestle dataset", processed by pipeline 1.
- An older `lvl1-20260803T143309Z-1-001/lvl1/` Google-Drive export exists;
  its `20mL-*`/`60mL-*` are byte-identical duplicates of the root-level
  ones (confirmed via md5sum) — only its `10mL-1/2/3.mp4` are unprocessed/unique.
- **`kıvampro_10_20_60mL_deneyler/iddsi1/` `iddsi2/` `iddsi3/`** (note the
  `ı`) — a second, later dataset, phone-recorded (handheld). Each
  subfolder has 9 videos: `iddsiN-10ml-1/2/3.mp4`, `-20ml-1/2/3.mp4`,
  `-60ml-1/2/3.mp4`. Processed by pipeline 2, outputs collected under
  `kivampro_iddsi/iddsi1/`, `iddsi2/`, `iddsi3/` (mirroring source
  subfolders), with a flat `kivampro_iddsi/results/` holding a copy of
  every `*_weights.xlsx` for handoff. Note: `iddsi2/`'s source folder has
  one stray misfiled video, `iddsi4-20ml-3.mp4` (belongs to a separate,
  incomplete `iddsi4/` dataset that has not been processed — leave it alone
  unless asked).

## Working conventions (see git history / prior sessions for the "why")

- **Fully automatic detection only** — no interactive ROI-selection step,
  even though manual clicking is more accurate. This was deliberately
  rejected once; don't reintroduce it as the primary path.
- **No video-specific fixes.** A fix diagnosed on one clip must generalize
  (wider search range, majority voting, a measured/justified threshold)
  and be validated against at least one other video before being
  considered done — not a per-video lookup table or special-cased
  constant. (A single-entry `if video_key == "...":` override exists once,
  for two confirmed-duplicate videos, as a deliberate, narrow exception —
  not a pattern to repeat casually.)
- **Don't reprocess videos beyond what was asked.** When told to fix
  something for one video, don't proactively rerun the whole existing
  batch "for regression testing" — mention that old results are now stale
  if a shared constant changed, but only rerun on request.
- **Keep debug output on by default** (`ocr_debug/` crops) so results can
  be visually spot-checked — treat any decode-rate improvement with
  suspicion until a few actual frames are visually verified; aggregate
  percentages have hidden real regressions before (a change that silently
  drops digits can still raise the aggregate number).
- **Ask before assuming a physical/scientific parameter** (e.g. liquid
  density) rather than guessing.
- **Output organization**: keep working scripts/outputs in dedicated
  subfolders (`nestle_syringie/`, `deney_verileri/`, etc.), not the
  project root.
