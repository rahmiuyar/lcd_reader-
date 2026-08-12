#!/usr/bin/env python3
"""
Post-process a *_weights.xlsx produced by the pipeline: a data-derived
dispense-start-corrected time column, and a second sheet converting weight
to remaining syringe volume (mL, assuming water density). Everything is
done as live Excel formulas (not baked-in Python values) referencing a
small, labeled parameter block, so the thresholds are visible and
adjustable directly in the spreadsheet.

History of the outlier filter (why there isn't one anymore):
  v1 - carry-forward filter (each row compared to the previous ACCEPTED
       value). One bad reading slipping through as "accepted" permanently
       dragged down every later correct-but-lower reading too.
  v2/v3 - local-median-window filter instead (window widened 7 -> 12 to
       stop a misread burst from locally outvoting the true value).
  v4 - added a leading-segment/reset detector. Reverted: only validated on
       2 videos, wrong on ~10 of 26 once applied dataset-wide (the real
       drop was more often an end-of-run tare AFTER the real data, not
       junk before it), and hard-errored 2 files to 0 kept rows.
  v5 - back to plain local-median rejection (tolerance 0.15g). Still wrong
       on most files: real fast transitions in this dataset (phone-
       recorded, lower frame rate than the tolerance assumed) routinely
       deviate from their own local median by more than 0.15g, so the
       filter was throwing out genuine transition data across the set,
       not just misreads - e.g. iddsi1-60ml-1 down to 8% kept.
  v6 - filter removed entirely (this version). No local_median_g,
       excluded, or weight_kept_g columns - every attempt at an
       automatic outlier rule has ended up wrong on a large chunk of this
       dataset in a new way each time. Sheet1's raw `weight` column
       (from the pipeline, untouched) is now what both the start
       detection and the Hacim sheet read from directly.

Sheet1 (original data) gains:
  E: is_started         - 1 once the raw weight first reaches the start
     threshold.
  F: time_corrected_s   - timestamp_sec minus the detected dispense-start
     time (first timestamp where is_started flips to 1, or 0 if the
     threshold is never reached) - negative before the detected start, 0
     at start.
  Parameter block: start threshold, syringe volume, density, plus a
     computed start-time cell - all editable/inspectable directly in the
     spreadsheet.

New sheet "Hacim" (Volume): time_corrected_s vs remaining volume in the
syringe (mL) = syringe_volume - weight / density, straight from the raw
weight column - #N/A only where weight itself is blank. Points-only
scatter chart (same style as add_charts.py's weight chart).

Usage (from inside nestle_syringie/):
    ../.venv/bin/python excel_postprocess.py <path-to-weights.xlsx> [...]
"""
import sys
sys.path.append('/home/bee/projeler/lcd_reader/.venv/lib/python3.14/site-packages')

from pathlib import Path
import openpyxl
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.marker import Marker
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.styles import Font

MARKER_COLOR = "D62728"

START_THRESHOLD_G = 0.05
DEFAULT_DENSITY = 1.0


def infer_nominal_volume_ml(path):
    """Nominal syringe size from the filename convention (..._10ml-N /
    _20mL-N / _60mL-N ...) - falls back to 10 if it can't be parsed."""
    name = Path(path).stem.lower()
    for size in (60, 20, 10):
        if f"{size}ml" in name:
            return float(size)
    return 10.0


def add_postprocess(path):
    path = Path(path)
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    n_rows = ws.max_row  # includes header row 1
    first_data_row = 2

    nominal_volume = infer_nominal_volume_ml(path)

    header_font = Font(bold=True)

    # clear stale E:K columns and N7:O14 parameter cells left over from
    # earlier versions of this script (local-median filter, reset
    # detection) - the current layout only uses E:F and N1:O8, and
    # leftover formulas beyond that would reference parameter cells this
    # version no longer writes.
    for r in range(1, n_rows + 1):
        for col in range(5, 12):  # E..K
            ws.cell(r, col).value = None
    for row_pair in (7, 9, 10, 11, 12, 13, 14):
        ws.cell(row_pair, 14).value = None  # N
        ws.cell(row_pair, 15).value = None  # O

    ws["E1"] = "is_started"
    ws["F1"] = "time_corrected_s"
    for cell in ("E1", "F1"):
        ws[cell].font = header_font

    ws["N1"] = "Parametreler"
    ws["N1"].font = header_font
    ws["N4"] = "Baslangic esigi (g)"
    ws["O4"] = START_THRESHOLD_G
    ws["N5"] = "Siringa hacmi (mL)"
    ws["O5"] = nominal_volume
    ws["N6"] = "Yogunluk (g/mL)"
    ws["O6"] = DEFAULT_DENSITY
    ws["N8"] = "Baslangic zamani (s)"
    ws["O8"] = (
        f"=IFERROR(INDEX(B{first_data_row}:B{n_rows},"
        f"MATCH(1,E{first_data_row}:E{n_rows},0)),0)"
    )

    for r in range(first_data_row, n_rows + 1):
        ws[f"E{r}"] = f"=IF(AND(ISNUMBER(D{r}),D{r}>=$O$4),1,0)"
        ws[f"F{r}"] = f"=B{r}-$O$8"

    # Raw weight vs corrected time - separate from the existing raw-weight-
    # vs-raw-time chart so both are visible side by side for comparison.
    ws._charts = ws._charts[:1]  # keep only the original raw-weight chart
    chart = ScatterChart()
    chart.title = f"{path.stem} - weight vs corrected time"
    chart.x_axis.title = "time since dispense start (s)"
    chart.y_axis.title = "weight (g)"
    chart.x_axis.axPos = "b"
    chart.y_axis.axPos = "l"
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.width = 24
    chart.height = 10
    chart.style = 2
    x_ref = Reference(ws, min_col=6, min_row=first_data_row, max_row=n_rows)  # F: time_corrected_s
    y_ref = Reference(ws, min_col=4, min_row=1, max_row=n_rows)  # D: weight
    series = Series(y_ref, x_ref, title_from_data=True)
    series.marker = Marker(symbol="circle", size=5)
    series.marker.graphicalProperties = GraphicalProperties(solidFill=MARKER_COLOR)
    series.marker.graphicalProperties.line.solidFill = MARKER_COLOR
    series.graphicalProperties.line.noFill = True
    chart.series.append(series)
    ws.add_chart(chart, "Q2")

    # --- New sheet: remaining syringe volume vs corrected time ---
    if "Hacim" in wb.sheetnames:
        del wb["Hacim"]
    vs = wb.create_sheet("Hacim")
    vs["A1"] = "time_corrected_s"
    vs["B1"] = "remaining_volume_mL"
    vs["A1"].font = header_font
    vs["B1"].font = header_font
    for r in range(first_data_row, n_rows + 1):
        vs[f"A{r}"] = f"=Sheet1!F{r}"
        vs[f"B{r}"] = f"=IF(NOT(ISNUMBER(Sheet1!D{r})),NA(),Sheet1!$O$5-(Sheet1!D{r}/Sheet1!$O$6))"

    vchart = ScatterChart()
    vchart.title = f"{path.stem} - remaining syringe volume vs time"
    vchart.x_axis.title = "time since dispense start (s)"
    vchart.y_axis.title = "remaining volume (mL)"
    vchart.x_axis.axPos = "b"
    vchart.y_axis.axPos = "l"
    vchart.x_axis.delete = False
    vchart.y_axis.delete = False
    vchart.width = 24
    vchart.height = 10
    vchart.style = 2
    vx_ref = Reference(vs, min_col=1, min_row=first_data_row, max_row=n_rows)
    vy_ref = Reference(vs, min_col=2, min_row=1, max_row=n_rows)
    vseries = Series(vy_ref, vx_ref, title_from_data=True)
    vseries.marker = Marker(symbol="circle", size=5)
    vseries.marker.graphicalProperties = GraphicalProperties(solidFill="2CA02C")
    vseries.marker.graphicalProperties.line.solidFill = "2CA02C"
    vseries.graphicalProperties.line.noFill = True
    vchart.series.append(vseries)
    vs.add_chart(vchart, "D2")

    wb.save(path)
    print(f"updated: {path}  (nominal syringe volume assumed {nominal_volume:.0f} mL from filename)")


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: excel_postprocess.py <weights.xlsx> [...]")
        return
    for p in paths:
        add_postprocess(p)


if __name__ == "__main__":
    main()
