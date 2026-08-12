---
name: feedback-always-add-charts
description: "User wants a weight-vs-time chart added to every *_weights.xlsx output going forward, not just when explicitly asked once."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a5761d75-d215-4b22-ad92-2af68d69d458
  modified: 2026-08-07T13:58:39.682Z
---

Always add a time-vs-weight scatter chart to every `*_weights.xlsx` file produced by any LCD-reader pipeline run, without waiting to be asked each time.

**Why:** After a batch run produced Excel files without charts, the user asked once ("grafikleri ekler misin") and then said explicitly "bundan sonra hep grafik ekle" (add charts from now on, always) - a standing instruction, not a one-off request. See [[project_lcd_pipeline]] for pipeline context.

**How to apply:** Whenever a script writes `*_weights.xlsx` (production `extract_lcd_weights_batch*.py`, or any experimental variant like the `deep_learning_tracker/extract_lcd_weights_nano_tracker*.py` line, including new version numbers not yet seen, or ad-hoc test-output folders like `iddsi2_v7test`), immediately follow up by adding the scatter chart - reuse/adapt the existing chart logic in `nestle_syringie/add_charts.py` (points-only ScatterChart, timestamp_sec on x, weight on y, replaces any existing chart on re-run) rather than waiting for the user to ask again. `nestle_syringie/deep_learning_tracker/add_charts_nano.py` takes the target output folder as a CLI arg (`add_charts_nano.py <folder_name>`) so it can be pointed at whichever output folder was just produced, versioned or not. Slipped once (2026-08-07): ran a test batch into a new folder and reported file paths to the user without charting first - do this as part of the SAME turn that produces the xlsx, before telling the user where to look.
