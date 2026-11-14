# Form layer on 3912 (Global Sea Surface Currents and Temperature, ECCO2, 3-2007 → 2-2008)

```
python scripts/run_form.py --id 3912 --start-date 2007-03 --months 12 --vmin 0 --vmax 32   # on a short-partition node
```

Embedding: DINOv2-small (plain-torch forward of `facebook/dinov2-small`), 133 frames, 22 s on 4 CPU cores.

| Section | Video time (s) | approx. date (linear-time assumption, no burned-in dates) |
|---|---|---|
| A | 0.0–25.5 | Mar → mid-Jul 2007 |
| B | 25.5–36.0 | mid-Jul → mid-Sep |
| C | 36.0–43.5 | mid-Sep → late Oct |
| D | 43.5–56.5 | late Oct → early Jan 2008 |
| E | 56.5–66.5 | Jan → Feb 2008 |

No section returns (one year, a monotone seasonal progression). Frame-level regions (k=6, silhouette 0.29,
dwell/hysteresis smoothed) do return: the region of ~16–27 s (Jun–Jul) comes back at 56.5–62.5 s (Jan).

Honesty checks (`form.json → diagnostics`):
- Camera is static (max global shift 0.14 px), so boundaries are **not** camera moves.
- Boundaries sit where the lead (global-mean SST) changes faster than elsewhere: ratio 1.68, circular-shift p = 0.016.
- Only 2 of 4 boundaries (25.5 s, 56.5 s) are reproduced by the same segmentation run on the score's own data
  curves (lead + 6 band values); novelty correlation with that data-only novelty is just 0.08. The biggest data change
  (~14–16 s, several bands step at once) is not a DINOv2 boundary.
- The handcrafted (colour + flow) embedding also matches 2 of 4 (25.5, 43.5 s).
So: the 25.5 s (≈ July) and 56.5 s (≈ January) boundaries are the robust ones; treat the others as embedding-specific.

Files: `form.json` (sections, labels, recurrences, returns, per-frame regions, novelty, diagnostics),
`overview.png`, `score_form.mid` and `form_timeline.json` (M1 score + form cues: cymbal swell/crash into each
boundary, a 4-note section motif; lead/band notes are byte-identical to `demo/3912/timeline.json`).
