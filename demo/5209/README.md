# 5209 Zonal Climate Anomalies 1880–2023 (GISTEMP): Planetary Bands, Warming World

`python -m jukebox.datasets.zonal_5209` (Quest `short` partition, about 7 min the first time, then cached)

- **Music data:** NASA GISS `ZonAnn.Ts+dSST.csv` (GISTEMP v4), 8 zones, 1880–2023. One year is one beat (0.194 s), locked to the SVS movie.
- **Lead:** the global mean, D4..D6 = −0.6..+1.4 °C. **Bands:** 8 zone voices strummed from north to south each year. Each has its own register (north higher) and covers −2.7..+3.3 °C.
- **Video inversion:** the movie is a strip chart and has no colorbar. The 5 axis lines calibrate x to °C (240 px/°C at 1920 px wide). Each zone marker is read on rows that show only the marker. The year timing comes from fitting piecewise-linear knots to the marker motion, using the video only (knot 0 at frame 0.004, 5.833 frames/year).

| check | r | MAE |
|---|---|---|
| video inversion vs CSV, all zones (1152 values) | 0.9997 | 0.0096 °C |
| per zone (worst: 90S-64S) | ≥ 0.9996 | ≤ 0.019 °C |
| audio round-trip, lead (R² 0.971) | 0.986 | 0.055 °C |
| audio round-trip, 8 zone stems | 0.91–0.99 | 0.09–0.13 °C |
| shuffled control, lead | R² −0.68 | |

Off-by-one check: shifting the years by ±1 raises the MAE from 0.0096 to 0.23 °C.

> **Integrated mix:** `mix.mp3` is now the full arrangement from `jukebox.render.arrange` (data melody + bands, model harmony, recorded voices, about −16 LUFS). See `demo/README.md` for the listening guide and the before/after fidelity table; `arrange_report.json` has the numbers.
