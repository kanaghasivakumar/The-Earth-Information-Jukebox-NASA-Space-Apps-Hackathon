# 4950 GEDI Forest Height: canopy melody + biophony control

`python -m jukebox.datasets.gedi_4950`. Movie: `GEDI_global_w_colorbar_1080p60.mp4`, a lit 3D globe that turns once in 51.5 s, sampled at 4 fps. Colorbar: the **0–50 m** variant, the same one burned into the movie (SVS also publishes 0–40 and 0–75 bars for other cuts).

- **Inversion:** `jukebox/datasets/chroma.py` matches RGB chromaticity, which cancels the lighting, and uses lightness only as a weak tie-breaker. On a lit globe, plain CIELAB matching reads shade as shorter forest. Checked on the movie's own burned-in colorbar: r 0.993, MAE 1.2 m (CIELAB on the same flat strip: r 0.997, MAE 0.8 m).
- **Masks:** off-disc, limb (ρ > 0.92), ocean, night side, burned-in colorbar, and non-vegetated pixels (< 3 m). Pixels are area-weighted by 1/cos(view angle). Valid pixels: 47% of land (GEDI swath gaps and the night side fail), 82% of the disc.
- **Melody:** mean canopy height of the visible vegetation, 14–24 m. **Bands:** the same value for the upper, middle and lower thirds of the disc (screen thirds, not true latitude).
- **Controls:** biophony = height × cover in view; geophony = ocean share; brightness = share of trees ≥ 25 m.
- **Audio round-trip:** melody R² 0.982 (r 0.991, MAE 0.29 m); bands R² 0.90–0.94; shuffled control R² −1.32.

> **Integrated mix:** `mix.mp3` is now the full arrangement from `jukebox.render.arrange` (data melody + bands, model harmony, recorded voices, about −16 LUFS). See `demo/README.md` for the listening guide and the before/after fidelity table; `arrange_report.json` has the numbers.
