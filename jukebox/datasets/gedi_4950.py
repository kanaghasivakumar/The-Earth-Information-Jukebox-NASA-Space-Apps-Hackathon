"""4950 GEDI Forest Height -> biophony level + canopy melody as the globe turns.

    python -m jukebox.datasets.gedi_4950 [--fps 4] [--width 960]

The movie (GEDI_global_w_colorbar_1080p60.mp4) is a lit 3D globe making one turn in
~51 s, with a burned-in colorbar "Height Above Ground, 0..50 m" (the 0to50 variant;
SVS also publishes 0to40 and 0to75 colorbars for other cuts). Pixels are inverted with
a shade-tolerant chromaticity match (datasets/chroma.py) against the SVS 0to50 colorbar.

Masks: outside the globe disc, the limb (rho > 0.92, where extrusions smear), ocean
(blue), dark night side (too dark to read), the burned-in colorbar/labels (lower left),
and non-vegetated land (< 3 m canopy). Weights: screen pixel -> sphere area 1/cos(view angle).

Music
  melody  area-weighted mean canopy height over vegetated visible pixels (rest if < 0.5 % of disc)
  bands   the same mean for the upper / middle / lower third of the visible disc
  controls biophony = canopy height x cover over the visible disc; geophony = ocean share;
           brightness = tall-forest (>= 25 m) share of vegetated pixels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..data import catalog
from ..data.ingest import download
from ..proof.fidelity import score
from ..render.synth import render_notes
from ..score.mapping import PitchMap, melody_from_curve
from .chroma import ChromaColorbar
from .common import (export_demo, grab_frame, iter_frames, movie_info, norm01, roundtrip, write_track, write_wav)

PAGE = 4950
BASE = "https://svs.gsfc.nasa.gov/vis/a000000/a004900/a004950/"
MOVIE = BASE + "GEDI_global_w_colorbar_1080p60.mp4"
CBAR = BASE + "GEDI_colorbar_0to50_01_print.jpg"
CBAR_BBOX = (51, 502, 486, 522)       # gradient strip in the 1024x576 print (0 m .. 50 m)
VMIN, VMAX = 0.0, 50.0
VEG_MIN = 3.0                         # m: below this a pixel counts as non-vegetated
TALL = 25.0


def find_disc(frames: list[np.ndarray]) -> tuple[float, float, float]:
    """Globe centre (cx, cy) and radius from the union of non-black pixels in sample frames."""
    nb = np.zeros(frames[0].shape[:2], bool)
    for f in frames:
        nb |= f.max(2) > 12
    h, w = nb.shape
    nb[int(h * .78):, : int(w * .35)] = False          # burned-in colorbar
    rows = np.where(nb.sum(1) > 3)[0]
    cy = (rows[0] + rows[-1]) / 2
    R = (rows[-1] - rows[0]) / 2
    # widest row ~ the equator of the disc; its midpoint gives cx
    mid = nb[int(cy) - 5:int(cy) + 5]
    xs = np.where(mid.any(0))[0]
    cx = (xs[0] + xs[-1]) / 2
    return float(cx), float(cy), float(R)


def find_inframe_colorbar(f: np.ndarray):
    """(x0, x1, y0, y1) of the burned-in gradient (0 m at x0, 50 m at x1), lower-left of the frame."""
    h, w = f.shape[:2]
    oy, ox = int(h * .78), 0
    reg = f[oy:, : int(w * .35)].max(2) > 30
    rows = np.where(reg.mean(1) > 0.3)[0]
    cnt = reg[rows[0]:rows[-1] + 1].sum(0)
    full = np.where(cnt >= np.median(cnt[cnt > 0]))[0]
    tick = np.where(cnt > np.median(cnt[cnt > 0]))[0]   # tick marks make columns taller
    x0 = int(full[0]) + 1
    x1 = int(tick[tick > x0 + 20][-1]) if tick[tick > x0 + 20].size else int(full[-1])
    y0, y1 = rows[0] + oy, rows[-1] + oy
    return x0, x1, int(y0 + (y1 - y0) * 0.35), int(y1 - 1)


def frame_features(f, cc: ChromaColorbar, disc, rho_max=0.92, max_dist=4.0):
    h, w = f.shape[:2]
    cx, cy, R = disc
    yy, xx = np.mgrid[0:h, 0:w]
    rho = np.hypot(xx - cx, yy - cy) / R
    inside = rho < rho_max
    inside[int(h * .78):, : int(w * .35)] = False
    wgt = np.where(inside, 1 / np.sqrt(np.clip(1 - rho ** 2, 1e-3, 1)), 0.0)
    rgb = f.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ocean = inside & (b > r + 8) & (b > g - 5)
    cand = inside & ~ocean
    vals = np.full((h, w), np.nan)
    ok = np.zeros((h, w), bool)
    if cand.any():
        v, m, _ = cc.invert(f[cand], max_dist=max_dist)
        vals[cand] = v
        ok[cand] = m
    veg = ok & (vals >= VEG_MIN)
    W = wgt.sum()
    def wmean(mask):
        sw = wgt[mask].sum()
        return float((vals[mask] * wgt[mask]).sum() / sw) if sw > 0.005 * W else np.nan
    thirds = []
    for k in range(3):
        y0, y1 = cy - R + 2 * R * k / 3, cy - R + 2 * R * (k + 1) / 3
        thirds.append(wmean(veg & (yy >= y0) & (yy < y1)))
    land = cand
    return {
        "mean_veg": wmean(veg),
        "thirds": thirds,
        "veg_frac": float(wgt[veg].sum() / W),
        "ocean_frac": float(wgt[ocean].sum() / W),
        "land_frac": float(wgt[land].sum() / W),
        "valid_land_frac": float(wgt[ok].sum() / max(1e-9, wgt[land].sum())),
        "valid_disc_frac": float(wgt[ok | ocean].sum() / W),
        "biomass": float((np.nan_to_num(vals) * wgt * veg).sum() / W),   # m x cover
        "tall_frac": float(wgt[veg & (vals >= TALL)].sum() / max(1e-9, wgt[veg].sum())),
    }, vals


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--demo", type=Path, default=Path("demo"))
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--max-dist", type=float, default=4.0)
    a = ap.parse_args(argv)
    d = a.data / str(PAGE)
    out = a.out / str(PAGE)
    out.mkdir(parents=True, exist_ok=True)
    page = catalog.fetch(PAGE, a.data.parent / "catalog")
    movie = download(MOVIE, d / Path(MOVIE).name)
    cb_path = download(CBAR, d / Path(CBAR).name)
    cb_img = cv2.imread(str(cb_path))[..., ::-1]
    cc = ChromaColorbar.from_image(cb_img, VMIN, VMAX, CBAR_BBOX, units="m")
    info = movie_info(movie)

    # geometry + in-frame colorbar check
    samples = [grab_frame(movie, t, a.width) for t in np.linspace(1, info["duration"] - 1, 6)]
    disc = find_disc(samples)
    x0, x1, y0, y1 = find_inframe_colorbar(samples[0])
    strip = samples[0][y0:y1, x0:x1 + 1]
    truth_cb = np.broadcast_to(np.linspace(VMIN, VMAX, strip.shape[1]), strip.shape[:2])
    v_cb, ok_cb, _ = cc.invert(strip, max_dist=a.max_dist, min_bright=0.0)
    cb_check = score(truth_cb[ok_cb], v_cb[ok_cb])
    cb_check["valid_frac"] = float(ok_cb.mean())
    from ..data.colorbar import Colorbar
    lab_cb = Colorbar.from_image(cb_img, VMIN, VMAX, bbox=CBAR_BBOX, samples=128)
    v_lab, ok_lab, _ = lab_cb.invert(strip, 15)
    cb_check_lab = score(truth_cb[ok_lab], v_lab[ok_lab])

    # 1. invert every sampled frame
    cache = d / f"features_{a.fps:g}fps_{a.width}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        feats, mid_vals, mid_frame = z["feats"].item(), z["mid_vals"], z["mid_frame"]
    else:
        rows, mid_vals, mid_frame = [], None, None
        n_exp = int(info["duration"] * a.fps)
        for i, f in enumerate(iter_frames(movie, a.width, a.fps)):
            ft, vals = frame_features(f, cc, disc, max_dist=a.max_dist)
            rows.append(ft)
            if i == int(n_exp * 0.45):
                mid_vals, mid_frame = vals.astype(np.float32), f.copy()
        feats = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
        np.savez_compressed(cache, feats=np.array(feats, dtype=object), mid_vals=mid_vals, mid_frame=mid_frame)
    T = len(feats["mean_veg"])
    t = np.arange(T) / a.fps
    duration = float(T / a.fps)

    # 2. score
    mel = feats["mean_veg"]
    lo, hi = np.nanpercentile(mel, 2), np.nanpercentile(mel, 98)
    pad = 0.05 * (hi - lo + 1e-9)
    mel_pm = PitchMap(lo - pad, hi + pad, low=57, span=24, root=9, scale="major_pent")       # A3..A5
    loud = norm01(feats["veg_frac"], 0, np.nanpercentile(feats["veg_frac"], 98))
    step = 1 / a.fps
    melody = melody_from_curve(t, mel, mel_pm, "melody canopy", step=None, loudness=loud)
    for n in melody:
        n.source = {"series": "area-weighted mean canopy height, vegetated visible pixels", "units": "m"}
    band_pm = PitchMap(VMIN, 30.0, low=40, span=24, root=9, scale="major_pent")
    names = ["upper", "middle", "lower"]
    bands = {}
    for k, nm in enumerate(names):
        pmk = PitchMap(band_pm.vmin, band_pm.vmax, band_pm.low + 5 * (1 - k), band_pm.span, 9, "major_pent")
        bands[nm] = (pmk, melody_from_curve(t, feats["thirds"][:, k], pmk, f"band {nm} third", step=None,
                                            pan=0.0, vel_range=(35, 70)))
    stems = {"melody_canopy": render_notes(melody, duration, "pluck", gain=.3)}
    groups = {"melody_canopy": "melody"}
    for nm in names:
        stems[f"band_{nm}"] = render_notes(bands[nm][1], duration, "softpad", gain=.08)
        groups[f"band_{nm}"] = "bands"

    # 3. proof
    fid, est = roundtrip(stems["melody_canopy"], mel_pm, t, mel)
    rng = np.random.default_rng(0)
    fin = np.isfinite(mel)
    shuf = mel.copy()
    shuf[fin] = mel[fin][rng.permutation(fin.sum())]
    ctrl_audio = render_notes(melody_from_curve(t, shuf, mel_pm, "ctrl", step=None), duration, "pluck", gain=.3)
    write_wav(out / "control_shuffled.wav", ctrl_audio / max(1e-9, np.abs(ctrl_audio).max()) * 0.9)
    fid_ctrl, _ = roundtrip(ctrl_audio, mel_pm, t, mel)
    fid_bands = {nm: roundtrip(stems[f"band_{nm}"], bands[nm][0], t, feats["thirds"][:, k])[0]
                 for k, nm in enumerate(names)}

    # 4. controls
    curves = {
        "biophony": norm01(feats["biomass"], 0, np.nanpercentile(feats["biomass"], 99)),
        "geophony": norm01(feats["ocean_frac"], 0, 1),
        "anthrophony": np.zeros(T),
        "brightness": norm01(feats["tall_frac"], 0, max(1e-6, np.nanpercentile(feats["tall_frac"], 99))),
    }
    notes = melody + [n for nm in names for n in bands[nm][1]]
    desc = ("Forest canopy height measured by the GEDI lidar on the International Space Station, "
            "shown on a globe that turns once. The plucked melody follows the average height of the "
            "visible forest: it climbs when the Amazon, Congo and Southeast Asian rainforests face you "
            "and rests over open ocean. A forest chorus swells with how much tall forest is in view.")
    legend = [{"voice": "melody_canopy", "meaning": f"mean canopy height of visible vegetation, A3 = {mel_pm.vmin:.1f} m, A5 = {mel_pm.vmax:.1f} m"},
              {"voice": "bands", "meaning": "same mean for the upper, middle and lower third of the globe, 0-30 m over 2 octaves"},
              {"voice": "biophony", "meaning": "control: canopy height x forest cover in view"},
              {"voice": "geophony", "meaning": "control: share of the visible globe that is ocean"},
              {"voice": "brightness", "meaning": "control: share of vegetation taller than 25 m"},
              {"voice": "anthrophony", "meaning": "silent: not measured by this dataset"}]
    write_track(out, page_id=PAGE, title=page.get("title", "GEDI Forest Height"), video_url=MOVIE,
                duration_s=duration, units="m (canopy height)", description=desc, legend=legend, stems=stems,
                stem_groups=groups, notes=notes, controls_t=t, curves=curves,
                fidelity={"r2": fid["r2"], "pearson_r": fid["pearson_r"], "mae": fid["mae"], "control_r2": fid_ctrl["r2"]},
                data_source={"music": "pixel inversion of the SVS movie with the SVS 0-50 m colorbar",
                             "colorbar_url": CBAR, "underlying": "GEDI L2B RH100, Apr 2019 - Apr 2020"},
                mix_gains={"melody_canopy": 1.0, **{f"band_{nm}": 0.8 for nm in names}})
    report = {
        "id": PAGE, "title": page.get("title"), "movie": MOVIE, "colorbar": CBAR,
        "colorbar_variant": "0-50 m (burned-in bar in the movie is labelled 0..50 with an open-ended arrow)",
        "fps": a.fps, "width": a.width, "frames": T, "duration_s": duration,
        "disc_px": disc, "inframe_colorbar_px": [x0, x1, y0, y1],
        "inframe_colorbar_check_chroma": cb_check, "inframe_colorbar_check_lab": cb_check_lab,
        "colorbar_resolution": cc.resolution,
        "valid_land_fraction_mean": float(np.nanmean(feats["valid_land_frac"])),
        "valid_disc_fraction_mean": float(np.nanmean(feats["valid_disc_frac"])),
        "veg_fraction_range": [float(np.nanmin(feats["veg_frac"])), float(np.nanmax(feats["veg_frac"]))],
        "melody_rest_fraction": float(np.mean(~np.isfinite(mel))),
        "mean_veg_range_m": [float(np.nanmin(mel)), float(np.nanmax(mel))],
        "melody_pitchmap": mel_pm.describe(),
        "audio_roundtrip": {"melody": fid, "bands": fid_bands, "shuffled_control": fid_ctrl},
        "notes": {"melody": len(melody), "bands": {nm: len(bands[nm][1]) for nm in names}},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    plot(out, t, feats, est, mid_frame, mid_vals, disc, cb_img)
    export_demo(out, a.demo / str(PAGE), names=("mix", "melody_canopy", "control_shuffled"))
    print(json.dumps(report, indent=1, default=float))


def plot(out, t, feats, est, frame, vals, disc, cb_img):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    ax[0, 0].imshow(frame)
    cx, cy, R = disc
    ax[0, 0].add_patch(plt.Circle((cx, cy), R * 0.92, fill=False, ec="c", lw=.6))
    ax[0, 0].set_title("frame (45%) + analysed disc"); ax[0, 0].axis("off")
    im = ax[0, 1].imshow(np.where(vals >= VEG_MIN, vals, np.nan), cmap="YlGn", vmin=0, vmax=50)
    plt.colorbar(im, ax=ax[0, 1], fraction=.03, label="m")
    ax[0, 1].set_title("inverted canopy height (vegetated, valid)"); ax[0, 1].axis("off")
    ax[1, 0].plot(t, feats["mean_veg"], label="mean veg height (m)")
    for k, nm in enumerate(["upper", "middle", "lower"]):
        ax[1, 0].plot(t, feats["thirds"][:, k], lw=.7, label=f"{nm} third")
    ax2 = ax[1, 0].twinx()
    ax2.plot(t, feats["veg_frac"], "k:", lw=.8, label="vegetated share of disc")
    ax[1, 0].legend(fontsize=7, loc="upper left"); ax[1, 0].set_xlabel("s")
    ax[1, 1].plot(t, feats["mean_veg"], label="data"); ax[1, 1].plot(t, est, ".", label="recovered from audio")
    ax[1, 1].set_title("audio round-trip"); ax[1, 1].legend(); ax[1, 1].set_xlabel("s")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
