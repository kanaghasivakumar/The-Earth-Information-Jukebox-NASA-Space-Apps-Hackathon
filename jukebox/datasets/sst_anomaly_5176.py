"""5176 Sea Surface Temperature Anomaly, near real time (MUR SST) -> the "today" track.

    python -m jukebox.datasets.sst_anomaly_5176 [--fps 4] [--width 512]

The SVS movie is regenerated daily: one frame per day for the last ~2.5 years
(901 frames, 30 s at 30p, equirectangular). Frame dates come from the SVS frames
folder listing (sst_mur_anomaly_YYYYMMDD.exr), so frame i is date i.
Colorbar: -4..+4 degC (sst_mur_anomaly_auto_colorbar_01). Land (dark grey), sea ice /
Antarctica (light grey) and the burned-in date box are rejected by the colour match.

Music: lead = area-weighted global mean anomaly; six latitude-band voices; controls:
geophony = global anomaly, brightness = share of ocean warmer than +1 degC.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from ..data import catalog
from ..data.colorbar import Colorbar
from ..data.features import BANDS, sequence_features
from ..data.ingest import download
from ..render.synth import render_notes
from ..score.mapping import PitchMap, band_voices, melody_from_curve
from .common import export_demo, iter_frames, movie_info, norm01, roundtrip, write_track, write_wav

PAGE = 5176
BASE = "https://svs.gsfc.nasa.gov/vis/a000000/a005100/a005176/"
MOVIE = BASE + "sst_anomaly_30_sec_4096x2048_2x1_30p.mp4"
CBAR = BASE + "sst_mur_anomaly_auto_colorbar_01_print.jpg"
CBAR_BBOX = (130, 250, 895, 276)     # gradient between the -4 and +4 ticks (1024x512 print)
FRAMES_DIR = BASE + "frames/4096x2048_2x1_30p/sst_mur_anomaly/"
VMIN, VMAX = -4.0, 4.0


def frame_dates(cache: Path) -> list[str]:
    if not cache.exists():
        with urllib.request.urlopen(urllib.request.Request(FRAMES_DIR, headers=catalog.UA), timeout=60) as r:
            html = r.read().decode("utf-8", "replace")
        names = sorted(set(re.findall(r"sst_mur_anomaly_(\d{8})\.", html)))
        cache.write_text("\n".join(names))
    return [d for d in cache.read_text().split() if d]


def mask_labels(f: np.ndarray) -> np.ndarray:
    """Black out the burned-in date box (lower-left)."""
    f = f.copy()
    h, w = f.shape[:2]
    f[int(h * 0.89):, : int(w * 0.15)] = 0
    return f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--demo", type=Path, default=Path("demo"))
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--max-delta-e", type=float, default=10.0)
    a = ap.parse_args(argv)
    d = a.data / str(PAGE)
    out = a.out / str(PAGE)
    out.mkdir(parents=True, exist_ok=True)
    page = catalog.fetch(PAGE, a.data.parent / "catalog")
    movie = download(MOVIE, d / Path(MOVIE).name)
    cb_img = cv2.imread(str(download(CBAR, d / Path(CBAR).name)))[..., ::-1]
    cbar = Colorbar.from_image(cb_img, VMIN, VMAX, units="°C", bbox=CBAR_BBOX)
    info = movie_info(movie)
    dates = frame_dates(d / "frame_dates.txt")

    cache = d / f"fields_{a.fps:g}fps_{a.width}.npz"
    if cache.exists():
        z = np.load(cache)
        fields, valid, idx, mid = z["fields"], z["valid"], z["idx"], z["mid"]
    else:
        fields, valid = [], []
        mid = None
        for i, f in enumerate(iter_frames(movie, a.width, a.fps)):
            f = mask_labels(f)
            v, m, _ = cbar.invert(f, a.max_delta_e)
            fields.append(v.astype(np.float32)); valid.append(m.mean())
            if i == 60:
                mid = f
        fields, valid = np.stack(fields), np.array(valid)
        idx = np.minimum(np.round(np.arange(len(fields)) / a.fps * info["fps"]).astype(int), info["frames"] - 1)
        np.savez_compressed(cache, fields=fields, valid=valid, idx=idx, mid=mid)
    T = len(fields)
    t = np.arange(T) / a.fps
    fdates = [dates[i] if len(dates) == info["frames"] and i < len(dates) else None for i in idx]
    feats = sequence_features(fields, t)
    gm = feats["mean"]
    duration = float(info["duration"])

    lo, hi = np.nanpercentile(gm, 1), np.nanpercentile(gm, 99)
    pad = 0.05 * (hi - lo + 1e-9)
    lead_pm = PitchMap(lo - pad, hi + pad, low=62, span=24, root=2, scale="major_pent")
    band_pm = PitchMap(-1.5, 2.0, low=43, span=30, root=2, scale="major_pent")
    lead = melody_from_curve(t, gm, lead_pm, "lead global", step=None)
    for n in lead:
        k = int(round(n.start * a.fps))
        n.source = {"date": fdates[k] if k < len(fdates) else None, "series": "global mean SST anomaly", "units": "degC"}
    names = [f"band {n}..{s}" for n, s in BANDS]
    bands = band_voices(t, feats["zonal"], band_pm, names=names)
    for n in bands:
        n.velocity = int(n.velocity * 0.6)
    stems = {"lead_global": render_notes(lead, duration, "pluck", gain=.3),
             "bands": render_notes(bands, duration, "softpad", gain=.07)}
    groups = {"lead_global": "melody", "bands": "bands"}

    fid, est = roundtrip(stems["lead_global"], lead_pm, t, gm)
    rng = np.random.default_rng(0)
    shuf = gm[rng.permutation(T)]
    ctrl = render_notes(melody_from_curve(t, shuf, lead_pm, "ctrl", step=None), duration, "pluck", gain=.3)
    write_wav(out / "control_shuffled.wav", ctrl / max(1e-9, np.abs(ctrl).max()) * 0.9)
    fid_ctrl, _ = roundtrip(ctrl, lead_pm, t, gm)

    warm = np.array([np.nanmean(f[np.isfinite(f)] > 1.0) if np.isfinite(f).any() else 0 for f in fields])
    curves = {"geophony": norm01(gm, lead_pm.vmin, lead_pm.vmax), "anthrophony": np.zeros(T),
              "biophony": np.zeros(T), "brightness": norm01(warm, 0, max(1e-6, np.nanpercentile(warm, 99)))}
    d0 = fdates[0] or "?"
    d1 = fdates[-1] or "?"
    pretty = lambda s: dt.datetime.strptime(s, "%Y%m%d").strftime("%d %b %Y") if s and s != "?" else "?"
    desc = (f"Daily sea surface temperature anomaly from the MUR analysis, {pretty(d0)} to {pretty(d1)}, "
            "updated by NASA every day. The plucked lead follows the global ocean's average departure from "
            "normal and six softer voices follow latitude bands from the Arctic to the Southern Ocean. "
            "Higher pitch means warmer than normal.")
    legend = [{"voice": "lead_global", "meaning": f"global mean SST anomaly, D4 = {lead_pm.vmin:+.2f} degC, D6 = {lead_pm.vmax:+.2f} degC"},
              {"voice": "bands", "meaning": "six latitude bands, north higher, -1.5..+2 degC"},
              {"voice": "geophony", "meaning": "control: global anomaly"},
              {"voice": "brightness", "meaning": "control: share of ocean warmer than +1 degC"},
              {"voice": "anthrophony/biophony", "meaning": "silent: not measured by this dataset"}]
    write_track(out, page_id=PAGE, title=page.get("title", "SST Anomaly - Near Real Time"), video_url=MOVIE,
                duration_s=duration, units="°C anomaly", description=desc, legend=legend, stems=stems,
                stem_groups=groups, notes=lead + bands, controls_t=t, curves=curves,
                fidelity={"r2": fid["r2"], "pearson_r": fid["pearson_r"], "mae": fid["mae"], "control_r2": fid_ctrl["r2"]},
                data_source={"music": "pixel inversion of the SVS movie with the SVS -4..+4 degC colorbar",
                             "colorbar_url": CBAR, "underlying": "JPL MUR SST anomaly (PO.DAAC)",
                             "dates": [d0, d1]},
                mix_gains={"lead_global": 1.0, "bands": 0.8},
                extra={"dates": fdates})
    report = {"id": PAGE, "title": page.get("title"), "movie": MOVIE, "colorbar": CBAR, "colorbar_bbox": CBAR_BBOX,
              "fps": a.fps, "width": a.width, "frames": T, "duration_s": duration, "dates": [d0, d1],
              "movie_frames": info["frames"], "dated_frames_listed": len(dates),
              "colorbar_resolution": cbar.resolution, "valid_pixel_fraction_mean": float(np.mean(valid)),
              "global_mean_range": [float(np.nanmin(gm)), float(np.nanmax(gm))],
              "lead_pitchmap": lead_pm.describe(), "band_pitchmap": band_pm.describe(),
              "audio_roundtrip": {"lead": fid, "shuffled_control": fid_ctrl},
              "notes": {"lead": len(lead), "bands": len(bands)}}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    plot(out, t, feats, gm, est, mid, fields, cb_img, fdates)
    export_demo(out, a.demo / str(PAGE), names=("mix", "lead_global", "control_shuffled"))
    print(json.dumps(report, indent=1, default=float))


def plot(out, t, feats, gm, est, mid, fields, cb_img, fdates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    ax[0, 0].imshow(mid); ax[0, 0].set_title("frame 60 (labels masked)"); ax[0, 0].axis("off")
    im = ax[0, 1].imshow(fields[-1], cmap="RdBu_r", vmin=-4, vmax=4)
    plt.colorbar(im, ax=ax[0, 1], fraction=.03, label="degC")
    ax[0, 1].set_title(f"inverted anomaly, last frame {fdates[-1]}"); ax[0, 1].axis("off")
    for b, (n, s) in enumerate(BANDS):
        ax[1, 0].plot(t, feats["zonal"][:, b], label=f"{n}..{s}")
    ax[1, 0].legend(fontsize=7); ax[1, 0].set_title("zonal mean anomaly (degC)"); ax[1, 0].set_xlabel("video s")
    ax[1, 1].plot(t, gm, label="global mean (data)"); ax[1, 1].plot(t, est, ".", label="recovered from audio")
    ax[1, 1].legend(); ax[1, 1].set_title("audio round-trip"); ax[1, 1].set_xlabel("video s")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
