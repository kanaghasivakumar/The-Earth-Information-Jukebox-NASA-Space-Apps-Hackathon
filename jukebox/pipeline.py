"""M1 end-to-end: one continuous-colorbar SVS movie -> data melody + band voices + proof.

    python -m jukebox.pipeline --id 3912 --vmin 0 --vmax 32 --units "°C" --fps 2

Outputs in outputs/<id>/: frames preview, features.npz, score.mid, mix.wav,
lead.wav, control_shuffled.wav, timeline.json, report.json, overview.png
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from .data import catalog
from .data.colorbar import Colorbar, auto_bbox
from .data.features import BANDS, sequence_features
from .data.ingest import ingest
from .proof.fidelity import recover_curve, score
from .render.synth import SR, mix, render_notes
from .score.mapping import PitchMap, band_voices, melody_from_curve, timeline, to_midi


def load_rgb(url: str, dest: Path) -> np.ndarray:
    import cv2
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=catalog.UA), timeout=60) as r:
            dest.write_bytes(r.read())
    bgr = cv2.imread(str(dest), cv2.IMREAD_COLOR)
    return bgr[..., ::-1].copy()


def write_wav(path: Path, audio: np.ndarray, sr: int = SR):
    import soundfile as sf
    sf.write(str(path), audio, sr, subtype="PCM_16")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--vmin", type=float, required=True)
    ap.add_argument("--vmax", type=float, required=True)
    ap.add_argument("--units", default="")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--movie-url", default=None)
    ap.add_argument("--colorbar-url", default=None)
    ap.add_argument("--max-delta-e", type=float, default=12.0)
    ap.add_argument("--shade-invariant", action="store_true", help="match hue, not lightness (dimmed/blended renders)")
    ap.add_argument("--min-chroma", type=float, default=0.0)
    ap.add_argument("--l-weight", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=0.5, help="seconds per melody note")
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    a = ap.parse_args(argv)

    out = a.out / str(a.id)
    out.mkdir(parents=True, exist_ok=True)

    # 1. ingest
    d = a.data / str(a.id)
    if not (d / "frames.npy").exists():
        ingest(a.id, a.data, a.fps, a.width, a.movie_url)
    frames = np.load(d / "frames.npy")
    t = np.load(d / "timestamps.npy")
    page = catalog.fetch(a.id, a.data.parent / "catalog")

    # 2. colorbar -> LUT
    cb_url = a.colorbar_url or (catalog.colorbars(page) or [{}])[0].get("url")
    if not cb_url:
        raise SystemExit("no colorbar found; pass --colorbar-url")
    cb_img = load_rgb(cb_url, d / Path(cb_url).name)
    bbox = auto_bbox(cb_img)
    cbar = Colorbar.from_image(cb_img, a.vmin, a.vmax, units=a.units, bbox=bbox)

    # 3. invert every frame
    fields, valid = [], []
    for f in frames:
        v, m, _ = cbar.invert(f, a.max_delta_e, l_weight=a.l_weight, min_chroma=a.min_chroma, shade_invariant=a.shade_invariant)
        fields.append(v)
        valid.append(m.mean())
    fields = np.stack(fields)
    feats = sequence_features(fields, t)
    np.savez_compressed(out / "features.npz", **{k: v for k, v in feats.items()})

    # 4. score: lead = global mean (zoomed, documented range), bands = absolute colorbar range
    gm = feats["mean"]
    lo, hi = np.nanpercentile(gm, 1), np.nanpercentile(gm, 99)
    pad = 0.05 * (hi - lo + 1e-9)
    lead_pm = PitchMap(lo - pad, hi + pad, low=60, span=24, root=2, scale="major_pent")  # D, 2 octaves
    band_pm = PitchMap(a.vmin, a.vmax, low=36, span=36, root=2, scale="major_pent")
    lead = melody_from_curve(t, gm, lead_pm, "lead", step=a.step, loudness=feats["activity"])
    names = [f"{n}..{s}" for n, s in BANDS]
    bands = band_voices(t, feats["zonal"], band_pm, names=[f"band {x}" for x in names], step=a.step * 4)
    for n in bands:
        n.velocity = int(n.velocity * 0.6)
    notes = lead + bands
    to_midi(notes, out / "score.mid", programs={"lead": 11, **{f"band {x}": 89 for x in names}})
    (out / "timeline.json").write_text(json.dumps(timeline(notes)))

    # 5. render
    dur = float(t[-1] + a.step)
    lead_audio = render_notes(lead, dur, "pluck")
    band_audio = render_notes(bands, dur, "pad", gain=.12)
    write_wav(out / "lead.wav", mix(lead_audio))
    write_wav(out / "mix.wav", mix(lead_audio, band_audio, gains=[1.0, 0.7]))

    # 6. proof: recover the curve from the lead stem; shuffled control
    grid = np.arange(t[0], t[-1] + 1e-9, a.step)
    truth = np.interp(grid, t[np.isfinite(gm)], gm[np.isfinite(gm)])
    est = recover_curve(lead_audio, SR, lead_pm, grid)
    real = score(truth, est)
    rng = np.random.default_rng(0)
    shuffled = truth[rng.permutation(len(truth))]
    ctrl_notes = melody_from_curve(grid, shuffled, lead_pm, "lead", step=None)
    ctrl_audio = render_notes(ctrl_notes, dur, "pluck")
    write_wav(out / "control_shuffled.wav", mix(ctrl_audio))
    ctrl = score(truth, recover_curve(ctrl_audio, SR, lead_pm, grid))

    report = {
        "id": a.id, "title": page.get("title"), "frames": int(len(frames)), "fps": a.fps,
        "duration_s": dur, "colorbar_url": cb_url, "colorbar_bbox": bbox,
        "colorbar_resolution": cbar.resolution, "units": a.units,
        "valid_pixel_fraction_mean": float(np.mean(valid)),
        "global_mean_range": [float(np.nanmin(gm)), float(np.nanmax(gm))],
        "lead_pitchmap": lead_pm.describe(), "band_pitchmap": band_pm.describe(),
        "notes": {"lead": len(lead), "bands": len(bands)},
        "fidelity_real": real, "fidelity_shuffled_control": ctrl,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    plot(out, t, feats, grid, truth, est, names, a.units, cb_img, bbox, frames)
    print(json.dumps(report, indent=1))


def plot(out, t, feats, grid, truth, est, names, units, cb_img, bbox, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    ax[0, 0].imshow(frames[len(frames) // 2]); ax[0, 0].set_title("frame (middle)"); ax[0, 0].axis("off")
    x0, y0, x1, y1 = bbox
    ax[0, 1].imshow(cb_img); ax[0, 1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="k"))
    ax[0, 1].set_title("colorbar + detected strip"); ax[0, 1].axis("off")
    for b, n in enumerate(names):
        ax[1, 0].plot(t, feats["zonal"][:, b], label=n)
    ax[1, 0].set_title(f"zonal means ({units})"); ax[1, 0].legend(fontsize=7); ax[1, 0].set_xlabel("s")
    ax[1, 1].plot(grid, truth, label="data (global mean)")
    ax[1, 1].plot(grid, est, ".", label="recovered from audio")
    ax[1, 1].set_title("round-trip fidelity"); ax[1, 1].legend(); ax[1, 1].set_xlabel("s")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
