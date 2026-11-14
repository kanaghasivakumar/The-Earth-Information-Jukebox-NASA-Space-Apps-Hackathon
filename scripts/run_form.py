"""Form layer for one dataset: embed frames -> sections/regions -> apply to the score -> form.json + overview.png

    srun -A p32576 -p short -t 00:30:00 --mem=16G -c 4 \
        /projects/e32706/omb8654/conda/envs/jukebox/bin/python scripts/run_form.py --id 3912

Reads data/raw/<id>/{frames,timestamps}.npy and (optionally) a score timeline.json
(default: outputs/<id>/timeline.json, else demo/<id>/timeline.json).
Writes outputs/<id>_form/: form.json, embeddings.npz, form_timeline.json, score_form.mid, overview.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", "/projects/e32706/omb8654/hf_cache")

from jukebox.form.apply import apply_form  # noqa: E402
from jukebox.form.embed import embed_frames  # noqa: E402
from jukebox.form.sections import analyse, foote_novelty, self_similarity, smooth  # noqa: E402
from jukebox.score.mapping import Note, timeline, to_midi  # noqa: E402


def camera_proxy(frames: np.ndarray) -> dict:
    """Signals that move when the CAMERA moves rather than the data: global translation (phase
    correlation) and change of the low-saturation 'background/land' mask between frames."""
    import cv2
    shift, mask_change, pix_change = [0.0], [0.0], [0.0]
    prev_g = prev_m = prev_f = None
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        m = hsv[..., 1] < 40
        if prev_g is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev_g, g)
            shift.append(float(np.hypot(dx, dy)))
            mask_change.append(float(np.mean(m != prev_m)))
            pix_change.append(float(np.mean(np.abs(f.astype(np.float32) - prev_f)) / 255))
        prev_g, prev_m, prev_f = g, m, f.astype(np.float32)
    return {"shift_px": np.array(shift), "bg_mask_change": np.array(mask_change), "pixel_change": np.array(pix_change)}


def curve_from_notes(notes: list[Note], voice: str, t: np.ndarray) -> np.ndarray:
    vn = sorted([n for n in notes if n.voice == voice], key=lambda n: n.start)
    out = np.full(len(t), np.nan)
    for n in vn:
        out[(t >= n.start) & (t < n.end)] = n.value
    return out


def data_fields(frames, cb_path: Path, vmin: float, vmax: float, grid=(8, 16)):
    """Invert the colorbar (same code path as jukebox.pipeline) -> SST-like fields, a coarse
    data-only embedding (block means), and global / zonal mean curves."""
    import cv2
    from jukebox.data.colorbar import Colorbar, auto_bbox
    from jukebox.data.features import BANDS, sequence_features
    cb = cv2.imread(str(cb_path))[..., ::-1].copy()
    cbar = Colorbar.from_image(cb, vmin, vmax, bbox=auto_bbox(cb))
    fields = np.stack([cbar.invert(f)[0] for f in frames])
    valid = float(np.isfinite(fields).mean())
    feats = sequence_features(fields, np.arange(len(fields)))
    T, H, W = fields.shape
    gh, gw = grid
    blk = fields[:, :H // gh * gh, :W // gw * gw].reshape(T, gh, H // gh, gw, W // gw)
    with np.errstate(all="ignore"):
        Xd = np.nanmean(blk, axis=(2, 4)).reshape(T, -1)
    keep = np.isfinite(Xd).mean(0) > 0.5
    Xd = Xd[:, keep]
    Xd = np.where(np.isfinite(Xd), Xd, np.nanmean(Xd, 0))
    curves = {"data: global mean": feats["mean"]}
    for b, (n, s) in enumerate(BANDS):
        curves[f"data: band {n}..{s}"] = feats["zonal"][:, b]
    return Xd, curves, valid


def agreement(form, X2, t, prominence, extra=None):
    """Run the same segmentation on another trajectory; how many boundaries match?"""
    f2 = analyse(X2, t, kernel_s=form.params["kernel_s"], min_section_s=form.params["min_section_s"],
                 prominence=prominence)
    b1 = [s.start_frame for s in form.sections[1:]]
    b2 = [s.start_frame for s in f2.sections[1:]]
    tol = max(1, form.params["kernel_half_frames"] // 2)
    return f2.novelty, {"boundaries_s": [float(t[b]) for b in b2], "labels": [s.label for s in f2.sections],
                        "matched": sum(any(abs(b - x) <= tol for x in b2) for b in b1), "of": len(b1),
                        "tolerance_s": tol * form.params["dt"], "dims": int(X2.shape[1]),
                        "novelty_corr": float(np.corrcoef(form.novelty, f2.novelty)[0, 1]), **(extra or {})}


def load_notes(path: Path) -> list[Note]:
    return [Note(**d) for d in json.loads(path.read_text())]


def boundary_stats(bounds_frames: list[int], signal: np.ndarray, win: int = 2) -> dict:
    """Is |signal| larger near boundaries than elsewhere? Ratio + a circular-shift permutation p-value."""
    s = np.nan_to_num(np.abs(signal))
    T = len(s)
    if not bounds_frames or s.std() == 0:
        return {"ratio": None, "p": None}

    def near_mean(bf):
        idx = sorted({min(T - 1, max(0, b + d)) for b in bf for d in range(-win, win + 1)})
        return s[idx].mean()

    obs = near_mean(bounds_frames) / (s.mean() + 1e-12)
    rng = np.random.default_rng(0)
    null = [near_mean([(b + k) % T for b in bounds_frames]) / (s.mean() + 1e-12)
            for k in rng.integers(1, T, 500)]
    return {"ratio": float(obs), "p": float((np.sum(np.array(null) >= obs) + 1) / (len(null) + 1))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--timeline", type=Path, default=None)
    ap.add_argument("--method", default="auto", choices=["auto", "dinov2", "handcrafted"])
    ap.add_argument("--kernel-s", type=float, default=None)
    ap.add_argument("--min-section-s", type=float, default=None)
    ap.add_argument("--prominence", type=float, default=0.1)
    ap.add_argument("--start-date", default=None, help="e.g. 2007-03 (3912) to annotate approximate dates")
    ap.add_argument("--months", type=float, default=None, help="months spanned by the video, e.g. 12")
    ap.add_argument("--instrument-change", action="store_true")
    ap.add_argument("--vmin", type=float, default=None, help="colorbar range: enables the data-field cross-check")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--colorbar", type=Path, default=None, help="colorbar PNG (default: first *cbar*.png in data dir)")
    a = ap.parse_args(argv)

    d = a.data / str(a.id)
    out = a.out / f"{a.id}_form"
    out.mkdir(parents=True, exist_ok=True)
    frames = np.load(d / "frames.npy")
    t = np.load(d / "timestamps.npy").astype(float)

    t0 = time.time()
    emb = embed_frames(frames, a.method)
    t_embed = time.time() - t0
    alt = embed_frames(frames, "handcrafted") if emb.method != "handcrafted" else None
    np.savez_compressed(out / "embeddings.npz", X=emb.X, t=t, method=emb.method,
                        **({"X_handcrafted": alt.X} if alt is not None else {}))
    form = analyse(emb.X, t, kernel_s=a.kernel_s, min_section_s=a.min_section_s, prominence=a.prominence)
    bounds = [s.start_frame for s in form.sections[1:]]

    # cross-check: the same novelty with the handcrafted (colour + flow) embedding
    agree = None
    nov_alt = None
    if alt is not None:
        form_alt = analyse(alt.X, t, kernel_s=form.params["kernel_s"], min_section_s=form.params["min_section_s"],
                           prominence=a.prominence)
        nov_alt = form_alt.novelty
        b_alt = [s.start_frame for s in form_alt.sections[1:]]
        tol = max(1, form.params["kernel_half_frames"] // 2)
        hit = sum(any(abs(b - x) <= tol for x in b_alt) for b in bounds)
        agree = {"handcrafted_boundaries_s": [float(t[b]) for b in b_alt], "matched": hit, "of": len(bounds),
                 "tolerance_s": tol * form.params["dt"],
                 "novelty_corr": float(np.corrcoef(form.novelty, nov_alt)[0, 1])}

    # cross-check 2: novelty on the colorbar-inverted DATA field itself (no render texture, no camera)
    data_check, nov_data, data_curves = None, None, {}
    cb_path = a.colorbar or next(iter(sorted(d.glob("*cbar*.png"))), None)
    if a.vmin is not None and a.vmax is not None and cb_path:
        Xd, data_curves, valid = data_fields(frames, cb_path, a.vmin, a.vmax)
        if Xd.shape[1] >= 4 and valid > 0.05:
            nov_data, data_check = agreement(form, Xd, t, a.prominence, {"valid_pixel_fraction": valid})
        else:
            data_check = {"skipped": f"colorbar inversion kept only {valid:.2%} of pixels", "valid_pixel_fraction": valid}
            data_curves = {}

    # score
    tl = a.timeline or next((p for p in [a.out / str(a.id) / "timeline.json", ROOT / "demo" / str(a.id) / "timeline.json"]
                             if p.exists()), None)
    notes = load_notes(tl) if tl else []
    score_check, nov_score = None, None
    if notes:   # cross-check 3: what the melody actually encodes (lead + band values from the score)
        cols = []
        for v in sorted({n.voice for n in notes if n.voice == "lead" or n.voice.startswith("band")}):
            c = curve_from_notes(notes, v, t)
            if np.isfinite(c).sum() > 2:
                c = np.where(np.isfinite(c), c, np.nanmean(c))
                cols.append((c - c.mean()) / (c.std() + 1e-9))
        if len(cols) >= 2:
            nov_score, score_check = agreement(form, np.stack(cols, 1), t, a.prominence, {"voices": len(cols)})
    fj = form.to_json()
    fj.update({"id": str(a.id), "embedding": {"method": emb.method, **emb.info, "seconds": round(t_embed, 1)}})
    if notes:
        new_notes, applied = apply_form(notes, form, instrument_change=a.instrument_change)
        fj["applied"] = applied
        fj["score_timeline"] = str(tl.relative_to(ROOT) if tl.is_relative_to(ROOT) else tl)
        (out / "form_timeline.json").write_text(json.dumps(timeline(new_notes)))
        voices = {n.voice for n in new_notes}
        progs = {v: (11 if v == "lead" else 89) for v in voices}
        progs.update(applied["programs"])
        to_midi(new_notes, out / "score_form.mid", programs=progs, drums=set(applied["drums"]))

    # skepticism: do boundaries track the data, or the camera / render?
    cam = camera_proxy(frames)
    diag = {"camera_shift_px_max": float(cam["shift_px"].max()),
            "bg_mask_change_max": float(cam["bg_mask_change"].max()),
            "boundary_vs_camera_shift": boundary_stats(bounds, cam["shift_px"]),
            "boundary_vs_bg_mask_change": boundary_stats(bounds, cam["bg_mask_change"]),
            "boundary_vs_pixel_change": boundary_stats(bounds, cam["pixel_change"]),
            "handcrafted_agreement": agree, "data_field_agreement": data_check,
            "score_data_agreement": score_check}
    curves = dict(data_curves)
    if notes:
        for v in sorted({n.voice for n in notes}):
            c = curve_from_notes(notes, v, t)
            if data_curves and v != "lead":
                continue   # band step-curves from notes are coarse; the inverted field is better
            if np.isfinite(c).sum() > 2:
                curves[v] = c
                dc = np.gradient(smooth(np.nan_to_num(c, nan=np.nanmean(c))[:, None], 2)[:, 0])
                diag[f"boundary_vs_d[{v}]"] = boundary_stats(bounds, dc)
    for k, c in data_curves.items():
        if np.isfinite(c).sum() > 2:
            dc = np.gradient(smooth(np.nan_to_num(c, nan=np.nanmean(c))[:, None], 2)[:, 0])
            diag[f"boundary_vs_d[{k}]"] = boundary_stats(bounds, dc)
    if a.start_date and a.months:
        y, m = map(int, a.start_date.split("-"))
        span = t[-1] + form.params["dt"]

        def date(sec):
            mm = (m - 1) + sec / span * a.months
            return f"{y + int(mm // 12)}-{int(mm % 12) + 1:02d} (+{(mm % 1) * 30:.0f}d)"
        for s in fj["sections"]:
            s["approx_date"] = date(s["start"])
        fj["date_assumption"] = f"linear: t=0 is {a.start_date}, video spans {a.months} months (no burned-in dates checked)"
    fj["diagnostics"] = diag
    (out / "form.json").write_text(json.dumps(fj, indent=1, default=float))
    plot(out / "overview.png", frames, t, form, cam, curves, nov_alt, str(a.id), emb.method,
         nov_data if nov_data is not None else nov_score,
         "colorbar-inverted data field" if nov_data is not None else "score data: lead + band values")
    summary = {"id": a.id, "method": emb.method, "embed_s": round(t_embed, 1), "k": form.k,
               "silhouette": form.silhouette, "sections": form.tuples(),
               "returns": [(s.index, s.label, s.start) for s in form.returns],
               "repetition_lag_s": form.repetition["lag_s"], "diagnostics": diag,
               "dates": [s.get("approx_date") for s in fj["sections"]]}
    print(json.dumps(summary, indent=1, default=float))


def plot(path, frames, t, form, cam, curves, nov_alt, name, method, nov_data=None, data_name="data"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    secs = form.sections
    n = len(secs)
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, max(n, 2), height_ratios=[1.0, 2.2, 1.2, 1.2])
    for i, s in enumerate(secs):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[(s.start_frame + s.end_frame - 1) // 2])
        ax.set_title(f"{s.label}{' (return)' if s.is_return else ''}\n{s.start:.1f}-{s.end:.1f}s", fontsize=8,
                     color=cmap(s.region % 10))
        ax.axis("off")
    half = max(n, 2) // 2
    ax = fig.add_subplot(gs[1, :half])
    ax.imshow(form.ssm, cmap="magma", extent=[t[0], t[-1], t[-1], t[0]])
    for s in secs[1:]:
        ax.axvline(s.start, color="c", lw=.8); ax.axhline(s.start, color="c", lw=.8)
    ax.set_title(f"{name}: self-similarity ({method}, centred cosine)"); ax.set_xlabel("s")
    ax = fig.add_subplot(gs[1, half:])
    prof = np.asarray(form.repetition["lag_profile"] or [0.0])
    ax.plot(np.arange(len(prof)) * form.params["dt"], prof, "k")
    if form.repetition["lag_s"]:
        ax.axvline(form.repetition["lag_s"], color="r", ls="--", label=f"lag {form.repetition['lag_s']:.1f}s")
        ax.legend()
    ax.set_title("repetition: mean SSM similarity vs lag"); ax.set_xlabel("lag (s)")
    ax = fig.add_subplot(gs[2, :])
    ax.plot(t, form.novelty, "k", label=f"novelty ({method})")
    if nov_alt is not None:
        ax.plot(t, nov_alt, color="0.6", lw=.8, label="novelty (handcrafted colour+flow)")
    if nov_data is not None:
        ax.plot(t, nov_data, color="g", lw=.9, ls="--", label=f"novelty ({data_name})")
    for s in secs:
        ax.axvspan(s.start, s.end, color=cmap(s.region % 10), alpha=.15)
        ax.text(s.start + .3, .92, s.label + ("*" if s.is_return else ""), fontsize=9, transform=ax.get_xaxis_transform())
    ax.plot(t, -0.1 - 0.05 * form.frame_regions, "|", ms=6, color="b", label="region per frame (hysteresis)")
    ax.set_xlim(t[0], t[-1] + form.params["dt"]); ax.legend(fontsize=7, loc="upper right"); ax.set_title("sections (* = return)")
    ax = fig.add_subplot(gs[3, :])
    for k, c in list(curves.items())[:8]:
        z = (c - np.nanmean(c)) / (np.nanstd(c) + 1e-9)
        ax.plot(t, z, lw=1.6 if k in ("lead", "data: global mean") else .7, label=k)
    ax.plot(t, cam["pixel_change"] / (cam["pixel_change"].max() + 1e-9) * 2 - 3, "k:", lw=.8, label="pixel change (render)")
    ax.plot(t, cam["shift_px"] / (cam["shift_px"].max() + 1e-9) * 2 - 3, "r:", lw=.8, label="camera shift proxy")
    for s in secs[1:]:
        ax.axvline(s.start, color="c", lw=.8)
    ax.set_xlim(t[0], t[-1] + form.params["dt"]); ax.legend(fontsize=6, ncol=4, loc="upper right")
    ax.set_title("data curves from the score (z-scored) vs boundaries"); ax.set_xlabel("video time (s)")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
