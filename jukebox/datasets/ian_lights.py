"""SVS 5058 Florida night lights before/after Hurricane Ian -> the city voice goes silent.

    python -m jukebox.datasets.ian_lights

Source: black_marble_slide_line4.mp4 (3840x2160, 30 fps, 15 s). A white slider line wipes west->east
(~1.5-5.5 s): left of it is September 30 2022 (after Ian), right of it August 30 (before). It holds,
then wipes back (~10-12.5 s). We take the clean before (frame 15) and after (frame 225) pictures,
invert the burned-in "Nighttime lights: less .. more" colorbar (relative, no numbers), and track
the slider line in every frame, so the sound changes exactly where the picture changes.

Music
- city (melody): 12 west->east column bands. When the line crosses a band, that band plays a run
  from its before-light pitch down to its after-light pitch (lights going out); on the way back the
  run rises again. Velocity = how much light the band had. Pan = column position.
- drone (bands): the light currently on screen (before part + after part), as a sustained chord.
- controls: anthrophony = light currently on screen / light before; biophony rises as the city
  goes dark; geophony = share of the screen showing the storm's aftermath (wind / rain bed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import catalog
from ..data.colorbar import Colorbar, auto_bbox
from ..proof.fidelity import score
from ..render.synth import midi_hz, render_notes
from ..score.mapping import Note, PitchMap, to_midi
from .events_common import (Stem, decode_stream, export_demo, note_pitches, render_curve_voice, smooth,
                            video_info, write_track)

PAGE = 5058
MOVIE = "https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005058/black_marble_slide_line4.mp4"
CBAR_REGION = (600, 250, 970, 330)        # full-res crop holding the colorbar strip
MASKS = [(0, 0, 1000, 560), (3000, 0, 3840, 130)]  # legend / inset map, date label
ZONES = {"Tampa Bay and Lakeland": (0, 540), "Bradenton and North Port": (540, 1350), "Fort Myers and Naples": (1350, 2160)}
BEFORE_K, AFTER_K = 15, 225


def line_x(frame, prev_x=None):
    """x of the white slider line (column whose pixels are near-white over most of the height)."""
    f = frame.astype(np.int16)
    white = (f.min(2) > 185) & (np.ptp(f, axis=2) < 30)
    col = white.mean(0)
    x = int(np.argmax(col))
    return (x if col[x] > 0.6 else None), float(col[x])


def light_map(rgb, cbar: Colorbar):
    v, ok, _ = cbar.invert(rgb, max_delta_e=14, min_chroma=18)
    L = np.where(ok, v, 0.0)
    for x0, y0, x1, y1 in MASKS:
        L[y0:y1, x0:x1] = 0
    return L


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=12)
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--demo", type=Path, default=Path("demo"))
    a = ap.parse_args(argv)
    out = a.out / str(PAGE)
    out.mkdir(parents=True, exist_ok=True)
    d = a.data / str(PAGE)
    src = d / Path(MOVIE).name
    if not src.exists():
        from ..data.ingest import download
        download(MOVIE, src)
    info = video_info(src)
    fps, duration = info["fps"], info["duration"]
    page = catalog.fetch(PAGE, a.data.parent / "catalog")

    # line track (960-wide decode, every frame)
    cache = d / "line_track.npz"
    if not cache.exists():
        xs, conf = [], []
        for f in decode_stream(src, None, 960, 540):
            x, c = line_x(f)
            xs.append(-1 if x is None else x * 4)
            conf.append(c)
        np.savez(cache, x=np.array(xs), conf=np.array(conf))
    lt = np.load(cache)
    lx = lt["x"].astype(float)
    lx[lx < 0] = np.nan
    t_fr = np.arange(len(lx)) / fps

    # before / after pictures, full res
    fb, fa = d / "before_f15.png", d / "after_f225.png"
    if not (fb.exists() and fa.exists()):
        fr = list(decode_stream(src, None, 3840, 2160, vf_extra=f"select='eq(n\\,{BEFORE_K})+eq(n\\,{AFTER_K})',setpts=N/FRAME_RATE/TB"))
        cv2.imwrite(str(fb), fr[0][..., ::-1]); cv2.imwrite(str(fa), fr[1][..., ::-1])
    before = cv2.imread(str(fb))[..., ::-1]
    after = cv2.imread(str(fa))[..., ::-1]
    H, W = before.shape[:2]
    x0, y0, x1, y1 = CBAR_REGION
    crop = before[y0:y1, x0:x1]
    bb = auto_bbox(crop)
    cbar = Colorbar.from_image(crop, 0.0, 1.0, bbox=bb, units="relative night light (less .. more)")
    Lb = light_map(before, cbar)
    La = light_map(after, cbar)
    # a hurricane removes lights, it does not add them: count "after" light only near "before" light
    near = cv2.dilate((Lb > 0).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    track_px = int(((La > 0) & ~near).sum())
    La = np.where(near, La, 0.0)

    edges = np.linspace(0, W, a.bands + 1).astype(int)
    B = np.array([Lb[:, edges[j]:edges[j + 1]].sum() for j in range(a.bands)])
    A = np.array([La[:, edges[j]:edges[j + 1]].sum() for j in range(a.bands)])
    total_b, total_a = B.sum(), A.sum()
    zones = {k: {"before": float(Lb[r0:r1].sum()), "after": float(La[r0:r1].sum())} for k, (r0, r1) in ZONES.items()}
    for z in zones.values():
        z["remaining"] = z["after"] / z["before"] if z["before"] > 0 else None

    # timing of the two wipes from the line track
    valid = np.isfinite(lx)
    runs, cur = [], None
    for i, v in enumerate(valid):
        if v and cur is None:
            cur = i
        if not v and cur is not None:
            runs.append((cur, i)); cur = None
    if cur is not None:
        runs.append((cur, len(valid)))
    runs = [r for r in runs if r[1] - r[0] > fps * 0.5]
    out_run, back_run = runs[0], runs[-1]

    def crossing_time(x, run, direction):
        ii = np.arange(run[0], run[1])
        xx = lx[ii]
        hit = np.where(xx >= x)[0] if direction > 0 else np.where(xx <= x)[0]
        return float(t_fr[ii[hit[0]]]) if len(hit) else None

    # notes
    lb = np.log1p(B / (B[B > 0].mean() if (B > 0).any() else 1) * 10)
    la = np.log1p(A / (B[B > 0].mean() if (B > 0).any() else 1) * 10)
    pm = PitchMap(0, float(lb.max()), low=55, span=24, root=7, scale="major_pent")
    lit = B > 0.01 * B.max()
    notes, gestures = [], []
    for direction, run in ((1, out_run), (-1, back_run)):
        for j in range(a.bands):
            if not lit[j]:
                continue
            xc = (edges[j] + edges[j + 1]) / 2
            t0 = crossing_time(xc, run, direction)
            if t0 is None:
                continue
            p_from, p_to = (lb[j], la[j]) if direction > 0 else (la[j], lb[j])
            P0, P1 = int(pm(p_from)[0]), int(pm(p_to)[0])
            vel = int(40 + 80 * np.sqrt(B[j] / B.max()))
            pan = float(2 * xc / W - 1)
            src_info = {"band": j, "x": [int(edges[j]), int(edges[j + 1])], "before": float(B[j]), "after": float(A[j]),
                        "remaining": float(A[j] / B[j]), "wipe": "out" if direction > 0 else "back"}
            seq = [P0] if P0 == P1 else [P0, P1]
            gl = []
            for i, p in enumerate(seq):
                ts = t0 + i * 0.12
                n = Note(ts, ts + (0.12 if i < len(seq) - 1 else 0.2), p, vel if i == 0 else int(vel * 0.8), "city",
                         float(p_from if i == 0 else p_to), pan, dict(src_info, step=i))
                notes.append(n); gl.append(n)
            gestures.append((gl, float(np.log(max(A[j], 1e-9) / B[j])) * direction))

    # on-screen light over time (before right of the line, after left of it)
    dtc = 0.05
    t_ctrl = np.arange(0, duration, dtc)
    xline = np.interp(t_ctrl, t_fr[valid], lx[valid]) if valid.any() else np.zeros_like(t_ctrl)
    # outside the wipes the line is off-screen: before (start/end) or after (hold) everywhere
    state_after = np.zeros_like(t_ctrl)
    tb0, tb1 = t_fr[out_run[0]], t_fr[out_run[1] - 1]
    tc0, tc1 = t_fr[back_run[0]], t_fr[back_run[1] - 1]
    colx = (edges[:-1] + edges[1:]) / 2
    shown = np.zeros_like(t_ctrl)
    for i, t in enumerate(t_ctrl):
        if t < tb0 or t > tc1:
            frac_after = np.zeros(a.bands, bool)
        elif tb1 < t < tc0:
            frac_after = np.ones(a.bands, bool)
        else:
            frac_after = colx < xline[i]
        shown[i] = (np.where(frac_after, A, B)).sum() / total_b
        state_after[i] = frac_after.mean()
    anth = np.clip(shown, 0, 1)
    rel = (anth - anth.min()) / max(1e-9, anth.max() - anth.min())
    curves = {"anthrophony": anth, "biophony": np.clip(0.15 + 0.75 * (1 - rel), 0, 1),
              "geophony": np.clip(0.1 + 0.5 * state_after, 0, 1), "brightness": anth}
    # drone: a sustained pad whose register follows the light on screen (one note per 0.5 s)
    dpm = PitchMap(float(total_a / total_b), 1.0, low=43, span=12, root=7, scale="major_pent")
    drone = []
    for t in np.arange(0, duration, 0.5):
        v = float(np.interp(t + 0.25, t_ctrl, anth))
        p = int(dpm(v)[0])
        if drone and drone[-1].pitch == p:
            drone[-1].end = t + 0.5
            continue
        drone.append(Note(float(t), float(t + 0.5), p, int(40 + 50 * v), "drone", v, 0.0, {"metric": "light on screen / before"}))
    notes += drone

    city = render_notes([n for n in notes if n.voice == "city"], duration, "pluck", gain=0.3)
    pad = render_notes(drone, duration, "pad", gain=0.12)
    stems = [Stem("city", "melody", city, 1.0), Stem("drone", "bands", pad, 0.6),
             Stem("traffic", "anthrophony", render_curve_voice("anthrophony", t_ctrl, curves["anthrophony"], duration, seed=31), 0.45),
             Stem("frogs-insects", "biophony", render_curve_voice("biophony", t_ctrl, curves["biophony"], duration, seed=32), 0.35),
             Stem("wind-rain", "geophony", render_curve_voice("geophony", t_ctrl, curves["geophony"], duration, seed=33), 0.25)]

    # proof: run interval recovered from audio vs log(after/before)
    cn = [n for n in notes if n.voice == "city"]
    # search only the PitchMap's range (+-1 semitone): rules out pyin's sub-octave errors
    mp = note_pitches(city, cn, fmin=float(midi_hz(pm.low - 1)), fmax=float(midi_hz(pm.low + pm.span + 1)), win=0.11,
                     frame_length=2048, hop=256)
    mp_of = {id(n): m for n, m in zip(cn, mp)}
    rec, tru = [], []
    for gl, x in gestures:
        p0, p1 = mp_of[id(gl[0])], mp_of[id(gl[-1])]
        if np.isfinite(p0) and np.isfinite(p1):
            rec.append(p1 - p0); tru.append(x)
    rec, tru = np.array(rec), np.array(tru)
    fidelity = {"city_runs": {"interval_vs_log_light_ratio_r": float(np.corrcoef(rec, tru)[0, 1]) if len(rec) > 2 and rec.std() > 0 else None,
                              "falls_on_wipe_out": float(np.mean(rec[tru < 0] <= 0)) if (tru < 0).any() else None, "n": int(len(rec)),
                              "stem": "city"},
                "line_tracked_frames": int(valid.sum())}
    lost = 100 * (1 - total_a / total_b)
    worst = min(zones, key=lambda k: zones[k]["remaining"] if zones[k]["remaining"] is not None else 9)
    title = page.get("title")
    desc = (f"Night lights of southwest Florida on August 30, 2022 and on September 30, after Hurricane Ian. "
            f"A slider line wipes the 'after' picture across the map; as it passes, each strip of the coast plays a "
            f"falling run as its lights go out. About {lost:.0f} percent of the light on the map is gone, most around "
            f"{worst}, so the city hum fades and frogs and insects take over until the line slides back.")
    legend = [
        {"voice": "pluck runs (city)", "meaning": "each strip of the map as the line crosses it: falling = lights lost (bigger fall = more lost), rising when the before picture returns; pan = position"},
        {"voice": "pad (drone)", "meaning": "total light on screen; lower = darker"},
        {"voice": "traffic hum (anthrophony)", "meaning": "light currently on screen relative to before the storm"},
        {"voice": "frogs and insects (biophony)", "meaning": "louder as the city goes dark"},
        {"voice": "wind and rain (geophony)", "meaning": "share of the screen showing the after-storm picture"},
    ]
    report = {"id": PAGE, "title": title, "movie": MOVIE, "movie_info": info,
              "frames": {"before": BEFORE_K, "after": AFTER_K}, "colorbar_bbox_in_crop": bb, "colorbar_region": CBAR_REGION,
              "wipes_s": {"out": [float(tb0), float(tb1)], "back": [float(tc0), float(tc1)]},
              "light_before": float(total_b), "light_after": float(total_a), "percent_lost": float(lost),
              "zones": zones, "bands": [{"x": [int(edges[j]), int(edges[j + 1])], "before": float(B[j]), "after": float(A[j])} for j in range(a.bands)],
              "after_light_outside_before_mask_px": track_px,
              "note": "relative brightness only (no numeric scale); after-light counted only within 3 px of before-light (removes the hurricane track and labels)",
              "notes": {"city": len(cn), "drone": len(drone)}, "fidelity": fidelity}
    to_midi(notes, out / "score.mid", programs={"city": 11, "drone": 89})
    write_track(out, page_id=PAGE, title=title, video_url=MOVIE, duration_s=duration,
                units="relative night light (less .. more), no numeric scale",
                description=desc, legend=legend, stems=stems, notes=notes, t_ctrl=t_ctrl, curves=curves,
                fidelity=fidelity, report=report,
                data_source={"kind": "pixels", "movie": MOVIE, "underlying": "NASA Black Marble (VIIRS) nightly radiance, 2022-08-30 vs 2022-09-30",
                             "scale": "relative: sequential colormap without numbers, inverted from the burned-in colorbar"})
    plot(out, before, after, Lb, La, edges, B, A, t_fr, lx, t_ctrl, curves)
    export_demo(out, a.demo / str(PAGE), ["mix"])
    print(json.dumps({k: report[k] for k in ("wipes_s", "percent_lost", "zones", "fidelity", "notes")}, indent=1))


def plot(out, before, after, Lb, La, edges, B, A, t_fr, lx, t_ctrl, curves):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    sm = lambda x: cv2.resize(x, (960, 540), interpolation=cv2.INTER_AREA)
    ax[0, 0].imshow(sm(before)); ax[0, 0].set_title("before (Aug 30)"); ax[0, 0].axis("off")
    ax[0, 1].imshow(sm(after)); ax[0, 1].set_title("after (Sep 30)"); ax[0, 1].axis("off")
    xc = (edges[:-1] + edges[1:]) / 2
    ax[1, 0].bar(xc, B, width=np.diff(edges) * 0.9, color="#f4a340", label="before")
    ax[1, 0].bar(xc, A, width=np.diff(edges) * 0.5, color="#6a3d9a", label="after")
    ax[1, 0].legend(); ax[1, 0].set_title("light per column band (inverted colorbar)"); ax[1, 0].set_xlabel("x px")
    ax[1, 1].plot(t_fr, lx / 3840, ".", ms=2, label="slider line x / width")
    for k, c in curves.items():
        ax[1, 1].plot(t_ctrl, c, label=k)
    ax[1, 1].legend(fontsize=7); ax[1, 1].set_xlabel("s"); ax[1, 1].set_title("line track and controls")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=95); plt.close(fig)


if __name__ == "__main__":
    main()
