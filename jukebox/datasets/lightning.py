"""SVS 5011 Lightning from the ISS (LIS) 2017-2023 -> percussion, thunder and a latitude line.

    python -m jukebox.datasets.lightning            # detect (cached) + score + render
    python -m jukebox.datasets.lightning --redetect

Source: the Science-On-a-Sphere cut (4096x2048, exact equirectangular, 60 fps, 100 s), so
pixel -> (lon, lat) is exact. Each LIS flash is drawn as a yellow dot that appears in one
frame and stays ~0.9 s (the roving window). A *flash onset* is a pixel whose yellow brightness
jumps by more than `--jump` between consecutive decoded frames; 8-connected onset pixels form
one event (t, lon, lat, area, intensity). Neighbouring dots that appear in the same frame merge.

Music
- events: tom hits quantized to a 16th grid at 120 BPM. Hits per beat follow the beat's flash
  intensity (quiet beats rest, the busiest get `--cap-beat`); they go to the strongest
  (16th step, 60 deg sector) cells, at most `--cap` per step. Pan = longitude, GM tom key / pitch = latitude (north = higher),
  velocity = summed onset intensity. All rate-limits are reported in report.json.
- melody: one note per beat; pitch = intensity-weighted mean latitude of new flashes
  (the seasonal north-south swing of the tropical storm belt), loudness = flash density.
- geophony (thunder) control curve = smoothed flash density.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from ..data import catalog
from ..proof.fidelity import recover_curve, score
from ..render.synth import SR, render_notes
from ..score.mapping import Note, PitchMap, melody_from_curve, to_midi
from .events_common import (Stem, export_demo, lon_pan, norm01, render_curve_voice, render_toms, smooth,
                            video_info, write_track)

PAGE = 5011
SPOT_TIMES = (20.0, 50.0, 80.0)
MOVIE = "https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005011/iss_lightning_sos_yearly_07312023_2048p60.mp4"
DATA_START = dt.date(2017, 1, 1)
DATA_END = dt.date(2023, 7, 31)   # "07312023" in the file name; page: Jan 2017 - Jul 2023
TOMS = [(-90, 41), (-25, 43), (-10, 45), (5, 47), (20, 48), (35, 50)]  # lat lower edge -> GM tom key
TOM_NAMES = {41: "low floor tom", 43: "high floor tom", 45: "low tom", 47: "low-mid tom", 48: "hi-mid tom", 50: "high tom"}


# ---------------------------------------------------------------- detection
def yellow_value(frame: np.ndarray) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    yel = (hsv[..., 0] >= 14) & (hsv[..., 0] <= 36) & (hsv[..., 1] > 80)
    return np.where(yel, hsv[..., 2], 0).astype(np.int16)


def detect(src: Path, fps: float, width: int, jump: int, min_px: int, keep=()):
    """Stream the movie; return events array (N, 6): t, lon, lat, area, intensity, frame and per-frame stats."""
    import cv2
    from .events_common import decode_stream
    W, H = width, width // 2
    prev = prev_f = None
    ev, visible, onset_px, kept = [], [], [], {}
    for k, f in enumerate(decode_stream(src, fps, W, H)):
        if k in keep and prev_f is not None:
            kept[k] = (prev_f.copy(), f.copy())
        prev_f = f
        yv = yellow_value(f)
        visible.append(int((yv > 0).sum()))
        if prev is None:
            prev = yv
            onset_px.append(0)
            continue
        inc = yv - prev
        mask = ((inc > jump) & (yv > 0)).astype(np.uint8)
        onset_px.append(int(mask.sum()))
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
        t = k / fps
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if a < min_px:
                continue
            x0, y0, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
            patch = yv[y0:y0 + h, x0:x0 + w][lab[y0:y0 + h, x0:x0 + w] == i]
            cx, cy = cen[i]
            lon = -180 + (cx + 0.5) / W * 360
            lat = 90 - (cy + 0.5) / H * 180
            ev.append((t, lon, lat, a, float(patch.mean()) / 255 * a, k))
        prev = yv
    return np.array(ev, float).reshape(-1, 6), np.array(visible), np.array(onset_px), kept


# ---------------------------------------------------------------- date axis
NH_PEAK_DAY = (8, 1)  # assumed calendar date of the yearly maximum of the northern-hemisphere flash share


def fit_calendar(tb, nh_share, t_first, t_last):
    """Movie time -> date from the annual north/south swing of lightning.

    The share of flashes in the northern hemisphere peaks once per northern summer. Peaks of the
    (smoothed) share give seconds-per-year by linear regression; the first peak is taken as summer
    2017 (data start Jan 2017) and dated NH_PEAK_DAY. Dates are therefore approximate (~1 month).
    """
    x = np.where(np.isfinite(nh_share), nh_share, np.nan)
    xs = smooth(np.nan_to_num(x, nan=np.nanmean(x)), 3)
    ok = (tb > t_first + 2) & (tb < t_last - 8)  # a peak at the very end is truncated by the data end
    cand = [i for i in range(1, len(xs) - 1) if ok[i] and xs[i] >= xs[i - 1] and xs[i] >= xs[i + 1] and xs[i] > 0.7]
    cand.sort(key=lambda i: -xs[i])
    peaks = []
    for i in cand:
        if all(abs(tb[i] - tb[j]) >= 7 for j in peaks):
            peaks.append(i)
    peaks = sorted(peaks)
    tp = tb[peaks]
    k = np.arange(len(tp))
    P, t0 = np.polyfit(k, tp, 1)
    resid = tp - (t0 + P * k)
    ref = dt.date(DATA_START.year, *NH_PEAK_DAY)
    cal = {"method": "NH-share peaks, linear in time; first peak = summer 2017 dated %02d-%02d" % NH_PEAK_DAY,
           "sec_per_year": float(P), "t_ref": float(t0), "ref_date": ref.isoformat(),
           "nh_peak_times": [float(v) for v in tp], "peak_fit_rms_s": float(np.sqrt((resid ** 2).mean())),
           "t_first_active": float(t_first), "t_last_active": float(t_last)}
    cal["date_first_active"] = date_at(t_first, cal, clamp=False).isoformat()
    cal["date_last_active"] = date_at(t_last, cal, clamp=False).isoformat()
    t_end = t0 + ((DATA_END - ref).days / 365.25) * P
    cal["t_of_page_end_date"] = float(t_end)
    cal["unexplained_tail_s"] = float(max(0.0, t_last - t_end))
    cal["note"] = ("peaks 2017-2022 are evenly spaced; flashes keep appearing for a few seconds after the time "
                   "that maps to 2023-07-31, so dates there are clamped to the page's end date")
    cal["page_range"] = [DATA_START.isoformat(), DATA_END.isoformat()]
    return cal


def date_at(t, cal, clamp=True) -> dt.date:
    days = (float(t) - cal["t_ref"]) / cal["sec_per_year"] * 365.25
    d = dt.date.fromisoformat(cal["ref_date"]) + dt.timedelta(days=days)
    return min(max(d, DATA_START), DATA_END) if clamp else d


# ---------------------------------------------------------------- score
def tom_key(lat: float) -> int:
    key = TOMS[0][1]
    for edge, k in TOMS:
        if lat >= edge:
            key = k
    return key


def hits_from_events(ev, duration, bpm, cap, cap_beat, sectors, cal, gamma=1.5):
    """Quantize flash onsets to a 16th-note grid and thin them so the density stays musical.

    Per beat: n_hits = round(cap_beat * clip((I_beat - p10) / (p98 - p10), 0, 1) ** gamma), where I is the
    summed onset intensity, so quiet stretches rest and the busiest storms fill the bar. The n_hits go
    to the strongest (16th step, longitude sector) cells of that beat, at most `cap` per step.
    """
    step = 60 / bpm / 4
    nsteps = int(np.ceil(duration / step))
    nbeats = (nsteps + 3) // 4
    si = np.minimum((ev[:, 0] / step).astype(int), nsteps - 1)
    sec = np.clip(((ev[:, 1] + 180) / 360 * sectors).astype(int), 0, sectors - 1)
    inten = ev[:, 4]
    cell = np.zeros((nsteps, sectors))
    np.add.at(cell, (si, sec), inten)
    beat_tot = np.add.reduceat(cell.sum(1), np.arange(0, nsteps, 4))[:nbeats]
    act = beat_tot[beat_tot > 0]
    lo, hi = (np.percentile(act, 10), np.percentile(act, 98)) if len(act) else (0, 1)
    vel_ref = np.log1p(np.percentile(cell[cell > 0], 99))
    order = np.lexsort((sec, si))
    bounds = np.searchsorted(si[order] * sectors + sec[order], np.arange(nsteps * sectors + 1))
    notes, n_target = [], []
    for b in range(nbeats):
        n_b = int(round(cap_beat * np.clip((beat_tot[b] - lo) / (hi - lo + 1e-9), 0, 1) ** gamma))
        n_target.append(n_b)
        if n_b == 0:
            continue
        s0, s1 = 4 * b, min(nsteps, 4 * b + 4)
        cells = [(cell[s, c], s, c) for s in range(s0, s1) for c in range(sectors) if cell[s, c] > 0]
        cells.sort(reverse=True)
        per_step, placed = {}, 0
        for val, s, c in cells:
            if placed >= n_b:
                break
            if per_step.get(s, 0) >= cap:
                continue
            k = s * sectors + c
            m = order[bounds[k]:bounds[k + 1]]
            w = inten[m]
            lon = float(np.average(ev[m, 1], weights=w))
            lat = float(np.average(ev[m, 2], weights=w))
            vel = int(np.clip(35 + 92 * np.log1p(val) / vel_ref, 30, 127))
            t0 = s * step
            notes.append(Note(t0, t0 + step, tom_key(lat), vel, "flashes", float(val), float(lon_pan(lon)),
                              {"lon": round(lon, 2), "lat": round(lat, 2), "n_flashes": int(len(m)),
                               "date": date_at(t0, cal).isoformat()}))
            per_step[s] = per_step.get(s, 0) + 1
            placed += 1
    notes.sort(key=lambda n: n.start)
    return notes, {"bpm": bpm, "grid": "1/16 note", "step_s": step, "cap_per_step": cap, "cap_per_beat": cap_beat,
                   "lon_sectors": sectors, "hits_per_beat_rule": f"round(cap_beat * clip((I - p10)/(p98 - p10), 0, 1)^{gamma})",
                   "beats_silent": int(sum(1 for x in n_target if x == 0)), "beats": int(nbeats),
                   "cells_with_flashes": int((cell > 0).sum()), "hits": len(notes),
                   "note": "a hit = all onsets in one (16th step, 60 deg sector) cell; weaker cells are dropped"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--jump", type=int, default=35)
    ap.add_argument("--min-px", type=int, default=1)
    ap.add_argument("--bpm", type=float, default=120)
    ap.add_argument("--cap", type=int, default=2, help="max hits per 16th step")
    ap.add_argument("--cap-beat", type=int, default=6, help="max hits per beat")
    ap.add_argument("--sectors", type=int, default=6, help="longitude sectors for hit grouping/panning")
    ap.add_argument("--redetect", action="store_true")
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
    page = catalog.fetch(PAGE, a.data.parent / "catalog")

    cache = d / f"events_{a.width}_{a.fps:g}_{a.jump}.npz"
    if a.redetect or not cache.exists():
        keep = [int(round(t * a.fps)) for t in SPOT_TIMES]
        ev, visible, onset_px, kept = detect(src, a.fps, a.width, a.jump, a.min_px, keep)
        np.savez_compressed(cache, ev=ev, visible=visible, onset_px=onset_px, spot_k=np.array(sorted(kept)),
                            spot=np.stack([np.stack(kept[k]) for k in sorted(kept)]))
    z = np.load(cache)
    ev, visible = z["ev"], z["visible"]
    duration = info["duration"]
    tf = np.arange(len(visible)) / a.fps

    # flash density per 0.1 s
    dt_c = 0.1
    t_ctrl = np.arange(0, duration, dt_c)
    rate = np.bincount(np.minimum((ev[:, 0] / dt_c).astype(int), len(t_ctrl) - 1), minlength=len(t_ctrl)) / dt_c
    rate_s = smooth(rate, 9)
    thunder = norm01(rate_s, 1, 99)

    # calendar from the seasonal NH share (1 s bins), anchored at the end of visible data
    tb = np.arange(0, duration, 1.0)
    bi = np.minimum(ev[:, 0].astype(int), len(tb) - 1)
    nh = np.bincount(bi, weights=(ev[:, 2] > 0) * ev[:, 4], minlength=len(tb))
    al = np.bincount(bi, weights=ev[:, 4], minlength=len(tb))
    nh_share = np.where(al > al.max() * 0.15, nh / np.maximum(al, 1e-9), np.nan)
    active = np.where(rate_s > 0.15 * np.nanmax(rate_s))[0]
    t_first, t_last = float(t_ctrl[active[0]]), float(t_ctrl[active[-1]])
    cal = fit_calendar(tb + 0.5, nh_share, t_first, t_last)

    # melody: weighted mean latitude per beat (2 s window), pitch over the observed range
    beat = 60 / a.bpm
    tm = np.arange(0, duration, beat)
    lat_c = np.full(len(tm), np.nan)
    for i, t0 in enumerate(tm):
        m = (ev[:, 0] >= t0 - 1) & (ev[:, 0] < t0 + 1)
        if ev[m].shape[0] >= 30:
            lat_c[i] = np.average(ev[m, 2], weights=ev[m, 4])
    lo, hi = np.nanpercentile(lat_c, 1), np.nanpercentile(lat_c, 99)
    pad = 0.05 * (hi - lo)
    lat_pm = PitchMap(lo - pad, hi + pad, low=57, span=24, root=9, scale="minor_pent")  # A minor pentatonic
    dens_beat = np.interp(tm, t_ctrl, thunder)
    melody = melody_from_curve(tm, lat_c, lat_pm, "storm-belt latitude", step=None, loudness=dens_beat, vel_range=(45, 100))
    for n in melody:
        n.source = {"metric": "flash-weighted mean latitude (deg)",
                    "date": date_at(n.start, cal).isoformat()}
    hits, rl = hits_from_events(ev, duration, a.bpm, a.cap, a.cap_beat, a.sectors, cal)
    rl["events_detected"] = int(len(ev))

    # render
    mel_audio = render_notes(melody, duration, "bell", gain=0.22)
    hit_audio = render_toms(hits, duration)
    geo = render_curve_voice("geophony", t_ctrl, thunder, duration, seed=11)
    curves = {"geophony": thunder, "anthrophony": np.zeros_like(thunder),
              "biophony": np.full_like(thunder, 0.15), "brightness": norm01(np.interp(t_ctrl, tf, visible), 1, 99)}
    bio = render_curve_voice("biophony", t_ctrl, curves["biophony"], duration, seed=12)

    # proof: latitude line recovered from the rendered melody stem; time-shuffled control
    fm = np.isfinite(lat_c)
    est = recover_curve(mel_audio, SR, lat_pm, tm)
    real = score(lat_c, est)
    rng = np.random.default_rng(0)
    sh = lat_c.copy()
    sh[fm] = lat_c[fm][rng.permutation(fm.sum())]
    ctrl_notes = melody_from_curve(tm, sh, lat_pm, "ctrl", step=None)
    ctrl = score(lat_c, recover_curve(render_notes(ctrl_notes, duration, "bell", gain=0.22), SR, lat_pm, tm))
    # proof for hits: hit-count envelope vs flash density
    hit_rate = np.bincount(np.minimum((np.array([n.start for n in hits]) / dt_c).astype(int), len(t_ctrl) - 1),
                           minlength=len(t_ctrl))
    env_r = float(np.corrcoef(smooth(hit_rate, 21), smooth(rate, 21))[0, 1])
    pan_lon_r = float(np.corrcoef([n.pan for n in hits], [n.source["lon"] for n in hits])[0, 1])

    fidelity = {"melody_latitude": {"real": real, "shuffled_control": ctrl, "stem": "latitude"},
                "hits_vs_flash_density_r": env_r, "pan_vs_longitude_r": pan_lon_r}
    title = page.get("title", "Lightning from ISS")
    nfl = int(len(ev))
    description = (
        f"Lightning seen by the ISS Lightning Imaging Sensor from {DATA_START:%B %Y} to {DATA_END:%B %Y}, drawn as yellow dots "
        f"on a flat world map; we detected about {nfl / 1e6:.1f} million new dots. Each drum hit is a cluster of new flashes: it is panned to where "
        "the storms are (west left, east right) and pitched by latitude, higher drums for northern storms. "
        "The bell line follows the average latitude of the storms, rising every northern summer and falling every "
        "southern summer, and the thunder rumble swells with the number of flashes.")
    legend = [
        {"voice": "toms (events)", "meaning": "new lightning flashes; pan = longitude, drum pitch = latitude (higher = further north), loudness = flash brightness x count"},
        {"voice": "bell (melody)", "meaning": "average latitude of new flashes; higher = storm belt further north"},
        {"voice": "thunder (geophony)", "meaning": "number of flashes per second"},
        {"voice": "night insects (biophony)", "meaning": "constant background bed, not data-driven in this track"},
    ]
    stems = [Stem("flashes", "events", hit_audio, 0.7), Stem("latitude", "melody", mel_audio, 1.3),
             Stem("thunder", "geophony", geo, 0.6), Stem("insects", "biophony", bio, 0.25)]
    notes = melody + hits
    to_midi(notes, out / "score.mid", programs={"storm-belt latitude": 14}, drums={"flashes"}, tempo=a.bpm)

    # spot-check numbers (filled by spot_check())
    spot = spot_check(out, a, ev, z["spot_k"], z["spot"])
    report = {
        "id": PAGE, "title": title, "movie": MOVIE, "movie_info": info,
        "detector": {"method": "yellow-brightness jump between consecutive decoded frames, 8-connected components",
                     "decode_fps": a.fps, "decode_width": a.width, "jump_threshold_V": a.jump, "min_px": a.min_px,
                     "events_detected": nfl,
                     "events_per_second_mean": float(nfl / duration),
                     "first_last_active_s": [t_first, t_last],
                     "note": "dots that appear in the same frame and touch merge into one event; counts are lower bounds"},
        "calendar": cal, "rate_limit": rl, "hits": len(hits), "melody_notes": len(melody),
        "latitude_pitchmap": lat_pm.describe(), "spot_check": spot, "fidelity": fidelity,
        "tom_map": {f">= {e} deg": TOM_NAMES[k] for e, k in TOMS},
    }
    track = write_track(
        out, page_id=PAGE, title=title, video_url=MOVIE, duration_s=duration, units="flashes (counts); latitude in degrees",
        description=description, legend=legend, stems=stems, notes=notes, t_ctrl=t_ctrl, curves=curves,
        fidelity=fidelity, report=report,
        data_source={"kind": "pixels", "movie": MOVIE, "underlying": "ISS LIS quality-controlled science data V1 (GHRC DAAC)",
                     "projection": "equirectangular, lon -180..180, lat 90..-90 (Science On a Sphere)",
                     "calendar": "movie time -> date by fitting the annual NH/SH flash cycle; see report.calendar"},
        extra={"calendar": {k: cal[k] for k in ("t_ref", "ref_date", "sec_per_year")} |
               {"clamp": [DATA_START.isoformat(), DATA_END.isoformat()],
                "date_of_t": "ref_date + (t - t_ref) / sec_per_year years, clamped"}})
    plot(out, ev, t_ctrl, rate_s, tm, lat_c, est, hits, cal, nh_share, tb)
    export_demo(out, a.demo / str(PAGE), ["mix"])
    print(json.dumps({k: report[k] for k in ("detector", "calendar", "rate_limit", "hits", "fidelity", "spot_check")}, indent=1))
    return track


def spot_check(out, a, ev, spot_k, spot):
    """Overlay detected onsets on a few frames and compare with an independent new-blob count."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W, H = a.width, a.width // 2
    res = []
    fig, axs = plt.subplots(len(spot_k), 2, figsize=(14, 4.2 * len(spot_k)))
    for r, (k, (p, c)) in enumerate(zip(spot_k, spot)):
        tt = k / a.fps
        e = ev[ev[:, 5] == k]
        # independent count: new yellow blobs = current yellow minus dilated previous yellow (misses dots
        # that land on existing dots, so it is a lower bound; the detector also catches those)
        yp, yc = yellow_value(p) > 0, yellow_value(c) > 0
        new = yc & ~cv2.dilate(yp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        nb = cv2.connectedComponents(new.astype(np.uint8), connectivity=8)[0] - 1
        res.append({"t": float(tt), "detected": int(len(e)), "new_isolated_blobs": int(nb),
                    "visible_yellow_px": int(yc.sum())})
        # crop: central Africa (0-22E, 5S-11N), small enough to count dots by eye
        x0, x1, y0, y1 = int(W * 0.50), int(W * 0.56), int(H * 0.44), int(H * 0.53)
        axs[r, 0].imshow(np.clip(p[y0:y1, x0:x1].astype(int) * 2, 0, 255).astype(np.uint8)); axs[r, 0].set_title(f"t={tt:.2f}s previous frame")
        axs[r, 1].imshow(np.clip(c[y0:y1, x0:x1].astype(int) * 2, 0, 255).astype(np.uint8), interpolation="nearest")
        X = (e[:, 1] + 180) / 360 * W - x0
        Y = (90 - e[:, 2]) / 180 * H - y0
        s = (X > 0) & (X < x1 - x0) & (Y > 0) & (Y < y1 - y0)
        axs[r, 1].scatter(X[s], Y[s], s=40, facecolors="none", edgecolors="c", lw=0.8)
        axs[r, 1].set_title(f"next frame + {int(s.sum())} detected onsets in crop (cyan)")
        res[-1]["detected_in_crop"] = int(s.sum())
        for ax in axs[r]:
            ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "spot_check.png", dpi=90); plt.close(fig)
    return res


def plot(out, ev, t_ctrl, rate_s, tm, lat_c, est, hits, cal, nh_share, tb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    sub = ev[:: max(1, len(ev) // 60000)]
    ax[0, 0].scatter(sub[:, 1], sub[:, 2], s=0.2, c=sub[:, 0], cmap="viridis", alpha=0.4)
    ax[0, 0].set_title(f"{len(ev):,} detected flash onsets (colour = movie time)"); ax[0, 0].set_xlabel("lon"); ax[0, 0].set_ylabel("lat")
    ax[0, 1].plot(t_ctrl, rate_s, lw=0.8, label="flash onsets / s (smoothed)")
    hr = np.histogram([n.start for n in hits], bins=np.arange(0, t_ctrl[-1] + 1, 1.0))[0]
    a2 = ax[0, 1].twinx(); a2.plot(np.arange(len(hr)) + 0.5, hr, "r", lw=0.8, label="tom hits / s")
    ax[0, 1].set_title("thunder curve and rate-limited hits"); ax[0, 1].set_xlabel("movie s")
    ax[1, 0].plot(tb + 0.5, nh_share, ".", ms=3, label="NH share of flashes")
    for p in cal["nh_peak_times"]:
        ax[1, 0].axvline(p, color="gray", lw=0.5); ax[1, 0].text(p, 0.03, date_at(p, cal).isoformat()[:7], fontsize=7, rotation=90)
    ax[1, 0].set_title(f"calendar: {cal['sec_per_year']:.2f} s / year (peak rms {cal['peak_fit_rms_s']:.2f} s); "
                       f"data {cal['date_first_active'][:7]}..{cal['date_last_active'][:7]}", fontsize=9); ax[1, 0].set_ylim(0, 1)
    ax[1, 1].plot(tm, lat_c, label="flash-weighted mean latitude"); ax[1, 1].plot(tm, est, ".", ms=3, label="recovered from audio")
    ax[1, 1].set_title("melody round trip"); ax[1, 1].legend(); ax[1, 1].set_xlabel("movie s")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
