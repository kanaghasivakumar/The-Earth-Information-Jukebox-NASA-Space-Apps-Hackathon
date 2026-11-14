"""SVS 5313 Change in Night Lights 2012-2023 (EIC version) -> the headline three-voice story.

    python -m jukebox.datasets.nightlights

The movie (7680x2160, 30 fps, 43.3 s) is two identical world maps side by side; we read the
left one (georeferenced against the 5011 equirectangular frame, see georef.py). Per the SVS
page description it runs four phases, which we detect from the pixels:

  A  annual averages 2012..2023 (year label changes every ~0.6 s)
  B  lights fade, then the 2012->2023 change map (purple = more light, orange = less)
  C  the annual series again, with pop-out zooms of Western Europe, Eastern Mediterranean,
     Ukraine and India
  D  the change map again, with the same zooms

The change colormap has no numeric scale ("lights out" .. "no change" .. "new lights"), so all
values are *relative*: we invert the burned-in colorbar to -1 .. +1 and weight each pixel by how
much light it adds over the faded base map. The zoom pop-outs occlude parts of the map in C/D,
so every number is measured on the clean A/B frames and *replayed* in sync with C/D.

Music (synced to the movie)
- years (melody): one note per year = global light (sum over the map, relative to the base map)
- sectors (bands): per year a west->east arpeggio over six 60 deg longitude sectors; pitch = that
  sector's light relative to 2012 (up = brighter), pan = longitude. In C: the four zoom regions.
- change-sweep (melody): in B a west->east sweep of twelve 30 deg sectors; each plays a run
  that rises for net brightening and falls for net dimming (interval ~ strength), pan = longitude.
  In D the four zoom regions play their runs.
- year-ticks (events): a tick at every detected year boundary.
- controls: anthrophony rises with light / brightening, biophony falls; geophony is a quiet bed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import catalog
from ..data.colorbar import Colorbar
from ..proof.fidelity import score
from ..render.synth import SR, render_notes
from ..score.mapping import Note, PitchMap, to_midi
from .events_common import (Stem, decode_stream, export_demo, lon_pan, midi_to_value, norm01, note_pitches,
                            render_curve_voice, smooth, video_info, write_track)
from .georef import GeoFit

PAGE = 5313
MOVIE = "https://svs.gsfc.nasa.gov/vis/a000000/a005300/a005313/nightlights_flat_series_and_change_wZooms_13_EIC_2160p30_h265.mp4"
YEARS = list(range(2012, 2024))
FULL_W, FULL_H = 7680, 2160
# Fitted by `python -m jukebox.datasets.georef --target <frame 5> --crop 0,0,3840,2160` (262 coastline
# matches, rms 12 px at full res). Left map, full-res pixels.
GEOREF = {"x0": 1796.9693515159172, "kx": 10.66857909002766,
          "ycoef": [-0.0001467748709813247, 0.005790735285235269, -11.907785973999799, 1082.422342951537],
          "w": 3840, "h": 2160}
# Burned-in colorbar strip (full-res), "lights out" (orange) .. "no change" (white) .. "new lights" (purple)
CBAR_BOX = (3764, 1225, 4370, 1240)
# areas of the left map to ignore (colorbar / year label overlay), full-res x0, y0, x1, y1
OVERLAY_MASK = [(3600, 950, 3840, 1300)]
LABEL_BOX = (3540, 1020, 4140, 1140)

ZOOMS = {  # pop-out regions named on the movie (lon0, lon1, lat0, lat1)
    "Western Europe": (-10, 16, 36, 56),
    "Eastern Mediterranean": (25, 37, 27, 37.5),
    "Ukraine": (22, 40.5, 44, 52.5),
    "India": (68, 90, 6, 33),
}
CONTINENTS = {  # first match wins
    "Middle East": (34, 63, 12, 42), "Europe": (-12, 40, 36, 72), "North Africa": (-18, 34, 12, 36),
    "Sub-Saharan Africa": (-20, 52, -36, 12), "South Asia": (60, 92, 5, 36), "East Asia": (100, 146, 20, 54),
    "Southeast Asia & Oceania": (92, 180, -48, 20), "Russia & Central Asia": (40, 180, 36, 75),
    "North America": (-170, -50, 15, 75), "Latin America": (-120, -30, -58, 15),
}


# ---------------------------------------------------------------- pass 1: segmentation stats
def pass1(src: Path, width=1920):
    import cv2
    s = width / FULL_W
    lx0, ly0, lx1, ly1 = [int(v * s) for v in LABEL_BOX]
    stats, small, labels = [], [], []
    for f in decode_stream(src, None, width, int(FULL_H * s)):
        left = f[:, : width // 2]
        hsv = cv2.cvtColor(left, cv2.COLOR_RGB2HSV)
        sv = (hsv[..., 1] > 60) & (hsv[..., 2] > 60)
        pur = ((hsv[..., 0] > 115) & (hsv[..., 0] < 165) & sv).mean()
        lum = left.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        stats.append((float(lum.mean()), float(pur), float((lum > 60).mean())))
        labels.append(cv2.cvtColor(f[ly0:ly1, lx0:lx1], cv2.COLOR_RGB2GRAY).astype(np.int16))
        small.append(cv2.resize(left, (480, 270), interpolation=cv2.INTER_AREA))
    return np.array(stats), np.stack(labels), np.stack(small)


def runs(mask):
    """[start, end) index pairs of True runs."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def segment(stats, labels, small, fps):
    lum, pur, lit = stats.T
    T = len(lum)
    from scipy.ndimage import median_filter
    is_change = smooth(pur, 5) > 5e-4          # purple pixels exist only on the change map
    lit_s = smooth(lit, 3)
    fade = lit_s < 0.6 * median_filter(lit_s, size=int(2 * fps) + 1, mode="nearest")  # lights fading out/in
    annual = ~is_change & ~fade
    ann_runs = [r for r in runs(annual) if r[1] - r[0] > fps * 2]
    chg_runs = [r for r in runs(is_change) if r[1] - r[0] > fps * 2]
    if len(ann_runs) < 2 or len(chg_runs) < 2:
        raise RuntimeError(f"segmentation failed: annual {ann_runs}, change {chg_runs}")
    A, C = ann_runs[0], ann_runs[1]
    B, D = chg_runs[0], chg_runs[1]
    # base map: dimmest frame between A and B
    # (the B->C fade goes fully dark; the A->B fade still shows ~30% of the lights)
    base_idx = int(B[1] + np.argmin(lit_s[B[1]:C[0]])) if C[0] > B[1] else int(np.argmin(lit_s))
    # zoom onset inside C / D: difference to the first block of the same kind, in the W. Europe inset area
    def zoom_onset(blk, ref_blk):
        ref = np.median(small[ref_blk[0] + 3: ref_blk[1] - 3], 0).astype(np.float32)
        y0, y1, x0, x1 = 92, 255, 86, 249  # W. Europe inset (480x270 coords of the left map)
        d = np.array([np.abs(small[k, y0:y1, x0:x1].astype(np.float32) - ref[y0:y1, x0:x1]).mean()
                      for k in range(blk[0], blk[1])])
        on = np.where(d > max(6.0, 0.5 * d.max()))[0]
        return (int(blk[0] + on[0]) if len(on) else None), d
    zC, dC = zoom_onset(C, A)
    zD, dD = zoom_onset(D, B)
    # year boundaries: label-region changes inside the annual blocks
    ldiff = np.r_[0, np.abs(np.diff(labels, axis=0)).mean((1, 2))]
    def years_in(blk, n_bounds=len(YEARS) - 1):
        """The n strongest label changes (non-max suppressed, >= 6 frames apart) inside the block."""
        lo, hi_ = blk
        seg = ldiff[lo + 1:hi_]
        cand = [i for i in range(len(seg)) if seg[i] > 0 and seg[i] >= seg[max(0, i - 1)] and seg[i] >= seg[min(len(seg) - 1, i + 1)]]
        cand.sort(key=lambda i: -seg[i])
        picked = []
        for i in cand:
            if all(abs(i - j) >= 6 for j in picked):
                picked.append(i)
            if len(picked) == n_bounds:
                break
        picked = sorted(lo + 1 + i for i in picked)
        thr = float(min(ldiff[picked])) if picked else float("nan")
        # sanity: the weakest kept change must stand well above the block's typical frame noise
        if len(picked) < n_bounds or thr < 4 * float(np.median(seg)) + 1e-6:
            return [], thr
        return picked, thr
    yA, thrA = years_in(A)
    yC, thrC = years_in(C)
    return {"A": A, "B": B, "C": C, "D": D, "base": base_idx, "zoom_C": zC, "zoom_D": zD,
            "year_bounds_A": yA, "year_bounds_C": yC, "thr": [thrA, thrC], "ldiff": ldiff,
            "is_change": is_change, "fade": fade}


def year_starts(blk, bounds, n=12):
    """Start frame of each of the n years: block start + detected boundaries (take the n-1 strongest)."""
    if len(bounds) != n - 1:
        return None
    return [blk[0]] + list(bounds)


# ---------------------------------------------------------------- pass 2: measurements
def grab(src: Path, idx: list[int], width=3840):
    idx = sorted(set(int(i) for i in idx))
    expr = "+".join(f"eq(n\\,{i})" for i in idx)
    fr = list(decode_stream(src, None, width, int(FULL_H * width / FULL_W),
                            vf_extra=f"select='{expr}',setpts=N/FRAME_RATE/TB"))
    if len(fr) != len(idx):
        raise RuntimeError(f"grabbed {len(fr)} of {len(idx)} frames")
    return dict(zip(idx, [f[:, : width // 2] for f in fr]))


def zoom_pan(lon):
    """The four zoom regions all sit between 10W and 90E: spread them across the stereo field (west left)."""
    return float(np.clip((lon - 40) / 50, -1, 1))


def lum709(rgb):
    return rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)


class MapGrid:
    def __init__(self, h, w):
        s = w / GEOREF["w"]
        g = GeoFit(GEOREF["x0"] * s, GEOREF["kx"] * s, np.asarray(GEOREF["ycoef"]) * s, w, h)
        lon, lat = g.grids()
        self.lon = ((lon + 180) % 360) - 180
        self.lat = lat
        LON, LAT = np.meshgrid(self.lon, self.lat)
        self.LON, self.LAT = LON, np.nan_to_num(LAT, nan=-99)
        self.valid = (self.LAT > -60) & (self.LAT < 78)
        for x0, y0, x1, y1 in OVERLAY_MASK:
            self.valid[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)] = False
        self.area = np.cos(np.deg2rad(np.clip(self.LAT, -89, 89)))

    def box(self, lon0, lon1, lat0, lat1):
        return self.valid & (self.LON >= lon0) & (self.LON < lon1) & (self.LAT >= lat0) & (self.LAT < lat1)

    def sectors(self, n):
        e = np.linspace(-180, 180, n + 1)
        return [(e[i], e[i + 1], self.box(e[i], e[i + 1], -90, 90)) for i in range(n)]


def change_field(change_rgb, base_rgb, cbar: Colorbar, min_light=6.0, min_chroma=10.0):
    """Signed relative change (-1 lights out .. +1 new lights) and light weight per pixel."""
    over = np.clip(change_rgb.astype(np.float32) - base_rgb.astype(np.float32), 0, 255)
    w = lum709(over)
    peak = over.max(2, keepdims=True)
    norm = np.where(peak > 0, over / np.maximum(peak, 1e-6) * 255, 0).astype(np.uint8)
    v, ok, _ = cbar.invert(norm, max_delta_e=40, shade_invariant=True, l_weight=0.3, min_chroma=min_chroma)
    v = np.where(ok, v, 0.0)  # unsaturated overlay = white = "no change"
    lit = w > min_light
    return np.where(lit, v, 0.0), np.where(lit, w, 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--demo", type=Path, default=Path("demo"))
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args(argv)
    out = a.out / str(PAGE)
    out.mkdir(parents=True, exist_ok=True)
    d = a.data / str(PAGE)
    src = d / "nl_2160p30_h265.mp4"
    if not src.exists():
        from ..data.ingest import download
        download(MOVIE, src)
    info = video_info(src)
    fps, duration = info["fps"], info["duration"]
    page = catalog.fetch(PAGE, a.data.parent / "catalog")

    c1 = d / "pass1.npz"
    if a.recompute or not c1.exists():
        stats, labels, small = pass1(src)
        np.savez_compressed(c1, stats=stats, labels=labels, small=small)
    z = np.load(c1)
    stats, labels, small = z["stats"], z["labels"], z["small"]
    seg = segment(stats, labels, small, fps)
    A, B, C, D = seg["A"], seg["B"], seg["C"], seg["D"]
    ysA = year_starts(A, seg["year_bounds_A"])
    ysC = year_starts(C, seg["year_bounds_C"])
    year_fallback = {}
    if ysA is None:  # uniform fallback (documented)
        year_fallback["A"] = len(seg["year_bounds_A"])
        ysA = list(np.linspace(A[0], A[0] + 11 * 18, 12).astype(int))
    if ysC is None:
        year_fallback["C"] = len(seg["year_bounds_C"])
        ysC = list(np.linspace(C[0], C[1] - 18, 12).astype(int))

    # pass 2: clean frames from A (end of each year's hold) and B (middle of the change map), base map
    endsA = ysA[1:] + [A[1]]
    pickA = [int((s + e) // 2) for s, e in zip(ysA[:-1], endsA[:-1])] + [int(min(ysA[-1] + 9, A[1] - 6))]
    pickB = list(np.linspace(B[0] + (B[1] - B[0]) * 0.2, B[0] + (B[1] - B[0]) * 0.8, 7).astype(int))
    base_ix = [seg["base"] - 1, seg["base"], seg["base"] + 1]
    c2 = d / "pass2.npz"
    want = sorted(set(pickA + pickB + base_ix + [5]))
    if a.recompute or not c2.exists() or list(np.load(c2)["idx"]) != want:
        fr = grab(src, want)
        np.savez_compressed(c2, idx=np.array(want), frames=np.stack([fr[i] for i in want]))
    z2 = np.load(c2)
    frames = dict(zip(z2["idx"].tolist(), z2["frames"]))
    base = np.median(np.stack([frames[i] for i in base_ix]), 0)
    H, W = base.shape[:2]
    grid = MapGrid(H, W)

    # colorbar from the frame itself (full-res coords -> pass-2 coords: x/2, y/2; it straddles the halves)
    cb_cache = d / "cbar_strip.npy"
    if a.recompute or not cb_cache.exists():
        x0, y0, x1, y1 = CBAR_BOX  # straddles the two halves, so decode the whole frame once
        k = pickB[len(pickB) // 2]
        whole = list(decode_stream(src, None, FULL_W, FULL_H, vf_extra=f"select='eq(n\\,{k})',setpts=N/FRAME_RATE/TB"))[0]
        np.save(cb_cache, whole[y0:y1, x0:x1])
    strip = np.load(cb_cache)
    cbar = Colorbar.from_image(strip, -1.0, 1.0, units="relative change (lights out -1 .. +1 new lights)")

    # annual light per year: light added over the faded base map
    lightA = np.stack([lum709(np.clip(frames[i].astype(np.float32) - base, 0, 255)) * grid.valid for i in pickA])
    glob = (lightA * grid.area).sum((1, 2))
    glob_rel = glob / glob[0]
    # change map
    ch = np.median(np.stack([frames[i] for i in pickB]), 0)
    v, wgt = change_field(ch, base, cbar)
    v, wgt = v * grid.valid, wgt * grid.valid

    def region_stats(mask):
        w = (wgt * grid.area)[mask]
        vv = v[mask]
        tot = float(w.sum())
        net = float((w * vv).sum())
        la = (lightA * grid.area)[:, mask].sum(1)
        return {"light_weight": tot, "net": net, "index": net / tot if tot > 0 else 0.0,
                "purple_px": int(((vv > 0.15) & (w > 0)).sum()), "orange_px": int(((vv < -0.15) & (w > 0)).sum()),
                "annual": la.tolist(), "annual_rel_2012": (la / la[0] if la[0] > 0 else la * 0).tolist()}

    sec12 = [(l0, l1, region_stats(m)) for l0, l1, m in grid.sectors(12)]
    sec6 = [(l0, l1, region_stats(m)) for l0, l1, m in grid.sectors(6)]
    zooms = {k: region_stats(grid.box(*b)) for k, b in ZOOMS.items()}
    taken = np.zeros_like(grid.valid)
    conts = {}
    for k, b in CONTINENTS.items():
        m = grid.box(*b) & ~taken
        taken |= m
        conts[k] = region_stats(m)
    # cross-check: the change map vs (2023 - 2012) from the annual frames, per 30 deg sector
    lit12 = [s for s in sec12 if s[2]["light_weight"] > 0.01 * sum(x[2]["light_weight"] for x in sec12)]
    dA = [(s[2]["annual"][-1] - s[2]["annual"][0]) / (s[2]["annual"][0] + 1e-9) for s in lit12]
    cross_r = float(np.corrcoef(dA, [s[2]["index"] for s in lit12])[0, 1]) if len(lit12) > 2 else float("nan")
    cont_d = {k: (c["annual"][-1] - c["annual"][0]) / (c["annual"][0] + 1e-9) for k, c in conts.items()}
    cross_r_cont = float(np.corrcoef(list(cont_d.values()), [c["index"] for c in conts.values()])[0, 1])

    # ------------------------------------------------------------ score (movie time)
    tf = lambda k: float(k) / fps
    notes: list[Note] = []
    years_pm = PitchMap(min(glob_rel) - 0.02, max(glob_rel) + 0.02, low=62, span=19, root=2, scale="major_pent")
    rel6 = np.array([s[2]["annual_rel_2012"] for s in sec6])  # (6, 12)
    share6 = np.array([s[2]["light_weight"] for s in sec6]); share6 = share6 / share6.sum()
    r6max = max(0.05, float(np.nanmax(np.abs(rel6[share6 > 0.02] - 1))))
    sec_pm = PitchMap(1 - r6max, 1 + r6max, low=50, span=24, root=2, scale="major_pent")
    years_notes, sector_notes, tick_notes, sweep_notes = [], [], [], []

    def year_line(starts, end, block, voice="years", vel=80):
        for y, s in enumerate(starts):
            e = starts[y + 1] if y + 1 < len(starts) else end
            p = int(years_pm(glob_rel[y])[0])
            years_notes.append(Note(tf(s), tf(e), p, vel, voice, float(glob_rel[y]), 0.0,
                                    {"year": YEARS[y], "block": block, "metric": "global light / 2012"}))
            tick_notes.append(Note(tf(s), tf(s) + 0.08, 76 if y == 0 else 72, 70 if y == 0 else 50, "year-ticks",
                                   float(YEARS[y]), 0.0, {"year": YEARS[y], "block": block}))

    def arpeggio(starts, end, block):
        for y, s in enumerate(starts):
            e = starts[y + 1] if y + 1 < len(starts) else end
            t0, t1 = tf(s), tf(e)
            k = 0
            step = 0.1
            while t0 + k * 6 * step < t1 - 0.05:
                for j in range(6):
                    if share6[j] < 0.02:
                        continue
                    ts = t0 + (k * 6 + j) * step
                    if ts >= t1:
                        break
                    l0, l1, st = sec6[j]
                    r = rel6[j, y]
                    sector_notes.append(Note(ts, ts + step * 1.6, int(sec_pm(r)[0]), int(40 + 60 * np.sqrt(share6[j] / share6.max())),
                                             "sectors", float(r), float(lon_pan((l0 + l1) / 2)),
                                             {"year": YEARS[y], "lon": [float(l0), float(l1)], "metric": "sector light / 2012"}))
                k += 1

    def zoom_lines(starts, end, block):
        zr = {k: np.array(z_["annual_rel_2012"]) for k, z_ in zooms.items()}
        zmax = max(0.05, max(float(np.abs(r - 1).max()) for r in zr.values()))
        zpm = PitchMap(1 - zmax, 1 + zmax, low=48, span=24, root=2, scale="major_pent")
        for name, r in zr.items():
            lon = float(np.mean(ZOOMS[name][:2]))
            for y, s in enumerate(starts):
                e = starts[y + 1] if y + 1 < len(starts) else end
                sector_notes.append(Note(tf(s), tf(e), int(zpm(r[y])[0]), 70, f"zoom {name}", float(r[y]), zoom_pan(lon),
                                         {"year": YEARS[y], "region": name, "metric": "regional light / 2012"}))
        return zpm

    def gesture(t0, dur, idx, voice, pan, src_info, center=74):
        allowed = PitchMap(0, 1, low=center - 14, span=28, root=2, scale="major_pent").allowed()
        c = int(np.argmin(np.abs(allowed - center)))
        k = int(round(abs(idx) * 6)) * (1 if idx >= 0 else -1)
        seq = [c] if k == 0 else [c, c + int(np.sign(k)) * max(1, abs(k) // 2), c + k]
        seq = [int(np.clip(q, 0, len(allowed) - 1)) for q in seq]
        n = len(seq)
        for i, q in enumerate(seq):
            ts = t0 + i * dur / max(n, 1) * 0.6
            te = t0 + dur * 0.92 if i == n - 1 else ts + dur / max(n, 1) * 0.6
            sweep_notes.append(Note(ts, te, int(allowed[q]), int(src_info.pop("_vel", 80)) if i == 0 else 80, voice, float(idx),
                                    float(pan), dict(src_info, step=i, interval_steps=k)))

    year_line(ysA, A[1], "A")
    arpeggio(ysA, A[1], "A")
    # B: west->east sweep of twelve 30 deg sectors over the clean change map
    b0, b1 = tf(B[0]) + 0.3, tf(B[1]) - 0.2
    lit_w = np.array([s[2]["light_weight"] for s in sec12])
    idx12 = np.array([s[2]["index"] for s in sec12])
    imax = max(1e-6, float(np.abs(idx12[lit_w > 0.01 * lit_w.sum()]).max()))
    sweep_t = []
    dur12 = (b1 - b0) / 12
    for j, (l0, l1, st) in enumerate(sec12):
        t0 = b0 + j * dur12
        sweep_t.append((t0, t0 + dur12, idx12[j] / imax if lit_w[j] > 0.01 * lit_w.sum() else 0.0))
        if lit_w[j] < 0.01 * lit_w.sum():
            continue  # (mostly ocean): rest
        gesture(t0, dur12, idx12[j] / imax, "change-sweep", lon_pan((l0 + l1) / 2),
                {"lon": [float(l0), float(l1)], "net_index": st["index"], "purple_px": st["purple_px"],
                 "orange_px": st["orange_px"], "block": "B",
                 "_vel": 45 + 80 * np.sqrt(lit_w[j] / lit_w.max())})
    # C: annual again with zooms -> global line + four regional voices
    year_line(ysC, C[1], "C", vel=60)
    zpm = zoom_lines(ysC, C[1], "C")
    # D: the four zooms west->east
    zorder = sorted(ZOOMS, key=lambda k: np.mean(ZOOMS[k][:2]))
    zidx = {k: zooms[k]["index"] for k in zorder}
    zmax = max(1e-6, max(abs(x) for x in zidx.values()))
    d0, d1 = tf(D[0]) + 0.3, min(tf(D[1]), duration) - 0.3
    durz = (d1 - d0) / len(zorder)
    for j, k in enumerate(zorder):
        t0 = d0 + j * durz
        sweep_t.append((t0, t0 + durz, zidx[k] / zmax))
        gesture(t0, durz, zidx[k] / zmax, "change-sweep", zoom_pan(np.mean(ZOOMS[k][:2])),
                {"region": k, "net_index": zidx[k], "block": "D", "_vel": 95})
    for n in sweep_notes:
        n.pan = float(np.clip(n.pan, -1, 1))
    for n in sector_notes:
        n.pan = float(np.clip(n.pan, -1, 1))
    notes = years_notes + sector_notes + sweep_notes + tick_notes

    # ------------------------------------------------------------ controls
    dtc = 0.05
    t_ctrl = np.arange(0, duration, dtc)
    from scipy.ndimage import median_filter
    lit_s = smooth(stats[:, 2], 3)
    up = np.clip(lit_s / (median_filter(lit_s, size=int(2 * fps) + 1, mode="nearest") + 1e-12), 0, 1)
    up = np.interp(t_ctrl, np.arange(len(up)) / fps, up)  # 1 = lights shown, ~0 during the fades
    anth = np.full(len(t_ctrl), np.nan)
    level = np.zeros(len(t_ctrl))
    gnorm = (glob_rel - glob_rel.min()) / max(1e-9, glob_rel.max() - glob_rel.min())
    for starts, blk in ((ysA, A), (ysC, C)):
        for y, s in enumerate(starts):
            e = starts[y + 1] if y + 1 < len(starts) else blk[1]
            m = (t_ctrl >= tf(s)) & (t_ctrl < tf(e))
            anth[m] = 0.3 + 0.6 * gnorm[y]
            level[m] = 0.3 + 0.7 * gnorm[y]
    for t0, t1, x in sweep_t:
        m = (t_ctrl >= t0) & (t_ctrl < t1)
        anth[m] = 0.55 + 0.4 * x
        level[m] = 0.5 + 0.5 * x
    anth[np.isnan(anth)] = 0.25  # between phases
    anth = smooth(anth, 7) * (0.3 + 0.7 * up)
    bio = smooth(np.clip(1.0 - anth, 0, 1), 7)
    bright = smooth(level * up, 5)
    curves = {"geophony": np.full(len(t_ctrl), 0.12), "anthrophony": np.clip(anth, 0, 1),
              "biophony": np.clip(bio, 0, 1), "brightness": bright}

    # ------------------------------------------------------------ render
    yrs = render_notes(years_notes, duration, "pluck", gain=0.28)
    sec_audio = render_notes([n for n in sector_notes if n.voice == "sectors"], duration, "pluck", gain=0.12)
    zoom_audio = render_notes([n for n in sector_notes if n.voice.startswith("zoom")], duration, "pad", gain=0.10)
    swp = render_notes(sweep_notes, duration, "pluck", gain=0.24)
    tck = render_notes(tick_notes, duration, "drum", gain=0.25)
    bands = sec_audio + zoom_audio[: len(sec_audio)] if len(zoom_audio) >= len(sec_audio) else zoom_audio + sec_audio[: len(zoom_audio)]
    geo = render_curve_voice("geophony", t_ctrl, curves["geophony"], duration, seed=3)
    an = render_curve_voice("anthrophony", t_ctrl, curves["anthrophony"], duration, seed=4)
    bi = render_curve_voice("biophony", t_ctrl, curves["biophony"], duration, seed=5)

    # ------------------------------------------------------------ proof
    # (1) years line: pitch recovered at each note onset vs global light
    yv = np.array([n.value for n in years_notes])
    est = midi_to_value(note_pitches(yrs, years_notes), years_pm)
    real = score(yv, est)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(yv))
    ctrl_notes = [Note(n.start, n.end, int(years_pm(yv[perm[i]])[0]), n.velocity, "ctrl", float(yv[perm[i]])) for i, n in enumerate(years_notes)]
    ctrl = score(yv, midi_to_value(note_pitches(render_notes(ctrl_notes, duration, "pluck", gain=0.28), ctrl_notes), years_pm))
    # (2) sweep: direction recovered from audio (last minus first note pitch per gesture) vs net change
    sp = note_pitches(swp, sweep_notes, fmin=200, fmax=2000, win=0.2)
    rec, tru = [], []
    for gi, (t0, t1, x) in enumerate(sweep_t):
        ix = [i for i, n in enumerate(sweep_notes) if t0 - 1e-6 <= n.start < t1 - 1e-6]
        if not ix or not np.isfinite(sp[ix[0]]) or not np.isfinite(sp[ix[-1]]):
            continue
        rec.append(float(sp[ix[-1]] - sp[ix[0]]))
        tru.append(float(x))
    rec, tru = np.array(rec), np.array(tru)
    sign_acc = float(np.mean(np.sign(np.round(rec)) == np.sign(np.round(tru * 6)))) if len(rec) else float("nan")
    sweep_r = float(np.corrcoef(rec, tru)[0, 1]) if len(rec) > 2 else float("nan")

    fidelity = {"years_melody": {"real": real, "shuffled_control": ctrl, "stem": "years"},
                "change_sweep": {"direction_sign_accuracy": sign_acc, "interval_vs_net_change_r": sweep_r, "n": int(len(rec)),
                                 "stem": "change-sweep"},
                "cross_check_change_map_vs_annual_2023_minus_2012": {"sector30_r": cross_r, "continent_r": cross_r_cont,
                                                                     "note": "two independent encodings in the same movie agree"}}

    # ------------------------------------------------------------ description
    big = {k: c for k, c in conts.items() if c["light_weight"] > 0.02 * sum(x["light_weight"] for x in conts.values())}
    up = sorted(big, key=lambda k: -big[k]["index"])
    regions = {**big, **zooms}
    dn = sorted([k for k, c in regions.items() if c["index"] < 0 or c["annual_rel_2012"][-1] < 0.97],
                key=lambda k: regions[k]["index"])
    gchg = (glob_rel[-1] - 1) * 100
    desc = (f"Night lights seen from space, 2012 to 2023: total light on the map grows by about {gchg:.0f} percent "
            f"(relative pixel brightness; the map has no numeric scale). "
            f"Rising notes and purple mark places that got brighter, most of all {up[0]} and {up[1]}; "
            + (f"falling notes and orange mark places that got dimmer, such as {' and '.join(dn[:2])}. " if dn else "almost nothing got dimmer. ")
            + "As the lights spread, the city hum grows and the chorus of birds and insects fades.")
    legend = [
        {"voice": "pluck (years)", "meaning": "total night light each year, 2012-2023; higher = more light"},
        {"voice": "pluck arpeggio (sectors)", "meaning": "six longitude sectors west to east, each pitched by its light relative to 2012; panned by longitude"},
        {"voice": "plucked runs (change sweep)", "meaning": "2012->2023 change map swept west to east: rising run = brighter (purple), falling run = dimmer (orange); bigger interval = stronger"},
        {"voice": "pads (zoom regions)", "meaning": "Western Europe, Eastern Mediterranean, Ukraine and India, each pitched by its light relative to 2012"},
        {"voice": "ticks (events)", "meaning": "a new year starts"},
        {"voice": "city hum (anthrophony)", "meaning": "louder with more light and with brightening"},
        {"voice": "birds and insects (biophony)", "meaning": "quieter as the lights grow"},
        {"voice": "wind (geophony)", "meaning": "quiet constant bed, not data-driven here"},
    ]
    stems = [Stem("years", "melody", yrs, 1.3), Stem("change-sweep", "melody", swp, 1.1), Stem("sectors", "bands", bands, 0.4),
             Stem("year-ticks", "events", tck, 0.5), Stem("city", "anthrophony", an, 0.45), Stem("chorus", "biophony", bi, 0.35),
             Stem("wind", "geophony", geo, 0.2)]
    to_midi(notes, out / "score.mid", programs={"years": 46, "sectors": 46, "change-sweep": 12, "year-ticks": 115,
                                                **{f"zoom {k}": 89 for k in ZOOMS}})
    segs = {k: [tf(seg[k][0]), tf(seg[k][1])] for k in "ABCD"}
    report = {
        "id": PAGE, "title": page.get("title"), "movie": MOVIE, "movie_info": info,
        "page_description": page.get("description", "").split("||")[0].strip(),
        "segments_s": segs, "segment_meaning": {"A": "annual series", "B": "change map", "C": "annual series + zooms", "D": "change map + zooms"},
        "base_map_frame": seg["base"], "zoom_onset_s": {"C": tf(seg["zoom_C"]) if seg["zoom_C"] else None, "D": tf(seg["zoom_D"]) if seg["zoom_D"] else None},
        "year_starts_s": {"A": [tf(x) for x in ysA], "C": [tf(x) for x in ysC]}, "year_detection_fallback": year_fallback,
        "frames_measured": {"A": pickA, "B": pickB, "base": base_ix},
        "georef": GEOREF, "colorbar": {"box_fullres": CBAR_BOX, "range": "-1 (lights out) .. 0 (no change) .. +1 (new lights)",
                                       "note": "no numeric scale on the page: values are relative"},
        "global_light_rel_2012": dict(zip(YEARS, glob_rel.tolist())),
        "continents": {k: {kk: c[kk] for kk in ("index", "net", "light_weight", "purple_px", "orange_px")} | {"annual_2023_vs_2012": cont_d[k]} for k, c in conts.items()},
        "zooms": {k: {kk: z_[kk] for kk in ("index", "purple_px", "orange_px", "annual_rel_2012")} for k, z_ in zooms.items()},
        "sectors30": [{"lon": [l0, l1], **{kk: s[kk] for kk in ("index", "light_weight", "purple_px", "orange_px")}} for l0, l1, s in sec12],
        "pitchmaps": {"years": years_pm.describe(), "sectors": sec_pm.describe(), "zooms": zpm.describe()},
        "notes": {k: sum(1 for n in notes if n.voice == k) for k in sorted({n.voice for n in notes})},
        "fidelity": fidelity,
    }
    track = write_track(out, page_id=PAGE, title=page.get("title"), video_url=MOVIE, duration_s=duration,
                        units="relative night-light change (no numeric scale): -1 lights out .. +1 new lights; years as light / 2012",
                        description=desc, legend=legend, stems=stems, notes=notes, t_ctrl=t_ctrl, curves=curves,
                        fidelity=fidelity, report=report,
                        data_source={"kind": "pixels", "movie": MOVIE, "underlying": "NASA Black Marble (VIIRS, Suomi NPP) annual composites",
                                     "scale": "relative: diverging colormap without numbers; inverted from the burned-in colorbar",
                                     "projection": "left of two identical flat maps, georeferenced against 5011 (cubic in latitude)"},
                        extra={"segments": [{"name": n_, "start": segs[k][0], "end": segs[k][1]} for k, n_ in
                                            (("A", "annual series"), ("B", "change map"), ("C", "annual series + zooms"), ("D", "change map + zooms"))]})
    plot(out, small, seg, glob_rel, sec12, conts, v, wgt, years_notes, est, sweep_t, rec, tru, fps, t_ctrl, curves)
    export_demo(out, a.demo / str(PAGE), ["mix"])
    print(json.dumps({k: report[k] for k in ("segments_s", "zoom_onset_s", "year_starts_s", "year_detection_fallback",
                                             "global_light_rel_2012", "fidelity")}, indent=1))
    print(json.dumps({k: round(c["index"], 3) for k, c in report["continents"].items()}, indent=1))
    print(desc)
    return track


def plot(out, small, seg, glob_rel, sec12, conts, v, wgt, years_notes, est, sweep_t, rec, tru, fps, t_ctrl, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(3, 2, 1); ax.imshow(np.clip(small[(seg["B"][0] + seg["B"][1]) // 2].astype(int) * 3, 0, 255).astype(np.uint8)); ax.set_title("change map (left map, x3 brightness)"); ax.axis("off")
    ax = fig.add_subplot(3, 2, 2)
    show = np.where(wgt > 0, v, np.nan)
    import cv2
    ax.imshow(cv2.resize(np.nan_to_num(show * np.sqrt(wgt / (wgt.max() + 1e-9)) * 4), (480, 270), interpolation=cv2.INTER_AREA), cmap="PuOr", vmin=-0.3, vmax=0.3)
    ax.set_title("inverted change x light (orange = less, purple = more)"); ax.axis("off")
    ax = fig.add_subplot(3, 2, 3)
    t = np.arange(len(seg["ldiff"])) / fps
    ax.plot(t, seg["ldiff"], lw=0.6, label="year-label change")
    for k, c in zip("ABCD", ("C0", "C1", "C2", "C3")):
        ax.axvspan(seg[k][0] / fps, seg[k][1] / fps, color=c, alpha=0.12)
        ax.text(seg[k][0] / fps, ax.get_ylim()[1] * 0.9, k)
    ax.set_title("phases A annual / B change / C annual+zoom / D change+zoom"); ax.set_xlabel("movie s")
    ax = fig.add_subplot(3, 2, 4)
    ax.plot(YEARS, glob_rel, "o-", label="global light / 2012")
    for k, c in conts.items():
        if c["light_weight"] > 0.04 * sum(x["light_weight"] for x in conts.values()):
            ax.plot(YEARS, c["annual_rel_2012"], lw=0.8, label=k)
    ax.legend(fontsize=6); ax.set_title("annual light relative to 2012")
    ax = fig.add_subplot(3, 2, 5)
    ax.bar([np.mean([l0, l1]) for l0, l1, _ in sec12], [s["index"] for _, _, s in sec12], width=25,
           color=["#8e6bd1" if s["index"] > 0 else "#e8964a" for _, _, s in sec12])
    ax.set_title("net change index per 30 deg sector (change map)"); ax.set_xlabel("lon")
    ax = fig.add_subplot(3, 2, 6)
    ax.plot(t_ctrl, curves["anthrophony"], label="anthrophony"); ax.plot(t_ctrl, curves["biophony"], label="biophony")
    ax.plot(t_ctrl, curves["brightness"], label="brightness", lw=0.6)
    ax.legend(fontsize=7); ax.set_title(f"controls; sweep direction from audio vs data r={np.corrcoef(rec, tru)[0, 1]:.2f}" if len(rec) > 2 else "controls")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=95); plt.close(fig)


if __name__ == "__main__":
    main()
