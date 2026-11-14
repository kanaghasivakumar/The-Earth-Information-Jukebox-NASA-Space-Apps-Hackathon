"""5209 Zonal Climate Anomalies 1880-2023 (GISTEMP) -> "Planetary Bands, Warming World".

    python -m jukebox.datasets.zonal_5209 [--out outputs] [--data data/raw]

The SVS movie is a strip chart, not a map: for each of 8 latitude zones a coloured
marker slides along a -2..+2 degC axis, one position per year. There is no colorbar
image on the SVS page, so the movie is inverted by *position*: the five axis lines
calibrate x -> degC, and the marker centre in each zone's row gives the anomaly.

Ground truth is NASA GISS GISTEMP v4 ZonAnn.Ts+dSST.csv (same 8 zones). The music
is driven by the CSV (true data); the movie gives only the year -> video-time sync.
Two fidelity numbers are reported:
  video_vs_csv   pixel-inverted marker values vs CSV, r and MAE per zone
  audio          pyin round-trip of each rendered stem vs the CSV values it encodes
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

import numpy as np

from ..data import catalog
from ..data.ingest import download
from ..proof.fidelity import score
from ..render.synth import SR, render_notes
from ..score.mapping import Note, PitchMap
from .common import (export_demo, grab_frame, iter_frames, movie_info, norm01, roundtrip, write_track)

PAGE = 5209
MOVIE = "https://svs.gsfc.nasa.gov/vis/a000000/a005200/a005209/GISTEMP_Zonal_1880-2023_C.mp4"
CSV_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv"
# north -> south, matching the rows of the chart
ZONES = ["64N-90N", "44N-64N", "24N-44N", "EQU-24N", "24S-EQU", "44S-24S", "64S-44S", "90S-64S"]
ZONE_LABEL = ["Arctic 64-90N", "N. mid-lat 44-64N", "N. subtropics 24-44N", "N. tropics 0-24N",
              "S. tropics 0-24S", "S. subtropics 24-44S", "S. mid-lat 44-64S", "Antarctic 64-90S"]
AXIS_VALUES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
W = 1920                       # decode width (half of the 4K master)
CHART_X = (540, 1900)          # chart region in 1920-wide frames (excludes the globe)
CHART_Y = (250, 800)


# ---------------------------------------------------------------- ground truth
def load_csv(path: Path) -> dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(CSV_URL, headers=catalog.UA), timeout=60) as r:
            path.write_bytes(r.read())
    rows = list(csv.DictReader(path.read_text().splitlines()))
    years = np.array([int(r["Year"]) for r in rows])
    def col(k):
        return np.array([float(r[k]) if r[k].strip() not in ("", "***", "****") else np.nan for r in rows])
    return {"year": years, "Glob": col("Glob"), "zones": np.stack([col(z) for z in ZONES], 1)}


# ---------------------------------------------------------------- video inversion
def _runs(mask: np.ndarray):
    ix = np.where(mask)[0]
    if ix.size == 0:
        return []
    return [(int(s[0]), int(s[-1])) for s in np.split(ix, np.where(np.diff(ix) > 1)[0] + 1)]


def find_axis(frame: np.ndarray) -> np.ndarray:
    """x of the five vertical axis lines: thin columns bright over most of the chart height."""
    mx = frame[CHART_Y[0]:CHART_Y[1], CHART_X[0]:CHART_X[1]].max(2)
    frac = (mx > 150).mean(0)
    runs = [r for r in _runs(frac > 0.6) if r[1] - r[0] < 10]
    xs = np.array([(a + b) / 2 + CHART_X[0] for a, b in runs])
    if len(xs) != 5:
        raise RuntimeError(f"expected 5 axis lines, found {xs}")
    return xs


def find_rows(frame: np.ndarray, axis_x: np.ndarray) -> list[tuple[int, int]]:
    """(top, bottom) of each zone's marker: wide bright blobs, sorted north -> south."""
    import cv2
    sub = frame[CHART_Y[0]:CHART_Y[1], CHART_X[0]:CHART_X[1]]
    m = (sub.max(2) > 60).astype(np.uint8)
    for x in axis_x:
        c = int(round(x)) - CHART_X[0]
        m[:, c - 4:c + 5] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, 9), np.uint8))  # drop thin ticks
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    blobs = [st[i] for i in range(1, n) if st[i, cv2.CC_STAT_WIDTH] >= 10 and st[i, cv2.CC_STAT_HEIGHT] >= 30]
    blobs.sort(key=lambda s: s[cv2.CC_STAT_TOP])
    rows = [(int(b[cv2.CC_STAT_TOP]) + CHART_Y[0], int(b[cv2.CC_STAT_TOP] + b[cv2.CC_STAT_HEIGHT]) + CHART_Y[0])
            for b in blobs]
    if len(rows) != 8:
        raise RuntimeError(f"expected 8 zone markers, found {len(rows)}: {rows}")
    return rows


def marker_x(frame: np.ndarray, top: int, axis_x: np.ndarray, thresh: float = 60.0, mw: int | None = None):
    """Centre x and width of the marker in one zone row.

    Uses rows just inside the marker's top edge: the marker is taller than the grey
    history ticks, so those rows contain only the marker (plus the axis lines,
    which are bridged by interpolation so an overlapping marker stays one run).
    """
    prof = frame[top + 2:top + 8, CHART_X[0]:CHART_X[1]].max(2).mean(0).astype(float)
    for x in axis_x:
        c = int(round(x)) - CHART_X[0]
        a, b = c - 4, c + 5
        prof[a:b] = np.minimum(prof[a - 2], prof[b + 1])
    runs = [r for r in _runs(prof > thresh) if r[1] - r[0] + 1 >= 10]
    if not runs:
        return np.nan, 0
    a, b = max(runs, key=lambda r: r[1] - r[0])
    width = b - a + 1
    if mw and width > mw + 2:
        # a fresh grey tick touching the marker widens the run: keep the brightest mw-wide window
        cs = np.r_[0, np.cumsum(prof[a:b + 1])]
        k = int(np.argmax(cs[mw:] - cs[:-mw]))
        a, b = a + k, a + k + mw - 1
    return (a + b) / 2 + CHART_X[0], width


def extract(movie: Path, cache: Path) -> dict:
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    info = movie_info(movie)
    last = grab_frame(movie, info["duration"] - 0.5, W)   # every zone has a marker at the end
    axis_x = find_axis(last)
    rows = find_rows(last, axis_x)
    mw = int(np.median([marker_x(last, t, axis_x)[1] for t, _ in rows]))
    xs, widths = [], []
    for f in iter_frames(movie, W):
        r = [marker_x(f, t, axis_x, mw=mw) for t, _ in rows]
        xs.append([p[0] for p in r]); widths.append([p[1] for p in r])
    xs, widths = np.array(xs, float), np.array(widths, float)
    slope, icpt = np.polyfit(axis_x, AXIS_VALUES, 1)
    out = {"x": xs, "width": widths, "axis_x": axis_x, "rows": np.array(rows), "fps": np.array(info["fps"]),
           "slope": np.array(slope), "icpt": np.array(icpt), "last_frame": last}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def _design(n_frames: int, t0: float, T: float, K: int) -> np.ndarray:
    """Frame -> knot interpolation weights for a piecewise-linear slide (knot k at frame t0 + k*T)."""
    f = np.arange(n_frames)
    u = np.clip((f - t0) / T, 0, K - 1)
    k = np.minimum(np.floor(u).astype(int), K - 2)
    a = u - k
    A = np.zeros((n_frames, K))
    A[f, k] = 1 - a
    A[f, k + 1] += a
    return A


def _solve(vals, A):
    """Least-squares knot values per zone (NaN frames dropped per zone); returns (y, sse)."""
    K = A.shape[1]
    y = np.full((K, vals.shape[1]), np.nan)
    sse = 0.0
    for b in range(vals.shape[1]):
        m = np.isfinite(vals[:, b])
        y[:, b], *_ = np.linalg.lstsq(A[m], vals[m, b], rcond=None)
        sse += float(((A[m] @ y[:, b] - vals[m, b]) ** 2).sum())
    return y, sse


def fit_years(vals: np.ndarray, n_years: int, eps: float = 0.005):
    """Recover one value per year from the movie alone.

    The markers slide linearly from year to year (no hold), so each zone's x(t) is a
    piecewise-linear curve whose knots are the yearly values. The last knot is where all
    motion stops; the first knot frame t0 and the period T are fitted by least squares
    (coarse-to-fine grid). No CSV values are used here.
    """
    F = len(vals)
    d = np.nanmax(np.abs(np.diff(vals, axis=0)), axis=1)
    moving = np.where(d > eps)[0]
    t_end0 = float(moving[-1] + 1) if moving.size else F - 1.0
    best = (np.inf, None, None)
    span_e, span_0, step = 3.0, 8.0, 0.5
    c_e, c_0 = t_end0, 0.0
    for _ in range(4):
        for te in np.arange(c_e - span_e, c_e + span_e + 1e-9, step):
            for t0 in np.arange(c_0 - span_0, c_0 + span_0 + 1e-9, step):
                T = (te - t0) / (n_years - 1)
                _, sse = _solve(vals, _design(F, t0, T, n_years))
                if sse < best[0]:
                    best = (sse, t0, T)
        c_0, c_e = best[1], best[1] + best[2] * (n_years - 1)
        span_e, span_0, step = step * 2, step * 2, step / 5
    _, t0, T = best
    y, sse = _solve(vals, _design(F, t0, T, n_years))
    rms = float(np.sqrt(sse / np.isfinite(vals).sum()))
    return y, t0, T, rms


# ---------------------------------------------------------------- music
def build_notes(onsets, years, zones, glob, beat, lead_pm, band_pms):
    """Lead = global mean (one note per year); 8 zone voices strummed north -> south each year."""
    lead, bands = [], {z: [] for z in ZONES}
    strum = min(0.012, beat / 16)
    ends = np.r_[onsets[1:], onsets[-1] + beat]
    vel_lead = (55 + 45 * norm01(glob, -0.6, 1.3)).astype(int)
    for i, (t0, t1, y) in enumerate(zip(onsets, ends, years)):
        p = int(lead_pm(glob[i])[0])
        if lead and lead[-1].pitch == p and abs(lead[-1].end - t0) < 1e-6:
            lead[-1].end = float(t1)
        else:
            lead.append(Note(float(t0), float(t1), p, int(vel_lead[i]), "lead global", float(glob[i]), 0.0,
                             {"year": int(y), "series": "Glob", "units": "degC anomaly vs 1951-1980"}))
        for b, z in enumerate(ZONES):
            v = zones[i, b]
            if not np.isfinite(v):
                continue
            pm = band_pms[b]
            pz = int(pm(v)[0])
            pan = float(-0.8 + 1.6 * b / 7)   # north left, south right
            vel = int(40 + 60 * norm01(abs(v), 0, 2.5))
            s = float(t0 + b * strum)
            prev = bands[z][-1] if bands[z] else None
            if prev and prev.pitch == pz and abs(prev.end - s) < 1e-6:
                prev.end = float(t1 + b * strum)
                continue
            bands[z].append(Note(s, float(t1 + b * strum), pz, vel, f"band {z}", float(v), pan,
                                 {"year": int(y), "series": z, "units": "degC anomaly vs 1951-1980"}))
    return lead, bands


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--demo", type=Path, default=Path("demo"))
    ap.add_argument("--band-timbre", default="keys", help="synth timbre for the 8 zone voices")
    a = ap.parse_args(argv)
    d = a.data / str(PAGE)
    out = a.out / str(PAGE)
    out.mkdir(parents=True, exist_ok=True)
    page = catalog.fetch(PAGE, a.data.parent / "catalog")
    movie = download(MOVIE, d / Path(MOVIE).name)
    gis = load_csv(a.data / "gistemp" / "ZonAnn.Ts+dSST.csv")

    # 1. invert the movie by marker position
    mk = extract(movie, d / "markers.npz")
    fps = float(mk["fps"])
    vals = mk["slope"] * mk["x"] + mk["icpt"]                       # (frames, 8) degC
    first_year, last_year = 1880, 2023                          # from the page title
    n_years = last_year - first_year + 1
    fit_cache = d / "yearfit.npz"
    if fit_cache.exists():
        z = np.load(fit_cache)
        vid, t0, T, fit_rms = z["y"], float(z["t0"]), float(z["T"]), float(z["rms"])
    else:
        vid, t0, T, fit_rms = fit_years(vals, n_years)
        np.savez(fit_cache, y=vid, t0=t0, T=T, rms=fit_rms)
    years_vid = np.arange(first_year, last_year + 1)
    onsets = (t0 + T * np.arange(n_years)) / fps                  # video time of each year's knot
    # 2. align with CSV
    sel = np.isin(gis["year"], years_vid)
    csv_z = gis["zones"][sel]
    csv_g = gis["Glob"][sel]
    years = gis["year"][sel]
    n = min(len(years), len(years_vid))
    vid, onsets, years_vid = vid[:n], onsets[:n], years_vid[:n]
    csv_z, csv_g, years = csv_z[:n], csv_g[:n], years[:n]
    assert (years == years_vid).all()
    per_zone = {}
    for b, z in enumerate(ZONES):
        s = score(csv_z[:, b], vid[:, b])
        per_zone[z] = {"r": s["pearson_r"], "mae": s["mae"], "r2": s["r2"], "n": s["n"],
                       "bias": float(np.nanmean(vid[:, b] - csv_z[:, b]))}
    allv = score(csv_z.ravel(), vid.ravel())
    # off-by-one check: a +-1 year shift should fit worse than 0
    lag = {}
    for k in (-1, 0, 1):
        A = csv_z[max(0, k):n - max(0, -k)]
        B = vid[max(0, -k):n - max(0, k)]
        lag[k] = float(np.nanmean(np.abs(A - B)))

    # 3. music: data = CSV, sync = video year onsets
    dur_video = movie_info(movie)["duration"]
    beat = float(np.median(np.diff(onsets)))
    lead_pm = PitchMap(-0.6, 1.4, low=62, span=24, root=2, scale="major_pent")          # D4..D6
    band_pms = []
    for b in range(8):
        off = int(round((7 - b) * 12 / 7)) - 6                   # north bands sit higher
        band_pms.append(PitchMap(-2.7, 3.3, low=42 + off, span=36, root=2, scale="major_pent"))
    lead, bands = build_notes(onsets, years, csv_z, csv_g, beat, lead_pm, band_pms)
    duration = float(max(dur_video, onsets[-1] + beat))
    stems = {"lead_global": render_notes(lead, duration, "pluck", gain=.3)}
    for b, z in enumerate(ZONES):
        stems[f"band_{z}"] = render_notes(bands[z], duration, a.band_timbre, gain=.12)
    groups = {"lead_global": "melody", **{f"band_{z}": "bands" for z in ZONES}}

    # 4. audio round-trip on the CSV curves (stems rendered alone)
    grid = onsets
    fid_lead, est_lead = roundtrip(stems["lead_global"], lead_pm, grid, csv_g)
    fid_bands = {}
    for b, z in enumerate(ZONES):
        # strummed band notes start b*strum later: shift the sampling grid accordingly
        f, _ = roundtrip(stems[f"band_{z}"], band_pms[b], grid + min(0.012, beat / 16) * b, csv_z[:, b])
        fid_bands[z] = f
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(csv_g))
    ctrl_lead, _ = build_notes(onsets, years, csv_z[perm], csv_g[perm], beat, lead_pm, band_pms)
    ctrl_audio = render_notes(ctrl_lead, duration, "pluck", gain=.3)
    fid_ctrl, _ = roundtrip(ctrl_audio, lead_pm, grid, csv_g)

    # 5. controls (per year, on the video clock)
    above1 = (csv_z > 1.0).mean(1)
    curves = {
        "geophony": norm01(csv_g, -0.6, 1.4),          # global heat -> intensity of wind/sea bed
        "anthrophony": np.zeros(n),                      # not in this dataset
        "biophony": np.zeros(n),                         # not in this dataset
        "brightness": above1,                            # share of zones > +1 degC
    }
    all_notes = lead + [x for z in ZONES for x in bands[z]]
    import shutil
    from .common import write_wav
    write_wav(out / "control_shuffled.wav", ctrl_audio / max(1e-9, np.abs(ctrl_audio).max()) * 0.9)
    fidelity = {"r2": fid_lead["r2"], "pearson_r": fid_lead["pearson_r"], "mae": fid_lead["mae"],
                "control_r2": fid_ctrl["r2"]}
    desc = ("NASA GISS surface temperature anomalies from 1880 to 2023 for eight latitude zones, "
            "relative to 1951-1980. Each year is one beat: the plucked lead plays the global mean and "
            "eight voices, strummed from the Arctic down to Antarctica, play each zone. "
            "Higher pitch means warmer; by the end every zone has climbed, the Arctic highest.")
    legend = [{"voice": "lead_global", "meaning": "global mean anomaly; D4 = -0.6 degC, D6 = +1.4 degC"},
              {"voice": "bands", "meaning": "one voice per latitude zone, north panned left, -2.7..+3.3 degC over 3 octaves"},
              {"voice": "geophony", "meaning": "control curve: global anomaly, louder when warmer"},
              {"voice": "brightness", "meaning": "control curve: fraction of zones warmer than +1 degC"},
              {"voice": "anthrophony/biophony", "meaning": "silent: not measured by this dataset"}]
    track = write_track(
        out, page_id=PAGE, title=page.get("title", "Zonal Climate Anomalies 1880-2023"), video_url=MOVIE,
        duration_s=duration, units="°C anomaly (vs 1951–1980)", description=desc, legend=legend,
        stems=stems, stem_groups=groups, notes=all_notes, controls_t=onsets, curves=curves,
        fidelity=fidelity,
        data_source={"music": "NASA GISS GISTEMP v4 ZonAnn.Ts+dSST.csv", "url": CSV_URL,
                     "sync": "SVS movie marker positions (year onsets)"},
        mix_gains={"lead_global": 1.0, **{f"band_{z}": 0.8 for z in ZONES}},
        extra={"video_inversion": {"r": allv["pearson_r"], "mae": allv["mae"]}})

    report = {
        "id": PAGE, "title": page.get("title"), "movie": MOVIE, "fps": fps, "frames": int(len(vals)),
        "duration_s": duration, "beat_s": beat, "years": [int(years[0]), int(years[-1])], "n_years": int(n),
        "colorbar": "none on the SVS page; inverted by marker position against the axis lines",
        "axis_x_px@1920": mk["axis_x"].tolist(), "px_per_degC": float(1 / mk["slope"]),
        "zone_rows_px@1920": mk["rows"].tolist(),
        "marker_width_px_median": float(np.nanmedian(mk["width"][mk["width"] > 0])),
        "valid_frame_fraction": float(np.isfinite(vals).all(1).mean()),
        "year_fit": {"first_knot_frame": float(t0), "frames_per_year": float(T), "rms_degC": fit_rms,
                     "method": "piecewise-linear knots fitted to marker motion, video only"},
        "lag_mae": lag,
        "video_vs_csv": {"all": allv, "per_zone": per_zone},
        "audio_roundtrip": {"lead_global": fid_lead, "bands": fid_bands, "shuffled_control": fid_ctrl},
        "lead_pitchmap": lead_pm.describe(), "band_pitchmaps": [p.describe() for p in band_pms],
        "notes": {"lead": len(lead), "bands": int(sum(len(v) for v in bands.values()))},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    plot(out, years, csv_z, vid, csv_g, est_lead, mk, onsets)
    export_demo(out, a.demo / str(PAGE), names=("mix", "lead_global", "control_shuffled"))
    print(json.dumps({k: report[k] for k in ("n_years", "year_fit", "lag_mae", "video_vs_csv")}, indent=1, default=float))
    print(json.dumps(report["audio_roundtrip"], indent=1, default=float))


def plot(out, years, csv_z, vid, csv_g, est_lead, mk, onsets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    f = mk["last_frame"]
    ax[0, 0].imshow(f)
    for x in mk["axis_x"]:
        ax[0, 0].axvline(x, color="c", lw=.5)
    for t, b in mk["rows"]:
        ax[0, 0].axhline(t + 4, color="y", lw=.5)
    ax[0, 0].set_title("last frame, axis lines + sampled marker rows"); ax[0, 0].axis("off")
    cmap = plt.get_cmap("coolwarm")
    for b, z in enumerate(ZONES):
        c = cmap(1 - b / 7)
        ax[0, 1].plot(years, csv_z[:, b] + 0 * b, color=c, lw=1, label=z)
        ax[0, 1].plot(years, vid[:, b], ".", color=c, ms=2)
    ax[0, 1].set_title("zone anomalies: CSV (lines) vs movie inversion (dots)"); ax[0, 1].legend(fontsize=6, ncol=2)
    ax[1, 0].scatter(csv_z.ravel(), vid.ravel(), s=3)
    lim = [-2.2, 3.2]
    ax[1, 0].plot(lim, lim, "k--", lw=.5); ax[1, 0].set_xlabel("CSV degC"); ax[1, 0].set_ylabel("video degC")
    ax[1, 0].set_title("video inversion vs GISTEMP CSV")
    ax[1, 1].plot(onsets, csv_g, label="GISTEMP global (CSV)")
    ax[1, 1].plot(onsets, est_lead, ".", label="recovered from lead audio")
    ax[1, 1].set_xlabel("video time (s)"); ax[1, 1].set_title("audio round-trip"); ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
