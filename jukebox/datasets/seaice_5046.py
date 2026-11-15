"""5046 Daily Arctic Sea Ice, By Year -> the "two winters" track.

    python -m jukebox.datasets.seaice_5046 [--width 960] [--stretch 1.5]

The SVS movie (arctic_sea_ice_moving_two_year_w_dates, 3840x2160, 30p, 730 frames, 24.33 s) shows one
day per frame on a fixed orthographic-looking globe centred near the North Pole. Sea ice is drawn
blue-white from JAXA AMSR2 (3-day running minimum, 15 % concentration threshold); land is Blue Marble
Next Generation, fading month to month, so it is snow-white in winter. There is no colorbar: the
movie shows extent (ice / no ice), not concentration.

Dates: the burned-in label reads 16 Apr 2021 on frame 0 (checked by eye at 0.5 s = 01 May 2021, 5.0 s =
13 Sep 2021, 10.5 s = 25 Feb 2022, 24.3 s = 15 Apr 2023), and the label crop changes on every frame, so
frame i is 16 Apr 2021 + i days. (The SVS frames folder now lists 2024304..2026303, a newer render
than the hosted movie, so it is not used for dates.)

Extent per frame:
  disc     globe centre and radius from the non-black pixels
  classes  water (blue), white (ice or snow), land-coloured (green/brown/grey, not blue) per pixel
  land     pixels that are land-coloured on >= 2 % of frames (snow-free summer land), plus the
           Greenland ice sheet: the never-water white component that contains Greenland's centre
  ice      white pixels that are not land, inside rho = r/R <= 0.88 (the hazy limb is cut)
  area     orthographic weight 1/sqrt(1 - rho^2), and (6371 km / R_px)^2 per pixel at the centre
Ground truth: NSIDC Sea Ice Index v4 daily extent (G02135, public https, no login), same dates.

Music: lead (flute) = sea-ice extent, more ice = higher pitch. The audio runs `stretch` (1.5) times
slower than the movie, so a year lasts 18.25 s; the player slows the video to match (`video_rate`).
The bowed-glass band plays the 1981-2010 NSIDC median extent for the same day of year, so the gap
between the two lines is the ice that is missing compared with 30 years ago.
Controls: geophony (ocean, wind) = open water (ice deficit from the year's maximum), brightness =
extent (the "glassy" bright bed when there is more ice), biophony (low, <= 0.35) = open water in the
summer months (seabirds feed at the ice edge), anthrophony silent.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from ..data import catalog
from ..data.ingest import download
from ..render.synth import render_notes
from ..score.mapping import PitchMap, melody_from_curve
from .common import export_demo, iter_frames, movie_info, norm01, roundtrip, write_track, write_wav
from ..proof.fidelity import score

PAGE = 5046
BASE = "https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005046/"
MOVIE = BASE + "arctic_sea_ice_moving_two_year_w_dates.webm"
VIDEO_MP4 = BASE + "arctic_sea_ice_moving_two_year_w_dates.mp4"
NSIDC = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/data/N_seaice_extent_daily_v4.0.csv"
NSIDC_CLIM = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/data/N_seaice_extent_climatology_1981-2010_v4.0.csv"
DATE0 = dt.date(2021, 4, 16)          # burned-in date of frame 0
EARTH_KM = 6371.0
RHO_MAX = 0.88
GREENLAND_SEED = (-0.35, -0.32)
GREENLAND_ICE_KM2 = 1.71e6          # Greenland ice sheet area (km2)       # Greenland's centre, (x, y) / R from the disc centre
LABEL_BOX = (0.035, 0.065, 0.22, 0.125)  # burned-in date (x0, y0, x1, y1) as a fraction of the frame


def find_disc(frame: np.ndarray) -> tuple[float, float, float]:
    """Globe centre and radius (px) from one frame: rows/columns with many non-black pixels."""
    nb = frame.max(2) > 30
    h, w = nb.shape
    x0, y0, x1, y1 = LABEL_BOX
    nb[: int(y1 * h) + 4, : int(x1 * w) + 4] = False     # date label
    rows = np.where(nb.sum(1) > 20)[0]
    cols = np.where(nb.sum(0) > 20)[0]
    cy, cx = (rows[0] + rows[-1]) / 2, (cols[0] + cols[-1]) / 2
    R = ((rows[-1] - rows[0]) + (cols[-1] - cols[0])) / 4
    return float(cx), float(cy), float(R)


def perspective_weight(rho: np.ndarray, D: float) -> np.ndarray:
    """Ground area per unit image area (Earth radii^2 per limb-radius^2) for a pinhole camera at
    distance D (Earth radii) looking at the centre; rho = image radius / limb radius.
    D -> infinity gives the orthographic 1/cos(gamma)."""
    tl = 1.0 / np.sqrt(D * D - 1)
    u = np.linspace(1e-4, 0.9999, 4000)
    th = np.arctan(u * tl)
    tt = D * np.cos(th) - np.sqrt(np.clip(1 - (D * np.sin(th)) ** 2, 0, None))
    gam = np.arctan2(tt * np.sin(th), D - tt * np.cos(th))
    w = np.sin(gam) * np.gradient(gam, u) / u
    return np.interp(rho, u, w)


def classify(f: np.ndarray):
    x = f.astype(np.int16)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    water = (b > r + 35) & (r < 160) & (b > 60)
    white = (r >= 170) & (g >= 170) & (b >= 180)
    landc = ~water & ~white & (x.max(2) > 40) & (b <= np.maximum(r, g) + 5)
    return water, white, landc


def measure(movie: Path, width: int, cache: Path) -> dict:
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    whites, n_water, n_land, label_diff, nonblack = [], None, None, [], None
    prev_lab = None
    mid = None
    for i, f in enumerate(iter_frames(movie, width)):
        water, white, landc = classify(f)
        if n_water is None:
            n_water = np.zeros(white.shape, np.int32); n_land = np.zeros(white.shape, np.int32)
            nonblack = np.zeros(white.shape, bool)
        n_water += water; n_land += landc; nonblack |= f.max(2) > 25
        whites.append(np.packbits(white))
        h, w = white.shape
        x0, y0, x1, y1 = LABEL_BOX
        lab = f[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].astype(np.float32).mean(2)
        label_diff.append(np.nan if prev_lab is None else float(np.abs(lab - prev_lab).mean()))
        prev_lab = lab
        if i == 150:
            mid = f.copy()
    out = {"whites": np.stack(whites), "shape": np.array(white.shape), "n_water": n_water, "n_land": n_land,
           "nonblack": nonblack, "label_diff": np.array(label_diff), "mid": mid}
    np.savez_compressed(cache, **out)
    return out


def masks(m: dict, greenland_km2: float = GREENLAND_ICE_KM2):
    H, W = m["shape"]
    T = len(m["whites"])
    cx, cy, R = find_disc(m["mid"])
    yy, xx = np.mgrid[0:H, 0:W]
    rho = np.hypot(xx - cx, yy - cy) / R
    inside = rho <= RHO_MAX
    land = m["n_land"] >= max(3, int(0.02 * T))
    # Greenland ice sheet: never water, never land-coloured, white on every frame -> one component
    import cv2
    always_white = np.ones((H, W), bool)
    for pw in m["whites"][::10]:
        always_white &= np.unpackbits(pw)[: H * W].reshape(H, W).astype(bool)
    perm = always_white & (m["n_water"] == 0) & ~land & inside
    n, lab = cv2.connectedComponents(perm.astype(np.uint8), connectivity=4)
    sx, sy = int(cx + GREENLAND_SEED[0] * R), int(cy + GREENLAND_SEED[1] * R)
    win = lab[sy - 3:sy + 4, sx - 3:sx + 4]
    lid = np.bincount(win[win > 0]).argmax() if (win > 0).any() else 0
    gl = (lab == lid) if lid > 0 else np.zeros_like(perm)
    land_all = land | gl
    sea = inside & ~land_all
    # camera distance D (Earth radii): the one at which the Greenland ice sheet has its known area.
    # This uses no sea-ice data, so the NSIDC comparison stays an independent check.
    Ds = np.geomspace(1.05, 200, 400)
    gl_area = np.array([perspective_weight(rho[gl], D).sum() * (EARTH_KM / R) ** 2 for D in Ds])
    D = float(np.interp(greenland_km2, gl_area[::-1], Ds[::-1])) if gl.any() else 1e3
    w = perspective_weight(np.clip(rho, 0, 0.9999), D)
    km2_px = (EARTH_KM / R) ** 2
    return {"disc": (cx, cy, R), "rho": rho, "sea": sea, "land": land_all, "greenland": gl, "perm": perm,
            "w": w, "km2_px": km2_px, "seed": (sx, sy), "camera_D": D,
            "greenland_km2_ortho": float((1 / np.sqrt(1 - np.clip(rho[gl], 0, .999) ** 2)).sum() * km2_px)}


def extent_series(m: dict, g: dict) -> np.ndarray:
    H, W = m["shape"]
    wsea = (g["w"] * g["sea"]).ravel()
    ext = []
    for pw in m["whites"]:
        white = np.unpackbits(pw)[: H * W].astype(bool)
        ext.append(float(wsea[white].sum()) * g["km2_px"] / 1e6)
    return np.array(ext)


def nsidc(path_daily: Path, dates: list[dt.date]) -> np.ndarray:
    rows = {}
    for line in path_daily.read_text().splitlines()[2:]:
        p = [s.strip() for s in line.split(",")]
        try:
            rows[dt.date(int(p[0]), int(p[1]), int(p[2]))] = float(p[3])
        except (ValueError, IndexError):
            continue
    ks = sorted(rows)
    o = np.array([k.toordinal() for k in ks], float)
    v = np.array([rows[k] for k in ks])
    return np.interp([d.toordinal() for d in dates], o, v)   # fills the rare missing days


def climatology(path: Path, dates: list[dt.date]) -> np.ndarray:
    med = {}
    for line in path.read_text().splitlines()[2:]:
        p = [s.strip() for s in line.split(",")]
        try:
            med[int(p[0])] = float(p[5])      # 50th percentile, 1981-2010
        except (ValueError, IndexError):
            continue
    return np.array([med.get(min(d.timetuple().tm_yday, 365), np.nan) for d in dates])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--demo", type=Path, default=Path("demo"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--days-per-note", type=int, default=5)
    ap.add_argument("--stretch", type=float, default=1.5, help="audio seconds per movie second")
    a = ap.parse_args(argv)
    d = a.data / str(PAGE)
    out = a.out / str(PAGE)
    out.mkdir(parents=True, exist_ok=True)
    page = catalog.fetch(PAGE, a.data.parent / "catalog")
    movie = download(MOVIE, d / Path(MOVIE).name)
    info = movie_info(movie)
    m = measure(movie, a.width, d / f"whites_{a.width}.npz")
    g = masks(m)
    ext_all = extent_series(m, g)
    F = len(ext_all)
    movie_fps = 30.0 if not info["fps"] or info["fps"] > 100 else info["fps"]
    dates = [DATE0 + dt.timedelta(days=i) for i in range(F)]
    truth = nsidc(download(NSIDC, d / Path(NSIDC).name), dates)
    clim = climatology(download(NSIDC_CLIM, d / Path(NSIDC_CLIM).name), dates)
    gt = score(truth, ext_all)
    slope, icpt = np.polyfit(ext_all, truth, 1)
    gt_lin = score(truth, slope * ext_all + icpt)
    lab = m["label_diff"]
    label_changes = float(np.nanmean(lab[1:] > 1.0))

    # notes: one per `days_per_note` days (mean over the days), audio time = movie time * stretch
    k = a.days_per_note
    nN = F // k
    idx0 = np.arange(nN) * k
    ext = np.array([ext_all[i:i + k].mean() for i in idx0])
    clim_n = np.array([clim[i:i + k].mean() for i in idx0])
    truth_n = np.array([truth[i:i + k].mean() for i in idx0])
    t = idx0 / movie_fps * a.stretch
    duration = round(F / movie_fps * a.stretch, 3)
    ndates = [dates[i].isoformat() for i in idx0]

    lo = min(np.nanmin(ext), np.nanmin(clim_n))
    hi = max(np.nanmax(ext), np.nanmax(clim_n))
    pad = 0.04 * (hi - lo)
    # F major, F4..F6: two octaves (15 steps) over the extent range
    lead_pm = PitchMap(round(lo - pad, 2), round(hi + pad, 2), low=65, span=24, root=5, scale="major")
    band_pm = PitchMap(lead_pm.vmin, lead_pm.vmax, low=lead_pm.low - 24, span=24, root=5, scale="major")
    lead = melody_from_curve(t, ext, lead_pm, "lead extent", step=None)
    for n in lead:
        j = int(round(n.start / a.stretch * movie_fps / k))
        n.source = {"date": ndates[min(j, nN - 1)], "series": "Arctic sea-ice extent (movie pixels)", "units": "million km2"}
    glass = melody_from_curve(t, clim_n, band_pm, "band 1981-2010 median", step=None, vel_range=(40, 70))
    stems = {"lead_extent": render_notes(lead, duration, "pluck", gain=.3),
             "bands": render_notes(glass, duration, "softpad", gain=.07)}
    groups = {"lead_extent": "melody", "bands": "bands"}

    fid, est = roundtrip(stems["lead_extent"], lead_pm, t, ext)
    rng = np.random.default_rng(0)
    shuf = ext[rng.permutation(nN)]
    ctrl = render_notes(melody_from_curve(t, shuf, lead_pm, "ctrl", step=None), duration, "pluck", gain=.3)
    write_wav(out / "control_shuffled.wav", ctrl / max(1e-9, np.abs(ctrl).max()) * 0.9)
    fid_ctrl, _ = roundtrip(ctrl, lead_pm, t, ext)
    fid_band, _ = roundtrip(stems["bands"], band_pm, t, clim_n)

    # controls (non-data voices; the form only cues these)
    open_w = norm01(np.nanmax(ext) - ext, 0, np.ptp(ext))
    doy = np.array([dt.date.fromisoformat(s).timetuple().tm_yday for s in ndates])
    summer = ((doy >= 152) & (doy <= 273)).astype(float)                       # Jun-Sep
    k5 = np.hanning(7); k5 /= k5.sum()
    summer = np.convolve(np.pad(summer, 3, mode="edge"), k5, mode="valid")
    curves = {"geophony": 0.25 + 0.75 * open_w,
              "anthrophony": np.zeros(nN),
              "biophony": 0.35 * open_w * summer,
              "brightness": norm01(ext, lead_pm.vmin, lead_pm.vmax)}
    pretty = lambda s: dt.date.fromisoformat(s).strftime("%-d %b %Y")
    desc = (f"Arctic sea ice, one day per frame, from {pretty(ndates[0])} to {dates[-1].strftime('%-d %b %Y')}: two "
            "full years of melt and freeze-up seen from above the North Pole (JAXA AMSR2, 15 % concentration). "
            "The flute is the sea-ice extent measured from the movie: more ice plays higher. It falls through "
            "each summer to a low in September and climbs back each winter, twice. A soft glass pad an octave "
            "and more below plays the 1981-2010 median extent for the same day, so the gap between them is the "
            "ice that is missing. Ocean and wind grow as water opens. The audio runs 1.5 times slower than the movie.")
    legend = [{"voice": "lead_extent", "meaning": f"sea-ice extent, F4 = {lead_pm.vmin:.1f}, F6 = {lead_pm.vmax:.1f} million km²"},
              {"voice": "bands", "meaning": "NSIDC 1981-2010 median extent for the same day of year, same scale two octaves lower"},
              {"voice": "geophony", "meaning": "control: open water (ice below the two-year maximum)"},
              {"voice": "biophony", "meaning": "control: summer open water, low (seabirds at the ice edge)"},
              {"voice": "brightness", "meaning": "control: sea-ice extent (a brighter, glassier bed with more ice)"},
              {"voice": "anthrophony", "meaning": "silent: not measured by this dataset"}]
    track = write_track(
        out, page_id=PAGE, title=page.get("title", "Daily Arctic Sea Ice, By Year"), video_url=VIDEO_MP4,
        duration_s=duration, units="million km²", description=desc, legend=legend, stems=stems,
        stem_groups=groups, notes=lead + glass, controls_t=t, curves=curves,
        fidelity={"r2": fid["r2"], "pearson_r": fid["pearson_r"], "mae": fid["mae"], "control_r2": fid_ctrl["r2"]},
        data_source={"music": "ice / water / land pixel classification of the SVS movie, area-weighted on the globe",
                     "underlying": "JAXA AMSR2 (GCOM-W1) 10 km sea-ice concentration, 3-day running minimum, 15 % threshold",
                     "ground_truth": "NSIDC Sea Ice Index v4 (G02135) daily Arctic extent",
                     "dates": [ndates[0], dates[-1].isoformat()]},
        mix_gains={"lead_extent": 1.0, "bands": 0.8},
        extra={"dates": ndates, "dates_t": [round(float(x), 3) for x in t],
               "video_rate": round(1 / a.stretch, 6), "video_alt_url": MOVIE,
               "time_axis": {"start": ndates[0], "end": dates[-1].isoformat(), "format": "day"},
               "caveat": "Extent is measured from the movie's pixels (blue-white ice vs ocean), not read from the "
                         "AMSR2 files. Against NSIDC's daily extent it tracks with r = %.3f." % gt["pearson_r"]})
    report = {
        "id": PAGE, "title": page.get("title"), "movie": MOVIE, "width": a.width, "movie_frames": F,
        "movie_duration_s": F / movie_fps, "stretch": a.stretch, "duration_s": duration,
        "dates": [dates[0].isoformat(), dates[-1].isoformat()], "date_check": {
            "frame0": DATE0.isoformat(), "label_changes_every_frame_fraction": label_changes,
            "read_by_eye": {"0.5 s": "2021-05-01", "5.0 s": "2021-09-13", "10.5 s": "2022-02-25", "24.3 s": "2023-04-15"}},
        "disc_px": g["disc"], "camera_distance_earth_radii": g["camera_D"],
        "greenland_ice_km2_if_orthographic": g["greenland_km2_ortho"], "rho_max": RHO_MAX, "km2_per_px_at_centre": g["km2_px"],
        "sea_px": int(g["sea"].sum()), "land_px": int(g["land"].sum()), "greenland_px": int(g["greenland"].sum()),
        "extent_range_mkm2": [float(ext_all.min()), float(ext_all.max())],
        "ground_truth": {"source": NSIDC, "daily": gt, "linear_fit": {"slope": slope, "intercept": icpt, **gt_lin},
                         "nsidc_range_mkm2": [float(truth.min()), float(truth.max())],
                         "notes_vs_nsidc": score(truth_n, ext)},
        "climatology": NSIDC_CLIM, "days_per_note": k, "notes": {"lead": len(lead), "bands": len(glass)},
        "lead_pitchmap": lead_pm.describe(), "band_pitchmap": band_pm.describe(),
        "audio_roundtrip": {"lead": fid, "bands_climatology": fid_band, "shuffled_control": fid_ctrl},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    np.savez_compressed(out / "series.npz", ext_all=ext_all, truth=truth, clim=clim, t=t, ext=ext)
    plot(out, m, g, ext_all, truth, clim, dates, t, ext, est, a.stretch, movie_fps)
    export_demo(out, a.demo / str(PAGE), names=("mix", "lead_extent", "control_shuffled"))
    print(json.dumps({k: report[k] for k in ("dates", "date_check", "disc_px", "extent_range_mkm2", "ground_truth",
                                             "audio_roundtrip")}, indent=1, default=float))


def plot(out, m, g, ext_all, truth, clim, dates, t, ext, est, stretch, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = m["shape"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].imshow(m["mid"]); ax[0, 0].set_title(f"frame 150 ({dates[150]})"); ax[0, 0].axis("off")
    white = np.unpackbits(m["whites"][150])[: H * W].reshape(H, W).astype(bool)
    vis = np.zeros((H, W, 3))
    vis[g["sea"]] = (0.1, 0.25, 0.6)
    vis[g["land"]] = (0.45, 0.4, 0.3)
    vis[g["greenland"]] = (0.7, 0.6, 0.4)
    vis[g["sea"] & white] = (1, 1, 1)
    ax[0, 1].imshow(vis); ax[0, 1].set_title("classes: sea (blue), ice (white), land (brown), Greenland (tan)")
    ax[0, 1].axis("off")
    ax[1, 0].plot(dates, truth, label="NSIDC Sea Ice Index (daily)")
    ax[1, 0].plot(dates, ext_all, label="measured from the movie")
    ax[1, 0].plot(dates, clim, "--", label="NSIDC 1981-2010 median")
    ax[1, 0].set_ylabel("million km²"); ax[1, 0].legend(); ax[1, 0].set_title("Arctic sea-ice extent")
    ax[1, 1].plot(t, ext, label="data (5-day means)"); ax[1, 1].plot(t, est, ".", label="recovered from audio")
    ax[1, 1].legend(); ax[1, 1].set_xlabel("audio s"); ax[1, 1].set_title("audio round trip")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
