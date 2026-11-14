"""Georeference an SVS flat map (unknown crop / projection) against an exact equirectangular map.

SVS EIC renders (5313 night lights, 5669 aerosols) are flat maps whose longitude origin and
latitude scaling are not documented. We use the 5011 Science-On-a-Sphere frame (exactly
equirectangular, 4096x2048) as the reference land mask and fit, from many local coastline
template matches:

    X = x0 + lon * kx              (linear in longitude)
    Y = poly3(lat)                 (cubic in latitude: equirect / Miller / equal-area all fit)

    python -m jukebox.datasets.georef --target data/raw/5313/full/f01.png --crop 0,0,3840,2160 \
        --target-thresh 2 --out data/raw/5313/georef.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ref_land_mask(ref_rgb: np.ndarray) -> np.ndarray:
    """5011 gray-earth frame -> land mask (per-row ocean level, so the polar shading cancels)."""
    g = ref_rgb.astype(np.float32).mean(2)
    ocean = np.percentile(g, 20, axis=1, keepdims=True)
    return (g - ocean) > 4


class GeoFit:
    def __init__(self, x0, kx, ycoef, w, h):
        self.x0, self.kx, self.ycoef, self.w, self.h = x0, kx, np.asarray(ycoef), w, h
        lat = np.linspace(-90, 90, 3601)
        y = np.polyval(self.ycoef, lat)
        o = np.argsort(y)
        self._ys, self._lats = y[o], lat[o]

    def xy(self, lon, lat):
        lon = np.asarray(lon, float)
        return self.x0 + lon * self.kx, np.polyval(self.ycoef, np.asarray(lat, float))

    def lon(self, x):
        return (np.asarray(x, float) - self.x0) / self.kx

    def lat(self, y):
        return np.interp(np.asarray(y, float), self._ys, self._lats, left=np.nan, right=np.nan)

    def grids(self):
        """Per-column longitude and per-row latitude (pixel centres) of the target image."""
        return self.lon(np.arange(self.w) + 0.5), self.lat(np.arange(self.h) + 0.5)

    def to_dict(self):
        return {"x0": float(self.x0), "kx": float(self.kx), "ycoef": [float(c) for c in self.ycoef],
                "w": self.w, "h": self.h, "model": "X = x0 + kx*lon ; Y = polyval(ycoef, lat)"}

    @classmethod
    def from_dict(cls, d):
        return cls(d["x0"], d["kx"], d["ycoef"], d["w"], d["h"])


def fit(target_mask: np.ndarray, ref_mask: np.ndarray, init: dict, *, half_lon=12, half_lat=7,
        search_px=None, yscales=(0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4), min_corr=0.45):
    """Local template matches of ref coastlines inside the target -> (lon, lat, X, Y) pairs -> fit."""
    import cv2
    H, W = target_mask.shape
    Hr, Wr = ref_mask.shape
    tm = target_mask.astype(np.float32)
    kx, x0 = init["kx"], init["x0"]
    ky, y0 = init["ky"], init["y0"]  # initial linear guess Y = y0 - ky*lat
    search_px = search_px or int(0.06 * H)
    pairs = []
    for lat in np.arange(-55, 71, 5.0):
        for lon in np.arange(-175, 176, 7.5):
            # reference patch (equirect)
            r0 = int((90 - (lat + half_lat)) / 180 * Hr); r1 = int((90 - (lat - half_lat)) / 180 * Hr)
            c0 = int((lon - half_lon + 180) / 360 * Wr); c1 = int((lon + half_lon + 180) / 360 * Wr)
            if c0 < 0 or c1 > Wr or r0 < 0 or r1 > Hr:
                continue
            ref = ref_mask[r0:r1, c0:c1].astype(np.float32)
            f = ref.mean()
            if f < 0.15 or f > 0.85:
                continue
            best = None
            for s in yscales:
                pw = int(round(2 * half_lon * kx))
                ph = int(round(2 * half_lat * ky * s))
                if pw < 8 or ph < 8:
                    continue
                tpl = cv2.resize(ref, (pw, ph), interpolation=cv2.INTER_AREA)
                Xc, Yc = x0 + lon * kx, y0 - lat * ky
                xa, ya = int(Xc - pw / 2 - search_px), int(Yc - ph / 2 - search_px)
                xb, yb = int(Xc + pw / 2 + search_px), int(Yc + ph / 2 + search_px)
                xa, ya, xb, yb = max(0, xa), max(0, ya), min(W, xb), min(H, yb)
                win = tm[ya:yb, xa:xb]
                if win.shape[0] <= ph or win.shape[1] <= pw:
                    continue
                res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                if best is None or mx > best[0]:
                    best = (mx, xa + loc[0] + pw / 2, ya + loc[1] + ph / 2, s)
            if best and best[0] >= min_corr:
                pairs.append((lon, lat, best[1], best[2], best[0], best[3]))
    P = np.array(pairs)
    if len(P) < 10:
        raise RuntimeError(f"only {len(P)} matches")
    # robust fits (two passes, drop >3 MAD residuals)
    keep = np.ones(len(P), bool)
    for _ in range(3):
        bx = np.polyfit(P[keep, 0], P[keep, 2], 1)
        by = np.polyfit(P[keep, 1], P[keep, 3], 3)
        rx = P[:, 2] - np.polyval(bx, P[:, 0])
        ry = P[:, 3] - np.polyval(by, P[:, 1])
        r = np.hypot(rx, ry)
        mad = np.median(np.abs(r[keep])) + 1e-6
        keep = r < 3 * 1.4826 * mad + 2
    g = GeoFit(bx[1], bx[0], by, W, H)
    stats = {"matches": int(len(P)), "inliers": int(keep.sum()),
             "rms_px_x": float(np.sqrt((rx[keep] ** 2).mean())), "rms_px_y": float(np.sqrt((ry[keep] ** 2).mean())),
             "deg_per_px_lon": float(1 / bx[0])}
    return g, stats, P, keep


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 of the map inside the target image")
    ap.add_argument("--work-width", type=int, default=1920)
    ap.add_argument("--target-thresh", type=float, default=2.0, help="land = mean RGB > thresh")
    ap.add_argument("--target-mode", default="bright", choices=["bright", "gray"])
    ap.add_argument("--ref", type=Path, default=Path("data/raw/5011/ref_equirect.png"))
    ap.add_argument("--init", default="-169.5,0.09375,80,0.071", help="lon0,deg_per_px_x,lat0,deg_per_px_y at full res")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    img = cv2.imread(str(a.target), cv2.IMREAD_COLOR)[..., ::-1]
    if a.crop:
        x0, y0, x1, y1 = map(int, a.crop.split(","))
        img = img[y0:y1, x0:x1]
    full_w, full_h = img.shape[1], img.shape[0]
    sc = a.work_width / full_w
    small = cv2.resize(img, (a.work_width, int(round(full_h * sc))), interpolation=cv2.INTER_AREA)
    if a.target_mode == "bright":
        tmask = small.mean(2) > a.target_thresh
    else:  # unsaturated mid-gray land (aerosol maps): low chroma, mid value
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        tmask = (hsv[..., 1] < 40) & (hsv[..., 2] > a.target_thresh) & (hsv[..., 2] < 200)
    ref = cv2.imread(str(a.ref), cv2.IMREAD_COLOR)[..., ::-1]
    rmask = ref_land_mask(cv2.resize(ref, (2048, 1024), interpolation=cv2.INTER_AREA))
    lon0, dx, lat0, dy = map(float, a.init.split(","))
    init = {"kx": sc / dx, "x0": -lon0 * sc / dx, "ky": sc / dy, "y0": lat0 * sc / dy}
    g, stats, P, keep = fit(tmask, rmask, init)
    # back to full-res pixel units
    full = GeoFit(g.x0 / sc, g.kx / sc, g.ycoef / sc, full_w, full_h)
    lon_g, lat_g = full.grids()
    doc = {**full.to_dict(), "stats": stats, "target": str(a.target), "crop": a.crop,
           "lat_top": float(np.nanmax(lat_g)), "lat_bottom": float(np.nanmin(lat_g)),
           "lon_left": float(lon_g[0]), "lon_right": float(lon_g[-1])}
    a.out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))
    # diagnostic plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].imshow(tmask, cmap="gray")
    ax[0].plot(P[keep, 2], P[keep, 3], "g+"); ax[0].plot(P[~keep, 2], P[~keep, 3], "rx")
    lats = np.linspace(-80, 85, 100)
    ax[1].plot(P[keep, 1], P[keep, 3] / sc, "g.", label="matches"); ax[1].plot(lats, full.xy(0, lats)[1], "k-", label="cubic fit")
    ax[1].set_xlabel("lat"); ax[1].set_ylabel("row (full res)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(a.out.with_suffix(".png"), dpi=90)


if __name__ == "__main__":
    main()
