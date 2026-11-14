"""Per-frame data features from inverted value fields (equirectangular grids).

All functions take a value field `v` of shape (H, W) with NaN for no-data,
where rows run north (row 0) to south and columns west to east.
"""
from __future__ import annotations

import numpy as np

# Six latitude bands, north to south (Planetary Bands, Warming World style)
BANDS = [(90, 60), (60, 30), (30, 0), (0, -30), (-30, -60), (-60, -90)]


def lat_lon_grid(h: int, w: int):
    lat = 90 - (np.arange(h) + 0.5) * 180 / h
    lon = -180 + (np.arange(w) + 0.5) * 360 / w
    return lat, lon


def area_weights(h: int) -> np.ndarray:
    lat, _ = lat_lon_grid(h, 1)
    return np.cos(np.deg2rad(lat))


def weighted_mean(v: np.ndarray, rows: slice | np.ndarray | None = None) -> float:
    w = area_weights(v.shape[0])[:, None] * np.ones_like(v)
    if rows is not None:
        v, w = v[rows], w[rows]
    m = np.isfinite(v)
    return float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")


def zonal_means(v: np.ndarray, bands=BANDS) -> np.ndarray:
    lat, _ = lat_lon_grid(*v.shape)
    return np.array([weighted_mean(v, (lat <= n) & (lat > s)) for n, s in bands])


def frame_features(v: np.ndarray) -> dict:
    finite = v[np.isfinite(v)]
    return {
        "mean": weighted_mean(v),
        "p10": float(np.percentile(finite, 10)) if finite.size else np.nan,
        "p90": float(np.percentile(finite, 90)) if finite.size else np.nan,
        "std": float(finite.std()) if finite.size else np.nan,
        "coverage": float(np.isfinite(v).mean()),
        "zonal": zonal_means(v),
        "lon_profile": np.nanmean(v, axis=0),  # west->east, for panning / sweeps
    }


def sequence_features(fields: np.ndarray, timestamps: np.ndarray) -> dict:
    """fields: (T, H, W). Returns arrays keyed by feature name (T, ...)."""
    rows = [frame_features(f) for f in fields]
    out = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    out["timestamps"] = np.asarray(timestamps, float)
    mean = out["mean"]
    out["change"] = np.r_[0.0, np.abs(np.diff(mean))]
    # compatibility with eic-vae/music/composer.py controls (normalized 0..1)
    out["activity"] = _norm(out["change"])
    out["movement"] = out["activity"]
    out["flow_magnitude"] = out["activity"]
    return out


def detect_flashes(prev: np.ndarray, cur: np.ndarray, thresh: float = 60.0, min_px: int = 1):
    """Transient bright spots (e.g. lightning) between two grayscale frames.

    Returns list of (row, col, intensity). Uses connected components on the
    positive brightness difference.
    """
    import cv2
    diff = cur.astype(np.int16) - prev.astype(np.int16)
    mask = (diff > thresh).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            r, c = int(round(cent[i][1])), int(round(cent[i][0]))
            out.append((r, c, float(diff[r, c])))
    return out


def _norm(x, lo=5, hi=95):
    x = np.asarray(x, float)
    f = np.isfinite(x)
    if not f.any():
        return np.zeros_like(x)
    a, b = np.percentile(x[f], [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1) if b > a else np.zeros_like(x)
