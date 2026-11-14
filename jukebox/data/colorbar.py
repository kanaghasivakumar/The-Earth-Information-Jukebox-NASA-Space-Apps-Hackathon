"""Invert colormapped SVS frames back to physical values using the published colorbar.

A colorbar image is sampled along its long axis to build a lookup table
(LUT) of CIELAB colours -> data values. Each frame pixel is assigned the value
of its nearest LUT colour; pixels whose nearest colour is further than
`max_delta_e` are treated as "no data" (land, labels, cloud, flow-line glyphs).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (..., 3) -> float CIELAB (..., 3) with L in [0, 100]."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    shape = rgb.shape
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).astype(np.float32).reshape(shape)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


@dataclass
class Colorbar:
    lab: np.ndarray      # (K, 3) LUT colours in CIELAB
    values: np.ndarray   # (K,) data value for each LUT entry
    units: str = ""

    @classmethod
    def from_image(cls, rgb: np.ndarray, vmin: float, vmax: float, *, units: str = "",
                   bbox: tuple[int, int, int, int] | None = None, samples: int = 256,
                   reverse: bool = False) -> "Colorbar":
        """Build a LUT from a colorbar image.

        bbox = (x0, y0, x1, y1) crops to the gradient strip (exclude tick labels).
        The long axis is the sampling axis; the short axis is median-averaged.
        """
        img = rgb if bbox is None else rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        horizontal = img.shape[1] >= img.shape[0]
        strip = np.median(img, axis=0 if horizontal else 1).astype(np.uint8)  # (L, 3)
        if not horizontal:
            strip = strip[::-1]  # vertical bars put the maximum at the top
        idx = np.linspace(0, len(strip) - 1, samples).round().astype(int)
        lut_rgb = strip[idx]
        values = np.linspace(vmin, vmax, samples)
        if reverse:
            values = values[::-1]
        return cls(to_lab(lut_rgb), values, units)

    @classmethod
    def from_anchors(cls, anchors: list[tuple[float, tuple[int, int, int]]], *, units: str = "",
                     samples: int = 256) -> "Colorbar":
        """Build a LUT by linear interpolation between (value, RGB) anchors."""
        anchors = sorted(anchors)
        v = np.array([a[0] for a in anchors], float)
        c = np.array([a[1] for a in anchors], float)
        values = np.linspace(v[0], v[-1], samples)
        rgb = np.stack([np.interp(values, v, c[:, k]) for k in range(3)], -1).round().astype(np.uint8)
        return cls(to_lab(rgb), values, units)

    def invert(self, rgb: np.ndarray, max_delta_e: float = 12.0, chunk: int = 65536,
               l_weight: float = 1.0, min_chroma: float = 0.0, shade_invariant: bool = False):
        """Return (values, valid_mask, delta_e) for an RGB image (H, W, 3).

        shade_invariant=True handles renders where the colormap is dimmed or blended
        toward a dark background (e.g. SVS flow lines): pixels are compared by hue
        direction in the a*b* plane (chroma normalised), lightness is down-weighted
        by `l_weight`, and unsaturated pixels (chroma < min_chroma: land, ocean
        background, labels) are rejected.
        """
        lab = to_lab(rgb).reshape(-1, 3)
        lut = self.lab
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        if shade_invariant:
            def norm_ab(x):
                c = np.hypot(x[:, 1], x[:, 2])[:, None] + 1e-6
                return np.c_[x[:, :1] * l_weight / 100.0 * 50.0, x[:, 1:] / c * 50.0]
            lab, lut = norm_ab(lab), norm_ab(lut)
        elif l_weight != 1.0:
            w = np.array([l_weight, 1.0, 1.0], np.float32)
            lab, lut = lab * w, lut * w
        best = np.empty(len(lab), np.int32)
        dist = np.empty(len(lab), np.float32)
        for s in range(0, len(lab), chunk):
            d = np.linalg.norm(lab[s:s + chunk, None, :] - lut[None, :, :], axis=-1)
            best[s:s + chunk] = d.argmin(1)
            dist[s:s + chunk] = d.min(1)
        values = self.values[best].reshape(rgb.shape[:2])
        valid = ((dist <= max_delta_e) & (chroma >= min_chroma)).reshape(rgb.shape[:2])
        return np.where(valid, values, np.nan), valid, dist.reshape(rgb.shape[:2])

    @property
    def resolution(self) -> float:
        """Value step between adjacent LUT entries (a lower bound on inversion error)."""
        return float(np.abs(np.diff(self.values)).mean())


def auto_bbox(rgb: np.ndarray, sat_thresh: int = 60, frac: float = 0.6) -> tuple[int, int, int, int]:
    """Find the gradient strip in a colorbar image: the largest block of saturated pixels.

    Tick labels and backgrounds are mostly unsaturated (white/grey/black), the strip is not.
    """
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3], np.uint8), cv2.COLOR_RGB2HSV)
    sat = (hsv[..., 1] > sat_thresh) & (hsv[..., 2] > 40)
    rows = np.where(sat.mean(1) > frac * sat.mean(1).max())[0]
    cols = np.where(sat.mean(0) > frac * sat.mean(0).max())[0]
    if rows.size == 0 or cols.size == 0:
        return 0, 0, rgb.shape[1], rgb.shape[0]
    # keep the longest contiguous run on each axis
    def run(ix):
        splits = np.split(ix, np.where(np.diff(ix) > 1)[0] + 1)
        r = max(splits, key=len)
        return int(r[0]), int(r[-1]) + 1
    y0, y1 = run(rows)
    x0, x1 = run(cols)
    # shave a 1px border to avoid anti-aliased edges
    return x0 + 1, y0 + 1, max(x0 + 2, x1 - 1), max(y0 + 2, y1 - 1)


@dataclass
class Palette:
    """Categorical palette: nearest-colour classification (e.g. aerosol types)."""
    names: list[str]
    lab: np.ndarray

    @classmethod
    def from_rgb(cls, mapping: dict[str, tuple[int, int, int]]) -> "Palette":
        names = list(mapping)
        return cls(names, to_lab(np.array([mapping[n] for n in names], np.uint8)))

    def classify(self, rgb: np.ndarray, max_delta_e: float = 25.0):
        lab = to_lab(rgb).reshape(-1, 3)
        d = np.linalg.norm(lab[:, None, :] - self.lab[None], axis=-1)
        cls_idx = d.argmin(1)
        cls_idx[d.min(1) > max_delta_e] = -1
        return cls_idx.reshape(rgb.shape[:2])

    def fractions(self, rgb: np.ndarray, max_delta_e: float = 25.0) -> dict[str, float]:
        c = self.classify(rgb, max_delta_e)
        n = max(1, c.size)
        return {name: float((c == i).sum()) / n for i, name in enumerate(self.names)}
