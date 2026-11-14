"""Shade-tolerant colorbar inversion for lit 3D renders (helper; colorbar.py is unchanged).

On a lit globe each pixel is roughly  rgb = shade * colormap(value)  with shade in (0, 1.2].
CIELAB nearest-neighbour (colorbar.Colorbar.invert) then reads darker = lower value whenever
the colormap's lightness rises with value (GEDI: brown -> green -> cream). Matching in RGB
*chromaticity* (r, g) / (r + g + b) cancels the shade factor; lightness is kept only as a
weak tie-breaker (`l_weight`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def chromaticity(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rgb, np.float32).reshape(-1, 3)
    s = x.sum(1, keepdims=True) + 1e-3
    return x[:, :2] / s, (s[:, 0] / (3 * 255.0))


@dataclass
class ChromaColorbar:
    rgb: np.ndarray       # (K, 3) LUT colours
    values: np.ndarray    # (K,)
    units: str = ""

    @classmethod
    def from_image(cls, rgb: np.ndarray, vmin: float, vmax: float, bbox, samples: int = 128, units: str = ""):
        x0, y0, x1, y1 = bbox
        img = rgb[y0:y1, x0:x1].astype(np.float32)
        horizontal = img.shape[1] >= img.shape[0]
        strip = np.median(img, axis=0 if horizontal else 1)
        if not horizontal:
            strip = strip[::-1]
        idx = np.linspace(0, len(strip) - 1, samples).round().astype(int)
        return cls(strip[idx], np.linspace(vmin, vmax, samples), units)

    def invert(self, rgb: np.ndarray, max_dist: float = 4.0, min_bright: float = 0.12,
               l_weight: float = 0.15, chunk: int = 65536):
        """Return (values, valid, dist) for pixels `rgb` (..., 3).

        dist = 100 * |chromaticity difference| + l_weight * 100 * |brightness difference|.
        Pixels darker than `min_bright` (mean channel / 255) are invalid: their chromaticity is noise.
        """
        shape = rgb.shape[:-1]
        c, lum = chromaticity(rgb)
        lc, llum = chromaticity(self.rgb)
        f = np.c_[c * 100, lum[:, None] * 100 * l_weight]
        lf = np.c_[lc * 100, llum[:, None] * 100 * l_weight]
        best = np.empty(len(f), np.int32)
        dist = np.empty(len(f), np.float32)
        for s in range(0, len(f), chunk):
            d = np.linalg.norm(f[s:s + chunk, None, :] - lf[None], axis=-1)
            best[s:s + chunk] = d.argmin(1)
            dist[s:s + chunk] = d.min(1)
        valid = (dist <= max_dist) & (lum >= min_bright)
        vals = np.where(valid, self.values[best], np.nan)
        return vals.reshape(shape), valid.reshape(shape), dist.reshape(shape)

    @property
    def resolution(self) -> float:
        return float(np.abs(np.diff(self.values)).mean())
