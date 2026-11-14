"""Voice layer: the planet's three voices rendered from recorded corpora.

    from jukebox.voices import render_voice
    audio = render_voice("biophony", t, level, duration=60.0, brightness=b, pan=p,
                         tune_to=[50, 54, 57, 62], seed=0)      # (N, 2) float32 @ 44.1 kHz

Contract (other builders code against this; keep it stable)
  group       "geophony" | "anthrophony" | "biophony"
  t           (K,) seconds, increasing; control points for the curves below
  level       (K,) or scalar in 0..1. Drives grain density AND loudness. 0 -> exact silence.
  duration    output length in seconds; N = round(duration * sr)
  brightness  (K,) 0..1 or None. Target spectral-centroid rank (0 dark, 1 bright).
  pan         (K,) -1 (west / left) .. +1 (east / right) or None (centre).
  sr          output sample rate (corpora are 44.1 kHz; other rates are resampled).
  seed        same inputs + same seed -> bit-identical output.

Extras (keyword-only, all optional)
  tune_to     list[int] MIDI pitches, or [(start_s, end_s, [pitches]), ...] chord timeline.
              Adds a resonator bank so the noise "sings" the chord. tune_mix sets dry/wet.
  sources     restrict to corpus tags, e.g. ["hydrophone"], ["thunder"], ["birds", "frogs"].
  flatness    (K,) 0..1 target noisiness (default per voice).
  width       random per-grain pan spread around `pan` (default 0.3).
  density     multiplies the grain rate (default 1).
  gain        linear output gain (default 1). Level 1 sits around -20 dBFS RMS.

Levels are calibrated, not normalised per call, so stems from separate calls mix sensibly.
Peaks are soft-limited at about -1 dBFS.
"""
from __future__ import annotations

import numpy as np

from .grains import GROUPS, SR, STYLES, Corpus, corpora_root, granulate, load_corpus, thunder_hits
from .tonal import midi_hz, resonate

__all__ = ["render_voice", "render_events", "corpus_summary", "GROUPS", "SR", "load_corpus", "corpora_root"]


def _soft_limit(x: np.ndarray, knee: float = 0.8, ceil: float = 0.89) -> np.ndarray:
    a = np.abs(x)
    over = a > knee
    if not over.any():
        return x
    y = x.copy()
    room = ceil - knee
    y[over] = np.sign(x[over]) * (knee + room * np.tanh((a[over] - knee) / room))
    return y


def _fades(x: np.ndarray, sr: int, s: float = 0.02) -> np.ndarray:
    n = min(int(s * sr), len(x) // 2)
    if n > 1:
        r = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
        x[:n] *= r[:, None]
        x[-n:] *= r[::-1, None]
    return x


def _to_sr(x: np.ndarray, sr: int, n: int) -> np.ndarray:
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        x = resample_poly(x, sr // g, SR // g, axis=0).astype(np.float32)
    if len(x) < n:
        x = np.pad(x, ((0, n - len(x)), (0, 0)))
    return x[:n]


def render_voice(group: str, t: np.ndarray, level: np.ndarray, duration: float, *,
                 brightness: np.ndarray | None = None, pan: np.ndarray | None = None,
                 sr: int = SR, seed: int = 0, tune_to=None, tune_mix: float = 0.5, tune_q: float = 40.0,
                 sources=None, flatness: np.ndarray | None = None, width: float = 0.3,
                 density: float = 1.0, gain: float = 1.0) -> np.ndarray:
    """Render one voice. Returns (round(duration*sr), 2) float32. See module docstring."""
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    if duration < 0:
        raise ValueError("duration must be >= 0")
    n = int(round(duration * sr))
    t = np.atleast_1d(np.asarray(t, float))
    if t.size > 1 and np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        t = t[order]
        level = np.asarray(level, float)[order] if np.ndim(level) else level
        brightness = None if brightness is None or np.ndim(brightness) == 0 else np.asarray(brightness)[order]
        pan = None if pan is None or np.ndim(pan) == 0 else np.asarray(pan)[order]
        flatness = None if flatness is None or np.ndim(flatness) == 0 else np.asarray(flatness)[order]
    x = granulate(group, t, level, duration, brightness=brightness, pan=pan, flatness=flatness,
                  seed=seed, sources=sources, width=width, density=density)
    if not np.any(x):
        return np.zeros((n, 2), np.float32)
    if tune_to:
        x = resonate(x, SR, tune_to, mix=tune_mix, q=tune_q)
    x = _fades(x * np.float32(gain), SR)
    x = _soft_limit(x)
    return _to_sr(x.astype(np.float32), sr, n)


def render_events(times, intensity, duration: float, *, pan=None, group: str = "geophony",
                  sources=("thunder",), sr: int = SR, seed: int = 0, length: float = 2.5) -> np.ndarray:
    """One-shot hits at exact times (e.g. ISS lightning flashes -> thunder). (N, 2) float32."""
    n = int(round(duration * sr))
    x = thunder_hits(times, intensity, duration, pan=pan, seed=seed, sources=sources, group=group, length=length)
    return _to_sr(_soft_limit(x), sr, n)


def corpus_summary() -> dict:
    """Minutes / files / grains / minutes-by-tag for each voice corpus."""
    return {g: load_corpus(g).summary() for g in GROUPS}
