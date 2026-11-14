"""Gentle tonal layer: a resonator bank tuned to the music's pitch set.

Noisy grains (surf, traffic, insects) pass through narrow band-pass resonators centred on
each MIDI pitch and a few of its harmonics, so the texture "sings" the current chord and
sits under the melody instead of fighting it. Wet energy is matched to the dry signal and
blended with `mix`.

`tune_to` accepts either
  * list[int]                          one pitch set for the whole render, or
  * list[(start_s, end_s, list[int])]  a chord timeline; segments are cross-faded with
                                       raised-cosine weights that sum to one.
"""
from __future__ import annotations

import numpy as np

HARMONIC_GAINS = (1.0, 0.5, 0.3)


def midi_hz(p) -> np.ndarray:
    return 440.0 * 2 ** ((np.asarray(p, float) - 69) / 12)


def _bank(x: np.ndarray, sr: int, pitches, q: float) -> np.ndarray:
    from scipy.signal import iirpeak, sosfilt, tf2sos

    y = np.zeros_like(x, dtype=np.float64)
    for p in sorted(set(int(v) for v in pitches)):
        f0 = float(midi_hz(p))
        for h, g in enumerate(HARMONIC_GAINS, start=1):
            f = f0 * h
            if f < 30 or f > 0.45 * sr:
                continue
            # constant-ish bandwidth in cents: Q grows a little with harmonic number
            b, a = iirpeak(f, q * (1 + 0.25 * (h - 1)), fs=sr)
            y += g * sosfilt(tf2sos(b, a), x, axis=0)
    return y


def _segments(tune_to, duration: float):
    if not tune_to:
        return []
    first = tune_to[0]
    if isinstance(first, (int, np.integer, float)):
        return [(0.0, duration, list(tune_to))]
    segs = sorted((float(a), float(b), list(p)) for a, b, p in tune_to)
    return [s for s in segs if s[1] > s[0] and s[2]]


def resonate(x: np.ndarray, sr: int, tune_to, *, mix: float = 0.5, q: float = 40.0,
             xfade: float = 0.25) -> np.ndarray:
    """x: (N, 2) float -> (N, 2) float32, dry/wet blend of a pitch-tuned resonator bank."""
    if not tune_to or mix <= 0 or not np.any(x):
        return x.astype(np.float32)
    n = len(x)
    duration = n / sr
    segs = _segments(tune_to, duration)
    if not segs:
        return x.astype(np.float32)
    x64 = x.astype(np.float64)
    if len(segs) == 1:
        wet = _bank(x64, sr, segs[0][2], q)
    else:
        # partition-of-unity weights: segment i owns [start_i, start_{i+1}); raised-cosine
        # cross-fades centred on each boundary (narrowed so neighbouring fades never overlap)
        edges = np.array([s[0] for s in segs[1:]])
        gaps = np.diff(np.concatenate([[0.0], edges, [duration]]))
        xf = max(1.0 / sr, min(xfade, 0.9 * float(gaps.min())))

        def rise(u):  # 0 -> 1 across [-xf/2, +xf/2]
            return 0.5 - 0.5 * np.cos(np.pi * np.clip(u / xf + 0.5, 0, 1))

        wet = np.zeros_like(x64)
        ring = int(0.5 * sr)
        for i, (_, _, p) in enumerate(segs):
            t0 = edges[i - 1] - xf / 2 if i > 0 else 0.0
            t1 = edges[i] + xf / 2 if i < len(segs) - 1 else duration
            a0, a1 = max(0, int(t0 * sr)), min(n, int(np.ceil(t1 * sr)) + 1)
            if a1 <= a0:
                continue
            tt = np.arange(a0, a1) / sr
            w = np.ones(a1 - a0)
            if i > 0:
                w *= rise(tt - edges[i - 1])
            if i < len(segs) - 1:
                w *= 1 - rise(tt - edges[i])
            s0 = max(0, a0 - ring)  # start early so the resonators are ringing at the fade-in
            wet[a0:a1] += _bank(x64[s0:a1], sr, p, q)[a0 - s0:] * w[:, None]
    d_rms = np.sqrt((x64 ** 2).mean()) + 1e-12
    w_rms = np.sqrt((wet ** 2).mean()) + 1e-12
    wet *= d_rms / w_rms
    return ((1 - mix) * x64 + mix * wet).astype(np.float32)
