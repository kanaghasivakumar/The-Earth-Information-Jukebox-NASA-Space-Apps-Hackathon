"""Round-trip fidelity: recover the data curve from the rendered audio and score it.

Pitch of the rendered melody stem is re-extracted with librosa.pyin, mapped back
through the PitchMap's linear range, and compared with the input curve (R², Pearson r).
A time-shuffled control is scored the same way against the *original* curve.
"""
from __future__ import annotations

import numpy as np


def recover_curve(audio: np.ndarray, sr: int, pm, times: np.ndarray, fmin=50.0, fmax=2000.0) -> np.ndarray:
    import librosa
    mono = audio.mean(1) if audio.ndim == 2 else audio
    f0, voiced, _ = librosa.pyin(mono, fmin=fmin, fmax=fmax, sr=sr, frame_length=4096, hop_length=512)
    ft = librosa.times_like(f0, sr=sr, hop_length=512)
    midi = librosa.hz_to_midi(f0)
    # sample at the data timestamps (median over each step), map back to data units
    est = np.full(len(times), np.nan)
    edges = np.r_[times, times[-1] + (times[-1] - times[-2] if len(times) > 1 else 1)]
    for i in range(len(times)):
        m = (ft >= edges[i]) & (ft < edges[i + 1]) & np.isfinite(midi)
        if m.any():
            est[i] = np.median(midi[m])
    x = (est - pm.low) / pm.span
    if pm.polarity < 0:
        x = 1 - x
    return pm.vmin + x * (pm.vmax - pm.vmin)


def score(truth: np.ndarray, est: np.ndarray) -> dict:
    m = np.isfinite(truth) & np.isfinite(est)
    if m.sum() < 3:
        return {"n": int(m.sum()), "r2": float("nan"), "pearson_r": float("nan"), "mae": float("nan")}
    t, e = truth[m], est[m]
    ss_res = ((t - e) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return {"n": int(m.sum()), "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "pearson_r": float(np.corrcoef(t, e)[0, 1]), "mae": float(np.abs(t - e).mean())}
