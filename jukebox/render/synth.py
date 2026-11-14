"""Small dependency-free additive synth so the pipeline always produces audio.

FluidSynth + a General MIDI SoundFont replaces this for the final render;
this keeps M1 runnable anywhere (CPU node, laptop).
"""
from __future__ import annotations

import numpy as np

SR = 44100

TIMBRES = {
    # harmonic amplitudes, attack, release (s)
    "pad": ([1, .5, .25, .12, .06], .25, .6),
    "pluck": ([1, .6, .3, .2, .1, .05], .005, .35),
    "bell": ([1, 0, .4, 0, .25, 0, .1], .002, 1.2),
    "drum": (None, .001, .12),
}


def midi_hz(p):
    return 440.0 * 2 ** ((np.asarray(p, float) - 69) / 12)


def render_notes(notes, duration: float, timbre: str = "pad", sr: int = SR, gain: float = .25) -> np.ndarray:
    """notes: iterable with .start .end .pitch .velocity .pan -> stereo float32 (N, 2)."""
    harm, att, rel = TIMBRES[timbre]
    out = np.zeros((int((duration + 2) * sr), 2), np.float32)
    rng = np.random.default_rng(0)
    for n in notes:
        s = int(n.start * sr)
        L = int((n.end - n.start + rel) * sr)
        if L <= 0 or s >= len(out):
            continue
        t = np.arange(L) / sr
        if harm is None:  # noise burst through a crude low-pass
            sig = rng.standard_normal(L)
            sig = np.convolve(sig, np.ones(8) / 8, mode="same")
        else:
            f = midi_hz(n.pitch)
            sig = sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(harm) if a and f * (k + 1) < sr / 2)
        env = np.minimum(1, t / max(att, 1e-4))
        held = n.end - n.start
        env *= np.where(t > held, np.exp(-(t - held) / (rel / 3)), 1.0)
        if timbre in ("pluck", "bell", "drum"):
            env *= np.exp(-t / (rel / 2))
        sig = sig * env * (n.velocity / 127) * gain
        pan = float(getattr(n, "pan", 0.0))
        lr = np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)], np.float32)
        e = min(len(out), s + L)
        out[s:e] += (sig[: e - s, None] * lr).astype(np.float32)
    return out[: int(duration * sr) + int(rel * sr)]


def mix(*stems, gains=None) -> np.ndarray:
    n = max(len(s) for s in stems)
    gains = gains or [1.0] * len(stems)
    out = np.zeros((n, 2), np.float32)
    for s, g in zip(stems, gains):
        out[: len(s)] += s * g
    peak = np.abs(out).max()
    return out / peak * 0.9 if peak > 0 else out
