"""Corpus-based concatenative (granular) synthesis for the three voices.

Pipeline
  corpus   <root>/<group>/*.flac (see scripts/fetch_corpora.py), 44.1 kHz mono
  slice    analysis grains of GRAIN_S every HOP_S seconds, per file
  describe librosa RMS (dB), spectral centroid (Hz), spectral flatness per grain,
           cached in <root>/<group>/descriptors.npz (rebuilt when the files change)
  select   targets (loudness, brightness, noisiness) in per-corpus rank space [0,1]^3
           -> k nearest grains (scipy cKDTree) -> seeded random pick among them,
           skipping grains used in the last few picks
  render   Hann-windowed overlap-add, 100-400 ms grains, density compensation,
           equal-power pan per grain, audio-rate level envelope (level 0 = exact silence)

If no corpus is found for a group (e.g. on a laptop without the assets folder) a small
procedural stand-in corpus is synthesised so callers never crash; `Corpus.synthetic`
says when that happened.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 44100
GROUPS = ("geophony", "anthrophony", "biophony")
DEFAULT_ROOT = Path("/projects/e32706/omb8654/assets/corpora")
GRAIN_S = 0.30          # analysis grain length
HOP_S = 0.10            # analysis hop
N_FFT, FRAME_HOP = 2048, 512
DESC_VERSION = 2
FEATURE_WEIGHTS = np.array([1.0, 1.0, 0.6])  # loudness, brightness, flatness in the KD-tree
LEVEL_EXP = 0.6         # envelope = level ** LEVEL_EXP (grain choice + density supply the rest)


@dataclass(frozen=True)
class GroupStyle:
    dur: tuple[float, float]   # rendered grain length range (s)
    rate: float                # grains per second at level 1
    loud: tuple[float, float]  # loudness-rank target at level 0 -> level 1
    flat: float                # default flatness-rank target
    calib: float               # output gain so level 1 sits near -20 dBFS RMS


STYLES = {
    "geophony": GroupStyle(dur=(0.25, 0.40), rate=16.0, loud=(0.20, 0.95), flat=0.55, calib=1.06),
    "anthrophony": GroupStyle(dur=(0.20, 0.40), rate=14.0, loud=(0.25, 0.95), flat=0.50, calib=1.75),
    "biophony": GroupStyle(dur=(0.12, 0.35), rate=12.0, loud=(0.30, 0.97), flat=0.35, calib=1.48),
}


def corpora_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("JUKEBOX_CORPORA", DEFAULT_ROOT))


# ---------------------------------------------------------------------------- corpus
@dataclass
class Corpus:
    group: str
    audio: np.ndarray                 # int16 mono, all files concatenated
    files: list[str]
    tags: list[str]                   # one tag per file
    file_bounds: np.ndarray           # (n_files, 2) start/end sample of each file in `audio`
    g_start: np.ndarray               # (n_grains,) start sample of each analysis grain
    g_file: np.ndarray                # (n_grains,) file index
    rms_db: np.ndarray
    centroid: np.ndarray              # Hz
    flatness: np.ndarray
    synthetic: bool = False
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def seconds(self) -> float:
        return len(self.audio) / SR

    @property
    def grain_tags(self) -> np.ndarray:
        return np.asarray(self.tags)[self.g_file]

    def summary(self) -> dict:
        tg = {}
        for (a, b), t in zip(self.file_bounds, self.tags):
            tg[t] = tg.get(t, 0.0) + (b - a) / SR / 60
        return {"group": self.group, "minutes": round(self.seconds / 60, 2), "files": len(self.files),
                "grains": int(len(self.g_start)), "synthetic": self.synthetic,
                "minutes_by_tag": {k: round(v, 2) for k, v in sorted(tg.items())}}

    # ---- selection space --------------------------------------------------
    def space(self, sources: tuple[str, ...] | None = None):
        """Rank-normalised descriptor space + KD-tree for a (tag-filtered) sub-corpus."""
        key = sources
        if key in self._cache:
            return self._cache[key]
        from scipy.spatial import cKDTree

        idx = np.arange(len(self.g_start))
        if sources:
            keep = np.isin(self.grain_tags, list(sources))
            if not keep.any():
                raise ValueError(f"{self.group}: no grains tagged {sources}; have {sorted(set(self.tags))}")
            idx = idx[keep]
        # drop the quietest 5% (silence, gaps between calls) from the candidates
        loud = self.rms_db[idx]
        idx = idx[loud >= np.percentile(loud, 5)]

        def rank(v):
            r = np.empty(len(v))
            r[np.argsort(v, kind="stable")] = np.arange(len(v))
            return r / max(len(v) - 1, 1)

        X = np.stack([rank(self.rms_db[idx]), rank(np.log(self.centroid[idx] + 1)), rank(self.flatness[idx])], 1)
        tree = cKDTree(X * FEATURE_WEIGHTS)
        ref = float(np.median(10 ** (self.rms_db[idx] / 20)))
        self._cache[key] = (idx, X, tree, ref)
        return self._cache[key]

    def grain(self, i: int, n: int, jitter: int = 0) -> np.ndarray:
        """n samples of analysis grain i (shifted by `jitter` samples, clamped to its file)."""
        a, b = self.file_bounds[self.g_file[i]]
        s = int(np.clip(self.g_start[i] + jitter, a, max(a, b - n)))
        seg = self.audio[s: min(s + n, b)].astype(np.float32) / 32768.0
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        return seg


def _signature(files: list[Path]) -> str:
    h = hashlib.sha1(f"v{DESC_VERSION}|{GRAIN_S}|{HOP_S}|{N_FFT}|{FRAME_HOP}".encode())
    for f in files:
        st = f.stat()
        h.update(f"{f.name}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


def _describe(y: np.ndarray, sr: int = SR):
    """Per analysis-grain descriptors for one file."""
    import librosa

    y = y.astype(np.float32)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=FRAME_HOP, center=True)[0]
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=FRAME_HOP, center=True)) ** 2
    cen = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)[0]
    flat = librosa.feature.spectral_flatness(S=S, power=1.0)[0]
    n_g = max(1, int((len(y) / sr - GRAIN_S) / HOP_S) + 1)
    starts = (np.arange(n_g) * HOP_S * sr).astype(np.int64)
    fpg = int(round(GRAIN_S * sr / FRAME_HOP))
    out = np.zeros((n_g, 3))
    for k, s in enumerate(starts):
        f0 = s // FRAME_HOP
        sl = slice(f0, min(f0 + fpg + 1, len(rms)))
        p = rms[sl] ** 2
        w = p + 1e-12
        out[k] = (10 * np.log10(p.mean() + 1e-12), (cen[sl] * w).sum() / w.sum(), (flat[sl] * w).sum() / w.sum())
    return starts, out


def _manifest_tags(root: Path) -> dict:
    m = root / "manifest.json"
    if not m.exists():
        return {}
    try:
        return {e["file"]: e.get("tag", "misc") for e in json.loads(m.read_text())}
    except Exception:
        return {}


@lru_cache(maxsize=8)
def load_corpus(group: str, root: str | None = None) -> Corpus:
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    base = corpora_root(root)
    files = sorted((base / group).glob("*.flac")) + sorted((base / group).glob("*.wav"))
    tagmap = _manifest_tags(base)
    if tagmap:  # only use files whose provenance/licence is recorded in the manifest
        extra = [f.name for f in files if f"{group}/{f.name}" not in tagmap]
        if extra:
            warnings.warn(f"{group}: ignoring {len(extra)} file(s) not in manifest.json: {extra[:3]}", stacklevel=2)
        files = [f for f in files if f"{group}/{f.name}" in tagmap]
    if not files:
        warnings.warn(f"no corpus at {base / group}; using a procedural stand-in. Run scripts/fetch_corpora.py",
                      stacklevel=2)
        return _synthetic_corpus(group)
    import soundfile as sf

    chunks, bounds, tags, pos = [], [], [], 0
    for f in files:
        y, sr = sf.read(str(f), dtype="int16", always_2d=True)
        y = y.mean(1).astype(np.int16) if y.shape[1] > 1 else y[:, 0]
        if sr != SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, SR)
            y = np.clip(resample_poly(y.astype(np.float32), SR // g, sr // g), -32768, 32767).astype(np.int16)
        chunks.append(y)
        bounds.append((pos, pos + len(y)))
        pos += len(y)
        tags.append(tagmap.get(f"{group}/{f.name}", "misc"))
    audio = np.concatenate(chunks)
    bounds = np.asarray(bounds, np.int64)

    sig = _signature(files)
    cache = base / group / "descriptors.npz"
    d = None
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z["signature"]) == sig:
                d = {k: z[k] for k in z.files}
        except Exception:
            d = None
    if d is None:
        gs, gf, desc = [], [], []
        for i, (a, b) in enumerate(bounds):
            st, de = _describe(audio[a:b].astype(np.float32) / 32768.0)
            gs.append(st + a)
            gf.append(np.full(len(st), i))
            desc.append(de)
        desc = np.concatenate(desc)
        d = dict(signature=np.array(sig), g_start=np.concatenate(gs), g_file=np.concatenate(gf),
                 rms_db=desc[:, 0], centroid=desc[:, 1], flatness=desc[:, 2],
                 files=np.array([f.name for f in files]))
        try:
            np.savez(cache, **d)
        except OSError as e:
            warnings.warn(f"could not write descriptor cache {cache}: {e}")
    return Corpus(group, audio, [f.name for f in files], tags, bounds, d["g_start"].astype(np.int64),
                  d["g_file"].astype(np.int64), d["rms_db"].astype(float), d["centroid"].astype(float),
                  d["flatness"].astype(float))


def _synthetic_corpus(group: str, seconds: float = 90.0) -> Corpus:
    """Procedural stand-in so the API works without downloaded corpora."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(zlib.crc32(group.encode()))
    n = int(seconds * SR)
    t = np.arange(n) / SR
    if group == "geophony":   # surf-like swells of low-passed noise + rumbles
        x = sosfilt(butter(2, 900, fs=SR, output="sos"), rng.standard_normal(n))
        x *= 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 7.0)) ** 2
        x += 0.5 * sosfilt(butter(2, 120, fs=SR, output="sos"), rng.standard_normal(n))
    elif group == "anthrophony":  # 120 Hz hum family + band-limited traffic wash
        x = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((60, .2), (120, .5), (180, .15), (240, .2)))
        x = x + 0.6 * sosfilt(butter(2, [80, 2500], "bandpass", fs=SR, output="sos"), rng.standard_normal(n))
    else:  # chirps + cricket pulses
        x = 0.02 * rng.standard_normal(n)
        for s in rng.uniform(0, seconds - 1, int(seconds * 3)):
            L = int(rng.uniform(.05, .25) * SR)
            tt = np.arange(L) / SR
            f = rng.uniform(2000, 6000) * (1 + rng.uniform(-.4, .4) * tt / tt[-1])
            i = int(s * SR)
            x[i: i + L] += np.sin(2 * np.pi * np.cumsum(f) / SR) * np.hanning(L) * rng.uniform(.2, 1)
        x += 0.15 * np.sin(2 * np.pi * 4500 * t) * (np.sin(2 * np.pi * 15 * t) > 0.6)
    x = (x / np.abs(x).max() * 0.7 * 32767).astype(np.int16)
    starts = np.arange(0, n - int(GRAIN_S * SR), int(HOP_S * SR), dtype=np.int64)
    G = 4096  # descriptor window (shorter than the grain; enough for a stand-in)
    win = np.hanning(G).astype(np.float32)
    frames = np.stack([x[s: s + G] for s in starts]).astype(np.float32) / 32768
    P = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2 + 1e-12
    fr = np.fft.rfftfreq(G, 1 / SR)
    rms = 10 * np.log10((frames ** 2).mean(1) + 1e-12)
    cen = (P * fr).sum(1) / P.sum(1)
    flat = np.exp(np.log(P).mean(1)) / P.mean(1)
    return Corpus(group, x, ["synthetic"], ["synthetic"], np.array([[0, n]]), starts, np.zeros(len(starts), np.int64),
                  rms, cen, flat, synthetic=True)


# ---------------------------------------------------------------------------- render
def _interp(t: np.ndarray, v, at: np.ndarray, default: float) -> np.ndarray:
    if v is None:
        return np.full(len(at), default, float)
    v = np.asarray(v, float)
    if v.ndim == 0:
        return np.full(len(at), float(v))
    ok = np.isfinite(v)
    if not ok.any():
        return np.full(len(at), default, float)
    return np.interp(at, t[ok], v[ok])


def _smooth(x: np.ndarray, sr: int, tau: float = 0.03) -> np.ndarray:
    """Zero-phase one-pole smoothing (forward + backward) of an audio-rate control."""
    from scipy.signal import filtfilt
    if len(x) < 16:
        return x
    a = np.exp(-1.0 / (tau * sr))
    return filtfilt([1 - a], [1, -a], x, padlen=min(len(x) - 1, int(3 * tau * sr)))


def granulate(group: str, t, level, duration: float, *, brightness=None, pan=None, flatness=None,
              seed: int = 0, sources=None, width: float = 0.3, density: float = 1.0,
              corpus: Corpus | None = None) -> np.ndarray:
    """Core grain renderer at 44.1 kHz -> (N, 2) float32 (no tonal layer, no limiter)."""
    style = STYLES[group]
    c = corpus or load_corpus(group)
    n_out = int(round(duration * SR))
    out = np.zeros((n_out + int(0.5 * SR), 2), np.float32)
    t = np.atleast_1d(np.asarray(t, float))
    lvl_raw = np.atleast_1d(np.asarray(level, float))
    if lvl_raw.size == 1 and t.size != 1:
        lvl_raw = np.full(t.size, float(lvl_raw[0]))
    if n_out == 0 or t.size == 0 or not np.any(np.nan_to_num(lvl_raw) > 1e-4):
        return out[:n_out]
    lvl_raw = np.clip(np.nan_to_num(lvl_raw), 0, 1)

    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(group.encode())])
    srcs = tuple(sorted(sources)) if sources else None
    idx, X, tree, ref_amp = c.space(srcs)
    k = int(min(12, len(idx)))
    hop = int(HOP_S * SR)

    # ---- schedule onsets: rate follows level -----------------------------
    onsets, now = [], 0.0
    while now < duration:
        lv = float(np.interp(now, t, lvl_raw))
        r = style.rate * density * lv ** 0.7
        if r < 0.05:
            now += 0.05
            continue
        onsets.append(now)
        now += rng.uniform(0.6, 1.4) / r
    if not onsets:
        return out[:n_out]
    on = np.asarray(onsets)
    m = len(on)
    lv = _interp(t, lvl_raw, on, 0.0)
    br = _interp(t, brightness, on, 0.5)
    fl = _interp(t, flatness, on, style.flat)
    pn = _interp(t, pan, on, 0.0)
    dur = rng.uniform(*style.dur, m)
    rate = style.rate * density * np.maximum(lv, 1e-6) ** 0.7

    lo, hi = style.loud
    targ = np.stack([lo + (hi - lo) * lv, np.clip(br, 0, 1), np.clip(fl, 0, 1)], 1)
    targ = np.clip(targ + rng.normal(0, [0.04, 0.06, 0.08], (m, 3)), 0, 1)
    _, nn = tree.query(targ * FEATURE_WEIGHTS, k=k)
    nn = np.atleast_2d(nn).reshape(m, -1)

    recent: list[int] = []
    for j in range(m):
        cand = [g for g in nn[j] if g not in recent] or list(nn[j])
        chosen = int(cand[rng.integers(len(cand))])
        pick = int(idx[chosen])
        recent.append(chosen)
        if len(recent) > 8:
            recent.pop(0)
        L = int(dur[j] * SR)
        g = c.grain(pick, L, int(rng.integers(-hop // 2, hop // 2 + 1)))
        g_rms = float(np.sqrt((g ** 2).mean()) + 1e-6)
        gain = np.clip((ref_amp / g_rms) ** 0.7, 0.25, 4.0)         # partial loudness normalisation
        gain /= np.sqrt(max(1.0, rate[j] * dur[j]))                  # overlap (density) compensation
        g = g * np.hanning(L).astype(np.float32) * gain
        p = float(np.clip(pn[j] + rng.uniform(-width, width), -1, 1))
        lr = np.array([np.cos((p + 1) * np.pi / 4), np.sin((p + 1) * np.pi / 4)], np.float32)
        s = int(on[j] * SR)
        e = min(len(out), s + L)
        out[s:e] += g[: e - s, None] * lr

    out = out[:n_out]
    # audio-rate loudness envelope: level 0 -> exact silence, no zipper noise
    env = np.interp(np.arange(n_out) / SR, t, lvl_raw)
    env = np.clip(_smooth(env, SR), 0, 1)
    env[env < 1e-4] = 0.0
    env = env ** LEVEL_EXP
    out *= (env * style.calib / max(ref_amp, 1e-4) * 0.1).astype(np.float32)[:, None]
    return out


def thunder_hits(times, intensity, duration: float, *, pan=None, seed: int = 0, sources=("thunder",),
                 group: str = "geophony", length: float = 2.5) -> np.ndarray:
    """One-shot events (lightning flashes -> thunder): a loud corpus segment per event with a
    5 ms attack and an exponential tail. intensity in 0..1 picks louder grains and scales gain."""
    c = load_corpus(group)
    n_out = int(round(duration * SR))
    out = np.zeros((n_out, 2), np.float32)
    times = np.atleast_1d(np.asarray(times, float))
    if times.size == 0:
        return out
    inten = np.clip(np.broadcast_to(np.asarray(intensity, float), times.shape), 0, 1)
    pans = np.broadcast_to(np.asarray(0.0 if pan is None else pan, float), times.shape)
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 99])
    try:
        idx, X, tree, ref = c.space(tuple(sources) if sources else None)
    except ValueError:
        idx, X, tree, ref = c.space(None)
    L = int(length * SR)
    tt = np.arange(L) / SR
    env = np.minimum(1, tt / 0.005) * np.exp(-tt / (length / 3))
    for ti, a, p in zip(times, inten, pans):
        if a <= 0 or ti >= duration:
            continue
        _, nn = tree.query(np.array([0.7 + 0.3 * a, 0.3, 0.6]) * FEATURE_WEIGHTS, k=min(8, len(idx)))
        pick = int(idx[np.atleast_1d(nn)[rng.integers(np.size(nn))]])
        g = c.grain(pick, L) * env * a / max(ref, 1e-4) * 0.08
        lr = np.array([np.cos((p + 1) * np.pi / 4), np.sin((p + 1) * np.pi / 4)], np.float32)
        s = int(ti * SR)
        e = min(n_out, s + L)
        out[s:e] += (g[: e - s, None] * lr).astype(np.float32)
    return out
