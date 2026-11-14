"""Shared helpers for the non-continuous datasets (5313, 5011, 5669, 5058).

Writes the track contract the web builder and integrator read:

    outputs/<id>/track.json      metadata, legend, stem list, fidelity, data source
    outputs/<id>/timeline.json   every Note with the data value it encodes
    outputs/<id>/controls.json   {"t": [...], "curves": {geophony, anthrophony, biophony, brightness}} in 0..1
    outputs/<id>/<stem>.wav      one WAV per stem (+ mix.wav)
    outputs/<id>/report.json     detection / mapping details
    outputs/<id>/overview.png    written by each dataset module

The three-voice stems rendered here are *placeholders* built from the control curves
(brown-noise rumble, mains hum + traffic hiss, bird/insect chirps). The voice builder's
`render_voice(group, t, level, ...)` replaces them by reading controls.json.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.ingest import probe_size, tool
from ..render.synth import SR, mix
from ..score.mapping import timeline

SVS_PAGE = "https://svs.gsfc.nasa.gov/{}"
GROUPS = {"melody", "bands", "events", "harmony", "geophony", "anthrophony", "biophony"}
CURVES = ("geophony", "anthrophony", "biophony", "brightness")


# ---------------------------------------------------------------- video io
def decode_stream(path: Path, fps: float | None, width: int, height: int | None = None, vf_extra: str = ""):
    """Yield RGB frames (H, W, 3) uint8 one at a time from ffmpeg (no full stack in memory)."""
    w0, h0 = probe_size(path)
    height = height or int(round(width * h0 / w0 / 2)) * 2
    chain = []
    if vf_extra:
        chain.append(vf_extra)
    if fps:
        chain.append(f"fps={fps}")
    chain.append(f"scale={width}:{height}:flags=area")
    cmd = [tool("ffmpeg"), "-v", "error", "-i", str(path), "-vf", ",".join(chain),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = width * height * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        p.stdout.close()
        p.wait()


def video_info(path: Path) -> dict:
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(float, s["r_frame_rate"].split("/"))
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": num / den,
            "frames": int(s.get("nb_frames") or 0), "duration": float(s.get("duration") or 0)}


# ---------------------------------------------------------------- curves
def norm01(x, lo=1, hi=99) -> np.ndarray:
    x = np.asarray(x, float)
    f = np.isfinite(x)
    if not f.any():
        return np.zeros_like(x)
    a, b = np.percentile(x[f], [lo, hi])
    return np.clip((np.nan_to_num(x, nan=a) - a) / (b - a), 0, 1) if b > a else np.zeros_like(x)


def smooth(x, win: int) -> np.ndarray:
    x = np.asarray(x, float)
    if win <= 1:
        return x
    k = np.hanning(win + 2)[1:-1]
    k /= k.sum()
    pad = win // 2
    return np.convolve(np.pad(x, pad, mode="edge"), k, mode="same")[pad:pad + len(x)]


def lon_pan(lon) -> np.ndarray | float:
    """-180 (west) .. +180 (east) -> pan -1 .. +1."""
    return np.clip(np.asarray(lon, float) / 180.0, -1, 1)


# ---------------------------------------------------------------- placeholder voices
def _env(t_ctrl, level, n, sr=SR):
    ts = np.arange(n) / sr
    return np.interp(ts, t_ctrl, np.clip(level, 0, 1)).astype(np.float32)


def _onepole(x, a):
    from scipy.signal import lfilter
    return lfilter([1 - a], [1, -a], x).astype(np.float32)


def render_curve_voice(group: str, t_ctrl, level, duration: float, seed: int = 0, sr: int = SR) -> np.ndarray:
    """Stand-in texture for a three-voice stem whose loudness follows `level` (0..1).

    geophony: low brown-noise rumble with slow swells (thunder / surf)
    anthrophony: mains hum (B1 + harmonics) plus band-limited traffic hiss
    biophony: chirps (bird / insect) whose density follows the level
    """
    rng = np.random.default_rng(seed)
    n = int((duration + 0.5) * sr)
    env = _env(t_ctrl, level, n, sr)
    ts = np.arange(n) / sr
    if group == "geophony":
        x = _onepole(rng.standard_normal(n).astype(np.float32), 0.995)
        x = _onepole(x, 0.98)
        swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * ts + rng.uniform(0, 6))
        sig = x / (np.abs(x).max() + 1e-9) * swell
    elif group == "anthrophony":
        hum = sum(a * np.sin(2 * np.pi * 61.74 * k * ts) for k, a in [(1, 1), (2, .5), (3, .3), (4, .15)])
        hiss = _onepole(rng.standard_normal(n).astype(np.float32), 0.9)
        hiss = hiss - _onepole(hiss, 0.995)  # crude band-pass (traffic wash)
        sig = 0.35 * hum / 2 + 0.65 * hiss / (np.abs(hiss).max() + 1e-9)
    elif group == "biophony":
        sig = np.zeros(n, np.float32)
        # chirp onsets: Poisson with rate proportional to the level (max ~12 / s)
        rate = 12 * env[:: sr // 100]  # per 10 ms
        hits = np.where(rng.random(len(rate)) < rate * 0.01)[0]
        for h in hits:
            s = h * (sr // 100)
            L = int(sr * rng.uniform(0.04, 0.12))
            if s + L >= n:
                continue
            tt = np.arange(L) / sr
            f0 = rng.uniform(2500, 5500)
            f = f0 * (1 + rng.choice([-1, 1]) * 0.3 * tt / tt[-1])
            ph = 2 * np.pi * np.cumsum(f) / sr
            sig[s:s + L] += (np.sin(ph) * np.hanning(L)).astype(np.float32) * rng.uniform(.4, 1)
        crick = np.sin(2 * np.pi * 4300 * ts) * (np.sin(2 * np.pi * 30 * ts) > 0.3) * 0.15
        sig = sig + crick
    else:
        raise ValueError(group)
    sig = (sig * env).astype(np.float32)
    return np.stack([sig, sig], 1) * 0.3


def render_toms(notes, duration: float, sr: int = SR, gain: float = 0.5) -> np.ndarray:
    """Pitched membrane hits (sine with pitch drop + noise click) for GM tom notes.

    note.pitch is a GM drum key; its tuning follows the key so latitude -> pitch stays audible.
    """
    out = np.zeros((int((duration + 1) * sr), 2), np.float32)
    rng = np.random.default_rng(1)
    L = int(0.35 * sr)
    tt = np.arange(L) / sr
    for n in notes:
        s = int(n.start * sr)
        if s >= len(out):
            continue
        f = 440 * 2 ** ((n.pitch - 69) / 12) * 2  # toms 41..50 -> ~175..300 Hz
        fdrop = f * (1 + 0.6 * np.exp(-tt / 0.02))
        body = np.sin(2 * np.pi * np.cumsum(fdrop) / sr) * np.exp(-tt / 0.12)
        click = rng.standard_normal(L) * np.exp(-tt / 0.004) * 0.4
        sig = (body + click) * (n.velocity / 127) * gain
        pan = float(n.pan)
        lr = np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)], np.float32)
        e = min(len(out), s + L)
        out[s:e] += (sig[: e - s, None] * lr).astype(np.float32)
    return out


# ---------------------------------------------------------------- writing the contract
@dataclass
class Stem:
    name: str
    group: str
    audio: np.ndarray
    gain: float = 1.0  # gain in mix.wav


def write_wav(path: Path, audio: np.ndarray, sr: int = SR):
    import soundfile as sf
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0.99:
        audio = audio / peak * 0.99
    sf.write(str(path), audio, sr, subtype="PCM_16")


def to_mp3(wav: Path, mp3: Path, bitrate: str = "96k"):
    mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([tool("ffmpeg"), "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame",
                    "-b:a", bitrate, str(mp3)], check=True)


def write_controls(out: Path, t, curves: dict) -> dict:
    missing = [c for c in CURVES if c not in curves]
    if missing:
        raise ValueError(f"controls missing {missing}")
    t = np.asarray(t, float)
    doc = {"t": [round(float(x), 3) for x in t],
           "curves": {k: [round(float(np.clip(v, 0, 1)), 4) for v in np.asarray(curves[k], float)] for k in CURVES}}
    for k, v in doc["curves"].items():
        if len(v) != len(t):
            raise ValueError(f"curve {k} has {len(v)} samples, t has {len(t)}")
    (out / "controls.json").write_text(json.dumps(doc))
    return doc


def write_track(out: Path, *, page_id: int, title: str, video_url: str, duration_s: float, units: str,
                description: str, legend: list[dict], stems: list[Stem], notes, t_ctrl, curves: dict,
                fidelity: dict, data_source: dict, report: dict, extra: dict | None = None) -> dict:
    """Write stems, mix, timeline, controls, report and track.json (the shared contract)."""
    out.mkdir(parents=True, exist_ok=True)
    for s in stems:
        if s.group not in GROUPS:
            raise ValueError(f"stem {s.name}: bad group {s.group}")
        write_wav(out / f"{s.name}.wav", s.audio)
    n = int(duration_s * SR) + SR
    padded = [np.pad(s.audio[:n], ((0, max(0, n - len(s.audio[:n]))), (0, 0))) for s in stems]
    write_wav(out / "mix.wav", mix(*padded, gains=[s.gain for s in stems]))
    (out / "timeline.json").write_text(_dumps(timeline(list(notes))))
    write_controls(out, t_ctrl, curves)
    (out / "report.json").write_text(_dumps(report, indent=1))
    track = {
        "id": page_id, "title": title, "svs_url": SVS_PAGE.format(page_id), "video_url": video_url,
        "duration_s": round(float(duration_s), 3), "units": units, "description": description,
        "legend": legend,
        "stems": [{"name": s.name, "file": f"{s.name}.wav", "group": s.group} for s in stems],
        "timeline": "timeline.json", "controls": "controls.json",
        "fidelity": fidelity, "data_source": data_source,
        "mix": "mix.wav", "report": "report.json", "overview": "overview.png",
    }
    if extra:
        track.update(extra)
    (out / "track.json").write_text(_dumps(track, indent=1))
    return track


def export_demo(out: Path, demo: Path, names=("mix",), bitrate="96k",
                copy=("track.json", "timeline.json", "controls.json", "report.json", "overview.png")):
    """Small MP3 copies of selected WAVs + the small contract files, for git (demo/<id>/)."""
    import shutil
    demo.mkdir(parents=True, exist_ok=True)
    for nm in names:
        to_mp3(out / f"{nm}.wav", demo / f"{nm}.mp3", bitrate)
    for f in copy:
        if (out / f).exists():
            shutil.copy2(out / f, demo / f)


def _clean(o):
    """NaN/inf -> None and numpy -> python, so the JSON is valid for browsers."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _dumps(o, **kw) -> str:
    return json.dumps(_clean(o), allow_nan=False, default=_json_default, **kw)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------- proof helpers
def note_pitches(audio: np.ndarray, notes, fmin=60.0, fmax=2000.0, win=0.25, sr: int = SR,
                 frame_length: int = 4096, hop: int = 512) -> np.ndarray:
    """Median pyin MIDI pitch near each note's onset ([start+30 ms, start+win]); NaN if unvoiced.

    Measuring at the onset (not over the whole note) avoids decayed plucks and ringing bell tails.
    """
    import librosa
    mono = audio.mean(1) if audio.ndim == 2 else audio
    f0, _, _ = librosa.pyin(mono, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop)
    ft = librosa.times_like(f0, sr=sr, hop_length=hop)
    midi = librosa.hz_to_midi(f0)
    out = np.full(len(notes), np.nan)
    for i, n in enumerate(notes):
        m = (ft >= n.start + 0.03) & (ft < min(n.end, n.start + win)) & np.isfinite(midi)
        if m.any():
            out[i] = np.median(midi[m])
    return out


def midi_to_value(midi, pm) -> np.ndarray:
    x = (np.asarray(midi, float) - pm.low) / pm.span
    if pm.polarity < 0:
        x = 1 - x
    return pm.vmin + x * (pm.vmax - pm.vmin)
