"""Shared helpers for dataset builders: decoding, normalisation, stems, track contract.

Track contract (outputs/<id>/):
  track.json     {id, title, svs_url, video_url, duration_s, units, description, legend,
                  stems: [{name, file, group}], timeline, controls, fidelity, data_source}
  timeline.json  list of Note dicts (jukebox.score.mapping.timeline)
  controls.json  {"t": [...], "curves": {geophony, anthrophony, biophony, brightness}} in 0..1
  *.wav stems, report.json, overview.png
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from ..data.ingest import probe_size, tool
from ..proof.fidelity import recover_curve, score
from ..render.synth import SR, TIMBRES

# extra timbres for short, dense notes (registered at import; synth.py is unchanged)
TIMBRES.setdefault("keys", ([1, .5, .3, .15, .08, .04], .004, .06))   # short release: no smear between 0.19 s beats
TIMBRES.setdefault("softpad", ([1, .45, .2, .08], .03, .15))
from ..score.mapping import timeline

GROUPS = {"melody", "bands", "events", "harmony", "geophony", "anthrophony", "biophony"}
CURVES = ("geophony", "anthrophony", "biophony", "brightness")


def iter_frames(path: Path, width: int, fps: float | None = None, crop: tuple[int, int, int, int] | None = None):
    """Stream RGB frames (H, W, 3) uint8 from a movie without holding them all in memory.

    crop = (x, y, w, h) in *output* pixels (applied after scaling to `width`).
    """
    w0, h0 = probe_size(path)
    height = int(round(width * h0 / w0 / 2)) * 2
    vf = ([f"fps={fps}"] if fps else []) + [f"scale={width}:{height}:flags=area"]
    ow, oh = width, height
    if crop:
        x, y, cw, ch = crop
        vf.append(f"crop={cw}:{ch}:{x}:{y}")
        ow, oh = cw, ch
    cmd = [tool("ffmpeg"), "-v", "error", "-i", str(path), "-vf", ",".join(vf),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = ow * oh * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)


def grab_frame(path: Path, t: float, width: int) -> np.ndarray:
    """One RGB frame at time t (seconds)."""
    w0, h0 = probe_size(path)
    height = int(round(width * h0 / w0 / 2)) * 2
    raw = subprocess.run([tool("ffmpeg"), "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-vf", f"scale={width}:{height}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8)[: width * height * 3].reshape(height, width, 3).copy()


def movie_info(path: Path) -> dict:
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(float, s["r_frame_rate"].split("/"))
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": num / den,
            "frames": int(s.get("nb_frames") or 0), "duration": float(s.get("duration") or 0)}


def norm01(x, lo=None, hi=None) -> np.ndarray:
    scalar = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, float))
    f = np.isfinite(x)
    lo = np.nanmin(x) if lo is None else lo
    hi = np.nanmax(x) if hi is None else hi
    y = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
    y[~f] = 0.0
    return float(y[0]) if scalar else y


def write_wav(path: Path, audio: np.ndarray, sr: int = SR):
    import soundfile as sf
    sf.write(str(path), np.clip(audio, -1, 1), sr, subtype="PCM_16")


def to_mp3(wav: Path, mp3: Path, bitrate: str = "96k"):
    mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([tool("ffmpeg"), "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame",
                    "-b:a", bitrate, str(mp3)], check=True)


def normalize_stems(stems: dict[str, np.ndarray], peak: float = 0.9) -> dict[str, np.ndarray]:
    """Scale all stems by one common factor so their sum peaks at `peak` (relative levels kept,
    so the web player can sum stems at unity gain and get the mix)."""
    n = max(len(s) for s in stems.values())
    tot = np.zeros((n, 2), np.float32)
    for s in stems.values():
        tot[: len(s)] += s
    k = peak / max(1e-9, float(np.abs(tot).max()))
    out = {}
    for name, s in stems.items():
        pad = np.zeros((n, 2), np.float32)
        pad[: len(s)] = s * k
        out[name] = pad
    return out


def roundtrip(audio: np.ndarray, pm, grid: np.ndarray, truth: np.ndarray, **kw) -> tuple[dict, np.ndarray]:
    """pyin the stem back to data units at `grid` and score against `truth`."""
    if "fmin" not in kw:   # bound pyin to the PitchMap's register (+-2 semitones): avoids octave errors
        from ..render.synth import midi_hz
        kw["fmin"] = float(midi_hz(pm.low - 2))
        kw["fmax"] = float(midi_hz(pm.low + pm.span + 2))
    est = recover_curve(audio, SR, pm, grid, **kw)
    return score(truth, est), est


def write_track(out: Path, *, page_id: int, title: str, video_url: str, duration_s: float, units: str,
                description: str, legend: list[dict], stems: dict[str, np.ndarray],
                stem_groups: dict[str, str], notes, controls_t, curves: dict, fidelity: dict,
                data_source: str | dict, mix_gains: dict[str, float] | None = None, extra: dict | None = None):
    """Write stems (WAV), mix.wav, timeline.json, controls.json and track.json."""
    out.mkdir(parents=True, exist_ok=True)
    stems = normalize_stems(stems)
    stem_list = []
    for name, audio in stems.items():
        grp = stem_groups[name]
        assert grp in GROUPS, grp
        f = f"{name}.wav"
        write_wav(out / f, audio)
        stem_list.append({"name": name, "file": f, "group": grp})
    gains = mix_gains or {}
    m = sum(stems[n] * gains.get(n, 1.0) for n in stems)
    peak = float(np.abs(m).max())
    write_wav(out / "mix.wav", m / peak * 0.9 if peak > 0 else m)

    (out / "timeline.json").write_text(json.dumps(timeline(notes)))
    for k in CURVES:
        assert k in curves, k
    ctrl = {"t": [round(float(x), 4) for x in controls_t],
            "curves": {k: [round(float(np.clip(v, 0, 1)), 4) for v in curves[k]] for k in CURVES}}
    (out / "controls.json").write_text(json.dumps(ctrl))
    track = {
        "id": page_id, "title": title, "svs_url": f"https://svs.gsfc.nasa.gov/{page_id}/",
        "video_url": video_url, "duration_s": round(float(duration_s), 3), "units": units,
        "description": description, "legend": legend, "stems": stem_list,
        "timeline": "timeline.json", "controls": "controls.json",
        "fidelity": {k: (None if fidelity.get(k) is None or not np.isfinite(fidelity[k]) else round(float(fidelity[k]), 4))
                     for k in ("r2", "pearson_r", "mae", "control_r2")},
        "data_source": data_source,
        "mix": "mix.wav",
    }
    if extra:
        track.update(extra)
    (out / "track.json").write_text(json.dumps(track, indent=1, ensure_ascii=False))
    return track


def export_demo(out: Path, demo: Path, names=("mix",), extra_files=("track.json", "report.json", "overview.png", "timeline.json", "controls.json")):
    """Small MP3 copies + the lightweight JSON/PNG for committing under demo/<id>/."""
    import shutil
    demo.mkdir(parents=True, exist_ok=True)
    for n in names:
        if (out / f"{n}.wav").exists():
            to_mp3(out / f"{n}.wav", demo / f"{n}.mp3")
    for f in extra_files:
        if (out / f).exists():
            shutil.copy(out / f, demo / f)
