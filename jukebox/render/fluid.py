"""Render MIDI files or Note lists to stereo float32 audio (44.1 kHz) with FluidSynth.

Uses the `fluidsynth` binary from the env and a General MIDI SoundFont (FluidR3_GM, MIT;
see render/SOUNDFONT.md and render/soundfont.py). When either is missing, `render_notes`
falls back to the additive synth in `render.synth`, so callers always get audio.

    from jukebox.render.fluid import render_notes
    audio = render_notes(notes, duration, programs={"lead": 11, "harmony": 0, "bass": 32})
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .soundfont import ensure_sf2, find_sf2
from .synth import SR

# timbre for the fallback synth, by voice name prefix
FALLBACK_TIMBRE = {"lead": "pluck", "harmony": "pad", "bass": "pluck", "events": "drum"}


def fluidsynth_bin() -> str | None:
    cand = Path(sys.executable).parent / "fluidsynth"
    if cand.exists():
        return str(cand)
    return shutil.which("fluidsynth")


def available(download: bool = False) -> bool:
    return fluidsynth_bin() is not None and (ensure_sf2() if download else find_sf2()) is not None


def render_midi(midi_path, sf2=None, sr: int = SR, gain: float = 0.6, reverb: bool = True,
                chorus: bool = False, duration: float | None = None) -> np.ndarray:
    """MIDI file -> (N, 2) float32 via `fluidsynth -F` (offline, faster than real time)."""
    import soundfile as sf
    exe = fluidsynth_bin()
    sf2 = Path(sf2) if sf2 else ensure_sf2()
    if exe is None or sf2 is None:
        raise RuntimeError("fluidsynth binary or SoundFont missing")
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "out.wav"
        cmd = [exe, "-ni", "-q", "-F", str(wav), "-T", "wav", "-O", "float", "-r", str(sr), "-g", str(gain),
               "-o", "audio.driver=file", "-o", "audio.period-size=512",
               "-o", f"synth.reverb.active={int(reverb)}", "-o", f"synth.chorus.active={int(chorus)}",
               "-o", "synth.polyphony=512", str(sf2), str(midi_path)]
        env = dict(os.environ, SDL_AUDIODRIVER="dummy")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0 or not wav.exists():
            raise RuntimeError(f"fluidsynth failed ({r.returncode}): {r.stderr[-500:]}")
        audio, file_sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if file_sr != sr:
        raise RuntimeError(f"fluidsynth wrote {file_sr} Hz, expected {sr}")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if duration is not None:
        n = int(round(duration * sr))
        audio = audio[:n] if len(audio) >= n else np.pad(audio, ((0, n - len(audio)), (0, 0)))
    return np.ascontiguousarray(audio[:, :2], dtype=np.float32)


def notes_to_midi(notes, path, programs: dict[str, int], drums=frozenset(), tail: float = 2.0,
                  volumes: dict[str, int] | None = None):
    """Like score.mapping.to_midi, plus a per-voice CC7 volume and a silent marker at the
    end so FluidSynth keeps rendering the release/reverb tail."""
    import pretty_midi
    from ..score.mapping import to_midi
    pm = to_midi(list(notes), path, programs, drums=set(drums))
    end = max((n.end for n in notes), default=0.0) + tail
    for inst in pm.instruments:
        if volumes and inst.name in volumes:
            inst.control_changes.insert(0, pretty_midi.ControlChange(7, int(volumes[inst.name]), 0.0))
        inst.control_changes.append(pretty_midi.ControlChange(110, 0, end))  # undefined CC: only extends the song
    pm.write(str(path))
    return pm


def render_notes(notes, duration: float, programs: dict[str, int], *, drums=frozenset(), sf2=None,
                 sr: int = SR, gain: float = 0.6, reverb: bool = True, tail: float = 2.0,
                 volumes: dict[str, int] | None = None, fallback: bool = True) -> np.ndarray:
    """Note list -> (N, 2) float32 of length duration + tail seconds.

    FluidSynth when available, else (fallback=True) render.synth per voice.
    """
    notes = list(notes)
    total = duration + tail
    try:
        with tempfile.TemporaryDirectory() as td:
            mid = Path(td) / "score.mid"
            notes_to_midi(notes, mid, programs, drums, tail, volumes)
            return render_midi(mid, sf2, sr, gain, reverb, duration=total)
    except Exception as e:
        if not fallback:
            raise
        print(f"[fluid] {e}; using render.synth fallback")
    return render_fallback(notes, total, drums, volumes, sr)


def render_fallback(notes, total: float, drums=frozenset(), volumes=None, sr: int = SR) -> np.ndarray:
    from .synth import render_notes as synth_notes
    out = np.zeros((int(round(total * sr)), 2), np.float32)
    for voice in sorted({n.voice for n in notes}):
        timbre = "drum" if voice in drums else next(
            (t for k, t in FALLBACK_TIMBRE.items() if voice.startswith(k)), "pad")
        g = (volumes or {}).get(voice, 100) / 127
        a = synth_notes([n for n in notes if n.voice == voice], total, timbre, sr=sr, gain=.25 * g)
        n = min(len(a), len(out))
        out[:n] += a[:n]
    return out


def stems(notes, duration: float, programs: dict[str, int], **kw) -> dict[str, np.ndarray]:
    """One rendered stem per voice (same length), e.g. for the web jukebox's stem toggles."""
    notes = list(notes)
    return {v: render_notes([n for n in notes if n.voice == v], duration, programs, **kw)
            for v in sorted({n.voice for n in notes})}
