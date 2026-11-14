"""Deterministic data -> notes mapping. Every note records the value it came from.

Design rules (see docs/THREE_VOICES.md):
- value -> pitch: linear across a fixed range (default 3 octaves), snapped to a scale
- one voice per latitude band; longitude -> pan
- events -> percussion at exact times; categories -> instruments
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

SCALES = {
    "major_pent": [0, 2, 4, 7, 9],
    "minor_pent": [0, 3, 5, 7, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "chromatic": list(range(12)),
}


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int
    voice: str
    value: float          # the data value this note encodes
    pan: float = 0.0      # -1 (west) .. +1 (east)
    source: dict = field(default_factory=dict)


@dataclass
class PitchMap:
    vmin: float
    vmax: float
    low: int = 48             # C3
    span: int = 36            # 3 octaves
    root: int = 0             # pitch class of the tonic
    scale: str = "major_pent"
    polarity: int = 1         # +1: higher value -> higher pitch

    def allowed(self) -> np.ndarray:
        pcs = [(self.root + s) % 12 for s in SCALES[self.scale]]
        return np.array([p for p in range(self.low, self.low + self.span + 1) if p % 12 in pcs])

    def continuous(self, v):
        x = (np.asarray(v, float) - self.vmin) / (self.vmax - self.vmin + 1e-12)
        x = np.clip(x if self.polarity > 0 else 1 - x, 0, 1)
        return self.low + x * self.span

    def __call__(self, v) -> np.ndarray:
        cont = np.atleast_1d(self.continuous(v))
        allowed = self.allowed()
        return allowed[np.abs(cont[:, None] - allowed[None]).argmin(1)]

    def max_error(self) -> float:
        """Worst-case value error from scale snapping (data units)."""
        gaps = np.diff(self.allowed())
        return float(gaps.max() / 2 / self.span * (self.vmax - self.vmin))

    def describe(self) -> dict:
        d = asdict(self)
        d["max_snap_error"] = self.max_error()
        return d


def melody_from_curve(t, values, pm: PitchMap, voice: str, *, step: float | None = None,
                      vel_range=(50, 100), loudness=None, pan: float = 0.0, merge_repeats=True) -> list[Note]:
    """One note per sample (or per `step` seconds); repeated pitches are tied."""
    t = np.asarray(t, float)
    v = np.asarray(values, float)
    if np.isfinite(v).sum() < 2:
        return []  # no usable data in this voice (e.g. a band with no valid pixels)
    if step:
        grid = np.arange(t[0], t[-1] + 1e-9, step)
        v = np.interp(grid, t[np.isfinite(v)], v[np.isfinite(v)])
        t = grid
    ends = np.r_[t[1:], t[-1] + (t[-1] - t[-2] if len(t) > 1 else 1.0)]
    pitches = pm(v)
    loud = np.full(len(t), 0.5) if loudness is None else np.clip(np.asarray(loudness, float), 0, 1)
    vel = (vel_range[0] + loud * (vel_range[1] - vel_range[0])).astype(int)
    notes: list[Note] = []
    for i in range(len(t)):
        if not np.isfinite(v[i]):
            continue
        if merge_repeats and notes and notes[-1].pitch == pitches[i] and abs(notes[-1].end - t[i]) < 1e-6:
            notes[-1].end = float(ends[i])
            continue
        notes.append(Note(float(t[i]), float(ends[i]), int(pitches[i]), int(vel[i]), voice, float(v[i]), pan))
    return notes


def band_voices(t, zonal: np.ndarray, pm: PitchMap, names=None, step=None) -> list[Note]:
    """zonal: (T, B) latitude-band means, north to south. Each band gets its own register."""
    T, B = zonal.shape
    names = names or [f"band{b}" for b in range(B)]
    notes = []
    for b in range(B):
        # northern bands sit higher; each band keeps the same value->pitch mapping offset by register
        offset = int(round((B - 1 - b) * 12 / max(1, B - 1))) - 6
        pmb = PitchMap(pm.vmin, pm.vmax, pm.low + offset, pm.span, pm.root, pm.scale, pm.polarity)
        notes += melody_from_curve(t, zonal[:, b], pmb, names[b], step=step)
    return notes


def events_to_hits(events, *, pitch=38, voice="events", width=None) -> list[Note]:
    """events: iterable of (time, col, intensity) -> percussion notes with pan from column."""
    out = []
    for time, col, inten in events:
        pan = 0.0 if width is None else float(2 * col / max(1, width - 1) - 1)
        vel = int(np.clip(40 + inten, 40, 127))
        out.append(Note(float(time), float(time) + 0.15, pitch, vel, voice, float(inten), pan))
    return out


def to_midi(notes: list[Note], path, programs: dict[str, int], drums: set[str] = frozenset(), tempo=100.0):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for voice in sorted({n.voice for n in notes}):
        inst = pretty_midi.Instrument(program=programs.get(voice, 0), is_drum=voice in drums, name=voice)
        vn = [n for n in notes if n.voice == voice]
        pan = float(np.mean([n.pan for n in vn])) if vn else 0.0
        inst.control_changes.append(pretty_midi.ControlChange(10, int(np.clip(64 + pan * 63, 0, 127)), 0.0))
        for n in vn:
            inst.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.start, max(n.end, n.start + 0.05)))
        pm.instruments.append(inst)
    pm.write(str(path))
    return pm


def timeline(notes: list[Note]) -> list[dict]:
    """JSON-able note list (for the web jukebox's value readout and the proof step)."""
    return [asdict(n) for n in sorted(notes, key=lambda n: (n.start, n.voice))]
