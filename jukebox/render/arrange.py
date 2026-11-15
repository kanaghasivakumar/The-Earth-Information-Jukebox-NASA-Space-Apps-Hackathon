"""One arranger for every track: data notes + model harmony + the three voices -> final stems and mix.

    python -m jukebox.render.arrange outputs/3912            # config from jukebox/render/arrangements.py
    python -m jukebox.render.arrange outputs/5011 --no-harmony

Input  outputs/<id>/{timeline.json, controls.json, track.json, report.json}
Output outputs/<id>/
  melody.wav bands*.wav events.wav harmony.wav geophony.wav anthrophony.wav biophony.wav
                       stems at mix gain (they sum to the pre-limiter mix; silent groups are skipped)
  mix.wav              all stems, loudness-normalised (about -16 LUFS) through a gentle peak limiter
  control_shuffled.wav the melody group re-rendered with the time-shuffled control lead (same gain)
  mix_control.wav      the same arrangement with the control lead instead of the data lead
  timeline.json        data notes (unchanged, as read) + harmony/bass/form notes (value null)
  track.json           stems with groups, fidelity of the final audio, arrangement summary
  arrange_report.json  per-group RMS, loudness, limiter, timings, fidelity before/after
  arrange.png          per-group level over time, mix spectrogram, lead round trip

Rules
- Data notes (melody, bands, events voices) are never edited. `arrange` asserts this after the
  harmony and form steps and again on the timeline it writes.
- Harmony comes from `score.infill.infill` (Anticipatory Music Transformer) around the lead notes.
- Voices come from `voices.render_voice` driven by the controls.json curves (all-zero curves are
  skipped), tuned with `tune_to=` to the harmony (or band) chord timeline.
- Per-group gains are fixed relative to the melody's active RMS, and one sidechain gain keeps the
  accompaniment bus (bands, harmony, voices; never the data event hits) 8 dB under the sounding lead,
  so the lead stays on top and stays recoverable by pyin in the full mix.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from ..score.mapping import Note, PitchMap
from . import fluid
from .synth import SR, midi_hz

GROUPS = ("melody", "bands", "events", "harmony", "geophony", "anthrophony", "biophony")
VOICE_GROUPS = ("geophony", "anthrophony", "biophony")
MODEL_VOICES = ("harmony", "bass")
FORM_PREFIX = "form "
DEFAULT_PROGRAMS = {"melody": 73, "bands": 89, "events": 115, "harmony": 48}  # flute lead (survives pyin)
# dB below the melody's active RMS. Bands/harmony: their active RMS. Voices: their active RMS scaled
# by the curve's peak level (a curve that never exceeds 0.12 stays ~18 dB lower than one reaching 1).
DEFAULT_GAINS_DB = {"melody": 0.0, "bands": -15.0, "events": -6.0, "harmony": -13.0,
                    "geophony": -10.0, "anthrophony": -10.0, "biophony": -11.0}
# model harmony: sustained strings, then a bass line (piano attacks were masking quiet lead notes)
DEFAULT_HARMONY_PASSES = [{48: "harmony"}, {32: "bass", 33: "bass"}]
DEFAULT_CC7 = {"harmony": 72, "bass": 100, "form cymbal": 55, "form motif": 70}


# ---------------------------------------------------------------- configuration
@dataclass
class Bed:
    """One recorded-voice layer: render_voice (curve-driven) or render_events (hits)."""
    group: str
    curve: str | None = None            # controls.json curve (default: the group's own)
    sources: tuple | None = None        # corpus tags
    tune_mix: float = 0.4
    width: float = 0.5
    density: float = 1.0
    gain_db: float = 0.0                # on top of the group's mix gain
    events_voice: str | None = None     # render_events at this timeline voice's note onsets
    length: float = 2.5                 # render_events tail
    brightness: bool = True             # use the brightness curve


@dataclass
class FidSpec:
    """Which notes to pyin back, and the PitchMap that turns pitch back into data units."""
    voices: tuple
    pm: dict
    label: str = "lead"
    stem: str = "melody"                # stem the lead check reads (a group name or a split stem name)
    orig_stem: str | None = None        # builder's original stem in outputs/<id>/ (the "before" number)
    solo: bool = False                  # only notes that do not overlap another note of the same voice
    win: float = 0.30
    offset: float = 0.03                # skip this much of each note's attack (a tick or drum on the onset)
    truth_key: str | None = None        # score against note.source[truth_key] instead of note.value


@dataclass
class Arrangement:
    id: str
    lead_voices: tuple = ("lead",)
    event_voices: tuple = ()
    programs: dict = field(default_factory=dict)      # voice -> GM program (overrides group default)
    drums: tuple = ()                                  # voices on the GM drum channel
    cc7: dict = field(default_factory=dict)            # voice -> CC7 volume
    harmony: dict | None = field(default_factory=dict)  # infill kwargs; None = no harmony
    harmony_max_per_s: float | None = None             # thin model notes to at most this many onsets/s/voice
    form: str | None = None                            # form.json path (relative to the repo root)
    form_kwargs: dict = field(default_factory=dict)
    beds: list = field(default_factory=list)           # [Bed]; default: one per non-zero curve
    gains_db: dict = field(default_factory=dict)
    split: tuple = ()                                  # groups rendered as one stem per voice
    fidelity: list = field(default_factory=list)       # [FidSpec]
    control: bool = True                               # shuffled-control A/B for the first FidSpec
    tune: str = "auto"                                 # chord source: auto | harmony | bands | lead
    seed: int = 0
    lufs: float = -16.0
    reverb: bool = True
    tail: float = 1.5
    pan_buckets: float = 0.25                          # pan resolution for voices whose pan varies
    mix_fidelity: bool = True
    duck_db: float | None = 8.0                        # keep the accompaniment this far under the sounding lead
    duck_max_db: float = 9.0                           # ... ducking it by at most this much
    duck_groups: tuple = ("bands", "harmony", "geophony", "anthrophony", "biophony")
    notes: str = ""
    text_subs: list = field(default_factory=list)      # [(old, new)] for description/legend (new timbres)


# ---------------------------------------------------------------- I/O helpers
def load_notes(path: Path) -> list[Note]:
    out = []
    for d in json.loads(Path(path).read_text()):
        d = dict(d)
        if d.get("value") is None:
            d["value"] = float("nan")
        d.setdefault("pan", 0.0)
        d.setdefault("source", {})
        out.append(Note(**d))
    return out


def json_safe(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, np.generic):
        return json_safe(o.item())
    return o


def write_wav(path: Path, audio: np.ndarray, sr: int = SR):
    import soundfile as sf
    sf.write(str(path), np.clip(audio, -1, 1).astype(np.float32), sr, subtype="PCM_16")


def fingerprint(notes) -> list[dict]:
    """Exact, order-independent identity of a note list (NaN-safe)."""
    rows = [json_safe(asdict(n)) for n in notes]
    return sorted(rows, key=lambda d: json.dumps(d, sort_keys=True))


# ---------------------------------------------------------------- grouping
def group_of(voice: str, cfg: Arrangement) -> str:
    if voice in cfg.lead_voices:
        return "melody"
    if voice in cfg.event_voices:
        return "events"
    if voice in MODEL_VOICES or voice.startswith(FORM_PREFIX):
        return "harmony"
    return "bands"


def is_data(n: Note, cfg: Arrangement) -> bool:
    return group_of(n.voice, cfg) != "harmony"


# ---------------------------------------------------------------- FluidSynth rendering
@dataclass
class Part:
    notes: list
    program: int
    drum: bool
    pan: float
    cc7: int
    name: str


def _parts(notes: list[Note], cfg: Arrangement, group: str) -> list[Part]:
    parts = []
    by_key: dict = {}
    for n in notes:
        prog = n.source.get("program") if n.voice in MODEL_VOICES else None
        key = (n.voice, prog)
        by_key.setdefault(key, []).append(n)
    for (voice, prog), vn in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
        if prog is None:
            prog = cfg.programs.get(voice, 0 if voice.startswith(FORM_PREFIX) else DEFAULT_PROGRAMS.get(group, 0))
        drum = voice in cfg.drums or voice == "form cymbal"
        cc7 = int(cfg.cc7.get(voice, DEFAULT_CC7.get(voice, 100)))
        pans = np.array([n.pan for n in vn], float)
        if np.ptp(pans) <= 0.05:
            parts.append(Part(vn, int(prog), drum, float(pans.mean()), cc7, voice))
            continue
        q = cfg.pan_buckets
        buckets: dict = {}
        for n in vn:
            buckets.setdefault(float(np.clip(round(n.pan / q) * q, -1, 1)), []).append(n)
        for p, bn in sorted(buckets.items()):
            parts.append(Part(bn, int(prog), drum, p, cc7, f"{voice}@{p:+.2f}"))
    return parts


def _midi(parts: list[Part], path: Path, end: float):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=100.0)
    for p in parts:
        inst = pretty_midi.Instrument(program=p.program, is_drum=p.drum, name=p.name)
        inst.control_changes.append(pretty_midi.ControlChange(7, p.cc7, 0.0))
        inst.control_changes.append(pretty_midi.ControlChange(10, int(np.clip(64 + p.pan * 63, 0, 127)), 0.0))
        for n in p.notes:
            inst.notes.append(pretty_midi.Note(int(np.clip(n.velocity, 1, 127)), int(n.pitch), float(n.start),
                                               float(max(n.end, n.start + 0.05))))
        inst.control_changes.append(pretty_midi.ControlChange(110, 0, end))
        pm.instruments.append(inst)
    pm.write(str(path))


def render_parts(parts: list[Part], duration: float, *, reverb: bool = True, tail: float = 1.5,
                 sf2=None) -> tuple[np.ndarray, str]:
    """Render parts to one (N, 2) buffer of length duration + tail. Non-drum parts share MIDI files
    (<= 15 channels each); each drum part gets its own file so its pan is its own."""
    import tempfile
    n_out = int(round((duration + tail) * SR))
    out = np.zeros((n_out, 2), np.float32)
    if not parts:
        return out, "none"
    if fluid.available():
        files: list[list[Part]] = []
        melodic = [p for p in parts if not p.drum]
        for i in range(0, len(melodic), 15):
            files.append(melodic[i:i + 15])
        files += [[p] for p in parts if p.drum]
        try:
            with tempfile.TemporaryDirectory() as td:
                for k, fp in enumerate(files):
                    mid = Path(td) / f"p{k}.mid"
                    _midi(fp, mid, duration + tail)
                    a = fluid.render_midi(mid, sf2, SR, gain=0.6, reverb=reverb, duration=duration + tail)
                    out += a[:n_out]
            return out, "fluidsynth"
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[arrange] fluidsynth failed ({e}); synth fallback")
            out[:] = 0
    for p in parts:
        notes = [replace(n, pan=p.pan, voice=p.name) for n in p.notes]
        a = fluid.render_fallback(notes, duration + tail, drums={p.name} if p.drum else frozenset(),
                                  volumes={p.name: p.cc7})
        m = min(len(a), n_out)
        out[:m] += a[:m]
    return out, "synth-fallback"


# ---------------------------------------------------------------- harmony + form
def add_harmony(notes: list[Note], duration: float, cfg: Arrangement, cache: Path | None = None,
                model=None, verbose=True) -> tuple[list[Note], dict]:
    """AMT infill around the lead voices. Cached in `cache` (keyed by lead notes + settings)."""
    from ..score import infill as I
    kw = dict(cfg.harmony or {})
    model_name = kw.pop("model", I.DEFAULT_MODEL)
    seed = int(kw.pop("seed", cfg.seed))
    lead = [n for n in notes if n.voice in cfg.lead_voices]
    kw.setdefault("instruments", DEFAULT_HARMONY_PASSES)
    if "key" not in kw and cfg.fidelity:   # the lead's own scale, not infer_key's guess: a D major
        pm = cfg.fidelity[0].pm             # pentatonic lead also fits A and G major
        root = int(pm.get("root", 0))
        kw["key"] = (root, "major") if "major" in pm.get("scale", "major") else ((root + 3) % 12, "major")
    if "ranges" not in kw and lead:   # keep the model's harmony under the whole lead register (clear lead, clean pyin)
        kw["ranges"] = {"harmony": (43, max(55, min(n.pitch for n in lead) - 1))}
    key = json.dumps({"lead": fingerprint(lead), "kw": json_safe(kw), "model": model_name if model is None else
                      getattr(model, "name", "custom"), "seed": seed, "duration": duration}, sort_keys=True)
    import hashlib
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    if cache is not None and cache.exists():
        c = json.loads(cache.read_text())
        if c.get("digest") == digest:
            new = [Note(**{**d, "value": float("nan")}) for d in c["notes"]]
            return new, {**c["info"], "cached": True}
    t0 = time.time()
    new = I.infill(notes, duration, model=model if model is not None else model_name, seed=seed,
                   lead_voices=tuple(cfg.lead_voices), verbose=verbose, **kw)
    info = {"model": model_name if model is None else getattr(model, "name", "custom"), "seed": seed,
            "seconds": round(time.time() - t0, 1), "kwargs": json_safe(kw),
            "notes": {v: sum(n.voice == v for n in new) for v in MODEL_VOICES},
            "clash_report": I.clash_report(lead, new), "cached": False}
    if cache is not None:
        cache.write_text(json.dumps(json_safe({"digest": digest, "info": info, "notes": [asdict(n) for n in new]})))
    return new, info


def thin_notes(new: list[Note], max_per_s: float, window: float = 1.0) -> list[Note]:
    """Keep the longest/loudest model notes so each voice has at most max_per_s onsets per second
    (only model notes are ever passed in; data notes are untouched)."""
    keep: list[Note] = []
    for v in sorted({n.voice for n in new}):
        cand = sorted([n for n in new if n.voice == v], key=lambda n: (-(n.end - n.start) * n.velocity, n.start, n.pitch))
        onsets: list[float] = []
        for n in cand:
            if sum(abs(o - n.start) < window / 2 for o in onsets) < max_per_s * window:
                keep.append(n)
                onsets.append(n.start)
    return sorted(keep, key=lambda n: (n.start, n.voice, n.pitch))


def add_form(notes: list[Note], form_path: Path, cfg: Arrangement) -> tuple[list[Note], dict]:
    from ..form.apply import apply_form
    form = json.loads(Path(form_path).read_text())
    kw = {"transpose": False, "motifs": False, "cymbal": True, "phrase_break": True, **cfg.form_kwargs}
    protect = tuple(sorted({n.voice for n in notes if is_data(n, cfg)}))
    out, fj = apply_form(notes, form, protect=protect, **kw)
    return out, {"sections": [(s["start"], s["end"], s["label"]) for s in fj["sections"]],
                 "boundaries": fj["cues"]["boundaries"], "non_data_changes": fj["non_data_changes"],
                 "kwargs": kw, "source": str(form_path)}


# ---------------------------------------------------------------- chords for the voices
def chord_timeline(notes: list[Note], duration: float, seg: float = 2.0, lo: int = 45, hi: int = 69,
                   max_pcs: int = 4) -> list:
    """[(start, end, [midi pitches])]: the most-sounded pitch classes in each window, voiced in lo..hi."""
    out, prev = [], None
    for a in np.arange(0.0, duration, seg):
        b = min(duration, a + seg)
        w = np.zeros(12)
        for n in notes:
            ov = min(n.end, b) - max(n.start, a)
            if ov > 0:
                w[n.pitch % 12] += ov * (n.velocity / 127)
        if w.sum() <= 0:
            if prev:
                out.append((float(a), float(b), prev))
            continue
        pcs = [int(p) for p in np.argsort(-w)[:max_pcs] if w[p] > 0.08 * w.max()]
        pitches = sorted({lo + ((pc - lo) % 12) for pc in pcs})
        pitches = [p if p <= hi else p - 12 for p in pitches]
        prev = pitches
        out.append((float(a), float(b), pitches))
    return out


# ---------------------------------------------------------------- loudness / limiter / levels
def _biquad(b, a, x):
    from scipy.signal import lfilter
    return lfilter(b, a, x, axis=0)


def k_weight(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """ITU-R BS.1770 K-weighting (high shelf + RLB high-pass), designed for any sample rate."""
    # stage 1: high shelf (+4 dB above ~1.7 kHz)
    G, Q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    A = 10 ** (G / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * Q)
    b = [A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * np.cos(w0)),
         A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)]
    a = [(A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * np.cos(w0)),
         (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha]
    y = _biquad(np.array(b) / a[0], np.array(a) / a[0], x)
    # stage 2: high pass ~38 Hz
    Q, fc = 0.5003270373238773, 38.13547087602444
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * Q)
    b = [(1 + np.cos(w0)) / 2, -(1 + np.cos(w0)), (1 + np.cos(w0)) / 2]
    a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
    return _biquad(np.array(b) / a[0], np.array(a) / a[0], y)


def lufs(x: np.ndarray, sr: int = SR) -> float:
    """Integrated loudness (BS.1770-4: 400 ms blocks, 75 % overlap, -70 LUFS and -10 LU gates)."""
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    y = k_weight(x.astype(np.float64), sr)
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    if len(y) < blk:
        return float("-inf")
    p2 = np.cumsum(np.r_[np.zeros((1, y.shape[1])), y ** 2], axis=0)
    starts = np.arange(0, len(y) - blk + 1, hop)
    z = (p2[starts + blk] - p2[starts]) / blk            # (n_blocks, ch) mean square
    L = -0.691 + 10 * np.log10(z.sum(1) + 1e-20)
    keep = L > -70
    if not keep.any():
        return float("-inf")
    rel = -0.691 + 10 * np.log10(z[keep].sum(1).mean() + 1e-20) - 10
    keep &= L > rel
    return float(-0.691 + 10 * np.log10(z[keep].sum(1).mean() + 1e-20))


def limiter(x: np.ndarray, ceiling_db: float = -1.0, lookahead: float = 0.005, release: float = 0.08,
            sr: int = SR) -> tuple[np.ndarray, float]:
    """Gentle look-ahead peak limiter on 1 ms blocks. Returns (audio, max gain reduction in dB)."""
    from scipy.ndimage import maximum_filter1d
    c = 10 ** (ceiling_db / 20)
    B = max(1, int(sr / 1000))
    nb = int(np.ceil(len(x) / B))
    pk = np.abs(np.pad(x, ((0, nb * B - len(x)), (0, 0)))).max(1).reshape(nb, B).max(1)
    la = max(1, int(round(lookahead * 1000)))
    env = maximum_filter1d(pk, size=2 * la + 1, mode="nearest")   # sees `la` ms ahead and behind
    target = np.minimum(1.0, c / np.maximum(env, 1e-9))
    g = np.empty(nb)
    r = np.exp(-1.0 / (release * 1000))
    cur = 1.0
    for i in range(nb):                                            # instant attack, smooth release
        cur = target[i] if target[i] < cur else target[i] + (cur - target[i]) * r
        g[i] = cur
    gs = np.interp(np.arange(nb * B) / B, np.arange(nb), g)[: len(x)]
    y = (x * gs[:, None]).astype(np.float32)
    return np.clip(y, -c, c), float(-20 * np.log10(g.min()))


def active_rms(x: np.ndarray, sr: int = SR, frame: float = 0.1, rel_db: float = -20.0) -> float:
    """RMS over the frames that are within rel_db of the loudest frame (ignores silence/rests)."""
    if x is None or not np.any(x):
        return 0.0
    m = x.mean(1) if x.ndim == 2 else x
    n = int(frame * sr)
    k = len(m) // n
    if k == 0:
        return float(np.sqrt(np.mean(m ** 2)))
    e = (m[: k * n].reshape(k, n) ** 2).mean(1)
    keep = e >= e.max() * 10 ** (rel_db / 10)
    return float(np.sqrt(e[keep].mean()))


def duck_gain(lead: np.ndarray, acc: np.ndarray, margin_db: float, max_db: float, sr: int = SR,
              frame: float = 0.01, attack: float = 0.03, release: float = 0.25) -> np.ndarray:
    """Sidechain gain (per sample) for the accompaniment bus: wherever the lead is sounding and the
    accompaniment's short-term RMS comes within `margin_db` of the lead's, turn the accompaniment down
    (at most `max_db`). One curve for the whole bus, so the voices keep their balance with each other."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    n = int(frame * sr)
    k = len(lead) // n + 1
    def env(x):
        m = np.pad(x.mean(1) ** 2, (0, k * n - len(x)))
        return np.sqrt(uniform_filter1d(m.reshape(k, n).mean(1), 8, mode="nearest"))   # ~80 ms RMS
    el, ea = env(lead), env(acc)
    el = maximum_filter1d(el, 15, mode="nearest")          # hold over ~150 ms so notes duck as a whole
    active = el > el.max() * 10 ** (-40 / 20)
    target = np.ones(k)
    want = el * 10 ** (-margin_db / 20) / np.maximum(ea, 1e-9)
    target[active] = np.clip(want[active], 10 ** (-max_db / 20), 1.0)
    g = np.empty(k)
    a_c, r_c = np.exp(-frame / attack), np.exp(-frame / release)
    cur = 1.0
    for i in range(k):
        c = a_c if target[i] < cur else r_c
        cur = target[i] + (cur - target[i]) * c
        g[i] = cur
    return np.interp(np.arange(len(lead)) / n, np.arange(k), g).astype(np.float32)


def db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-12)))


# ---------------------------------------------------------------- fidelity
_PYIN_SR = 22050


def pyin_track(audio: np.ndarray, fmin: float, fmax: float, sr: int = SR):
    """Register-bounded pyin on a 22.05 kHz mono copy. Returns (times, midi)."""
    import librosa
    from scipy.signal import resample_poly
    mono = audio.mean(1) if audio.ndim == 2 else audio
    if sr != _PYIN_SR:
        from math import gcd
        g = gcd(sr, _PYIN_SR)
        mono = resample_poly(mono, _PYIN_SR // g, sr // g)
    f0, _, _ = librosa.pyin(mono.astype(np.float32), fmin=fmin, fmax=fmax, sr=_PYIN_SR, frame_length=2048,
                            hop_length=256)
    return librosa.times_like(f0, sr=_PYIN_SR, hop_length=256), librosa.hz_to_midi(f0)


def note_estimates(track, notes: list[Note], win: float = 0.30, offset: float = 0.03) -> np.ndarray:
    ft, midi = track
    out = np.full(len(notes), np.nan)
    for i, n in enumerate(notes):
        hi = min(n.end, n.start + win)
        lo = n.start + min(offset, 0.3 * (hi - n.start))
        m = (ft >= lo) & (ft < hi) & np.isfinite(midi)
        if m.any():
            out[i] = np.median(midi[m])
    return out


def midi_to_value(midi, pm: PitchMap) -> np.ndarray:
    x = (np.asarray(midi, float) - pm.low) / pm.span
    if pm.polarity < 0:
        x = 1 - x
    return pm.vmin + x * (pm.vmax - pm.vmin)


def score_values(truth, est) -> dict:
    truth, est = np.asarray(truth, float), np.asarray(est, float)
    m = np.isfinite(truth) & np.isfinite(est)
    if m.sum() < 3:
        return {"n": int(m.sum()), "r2": None, "pearson_r": None, "mae": None}
    t, e = truth[m], est[m]
    ss_tot = ((t - t.mean()) ** 2).sum()
    return {"n": int(m.sum()), "r2": float(1 - ((t - e) ** 2).sum() / ss_tot) if ss_tot > 0 else None,
            "pearson_r": float(np.corrcoef(t, e)[0, 1]) if np.std(e) > 0 else None,
            "mae": float(np.abs(t - e).mean())}


def truth_value(n: Note, spec: FidSpec) -> float:
    v = n.source.get(spec.truth_key) if spec.truth_key else n.value
    return float("nan") if v is None else float(v)


def fid_notes(notes: list[Note], spec: FidSpec) -> list[Note]:
    ns = sorted([n for n in notes if n.voice in spec.voices and np.isfinite(truth_value(n, spec))],
                key=lambda n: n.start)
    if spec.solo:
        ns = [n for n in ns if not any(o is not n and o.voice == n.voice and o.start < n.end and o.end > n.start
                                       for o in ns)]
    return ns


def fidelity(audio: np.ndarray, notes: list[Note], spec: FidSpec, truth_notes: list[Note] | None = None,
             track=None) -> dict:
    """Note-level round trip: pyin each note's onset window, map back through the PitchMap, score
    against the data value (R²) and against the written pitch (share within half a semitone)."""
    pm = PitchMap(**{k: spec.pm[k] for k in ("vmin", "vmax", "low", "span", "root", "scale", "polarity") if k in spec.pm})
    ns = fid_notes(notes, spec)
    truth = truth_notes if truth_notes is not None else ns
    if not ns:
        return {"n": 0}
    if track is None:
        track = pyin_track(audio, *pyin_range(notes, spec))
    est = note_estimates(track, ns, spec.win, spec.offset)
    out = score_values([truth_value(n, spec) for n in truth], midi_to_value(est, pm))
    ok = np.isfinite(est)
    err = np.abs(est[ok] - np.array([n.pitch for n in ns])[ok])
    out["pitch_within_half_semitone"] = float((err < 0.5).mean()) if ok.any() else None
    out["voiced"] = float(ok.mean())
    return out


def pyin_range(notes: list[Note], spec: FidSpec) -> tuple[float, float]:
    """pyin search band: the checked notes' own sounding range +-1 semitone (register-bounded, like the
    builders' +-2 around the PitchMap, but it also keeps a drone or pad just under the lead out)."""
    ps = [n.pitch for n in fid_notes(notes, spec)]
    lo = min(ps) - 1
    return float(midi_hz(lo)), float(midi_hz(max(max(ps) + 1, lo + 12)))   # pyin needs about an octave


def shuffled_control(notes: list[Note], voices, seed: int = 0) -> list[Note]:
    """Same note slots, pitches/velocities/values permuted in time (like the builders' controls)."""
    lead = sorted([n for n in notes if n.voice in voices], key=lambda n: n.start)
    perm = np.random.default_rng(seed).permutation(len(lead))
    return [replace(n, pitch=lead[j].pitch, velocity=lead[j].velocity, value=lead[j].value)
            for n, j in zip(lead, perm)]


# ---------------------------------------------------------------- main
def _controls(out: Path, duration: float):
    p = out / "controls.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    t = np.asarray(c["t"], float)
    curves = {k: np.nan_to_num(np.asarray(v, float)) for k, v in c["curves"].items()}
    return t, curves, c


def arrange(out: Path, cfg: Arrangement, *, harmony: bool = True, model=None, root: Path | None = None,
            plot: bool = True, verbose: bool = True, mix_fidelity: bool | None = None) -> dict:
    out = Path(out)
    root = root or Path(__file__).resolve().parents[2]
    t_all = time.time()
    if not (out / "track_data.json").exists():   # the builder's track.json, kept as the source of truth
        (out / "track_data.json").write_text((out / "track.json").read_text())
    track = json.loads((out / "track_data.json").read_text())
    report_in = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    src_path = out / "timeline_data.json"
    if not src_path.exists():   # keep the builder's data timeline once; later runs re-arrange from it
        src_path.write_text((out / "timeline.json").read_text())
    data_notes = [n for n in load_notes(src_path) if group_of(n.voice, cfg) != "harmony"]
    data_fp = fingerprint(data_notes)
    duration = float(track.get("duration_s") or report_in.get("duration_s") or max(n.end for n in data_notes))
    timings = {}
    info: dict = {"id": cfg.id, "duration_s": duration, "seed": cfg.seed}

    # 1. harmony around the fixed lead
    notes = list(copy.deepcopy(data_notes))
    if harmony and cfg.harmony is not None and any(n.voice in cfg.lead_voices for n in notes):
        t0 = time.time()
        new, info["harmony"] = add_harmony(notes, duration, cfg, out / "harmony_cache.json", model=model,
                                           verbose=verbose)
        timings["harmony"] = time.time() - t0
        if cfg.harmony_max_per_s:
            kept = thin_notes(new, cfg.harmony_max_per_s)
            info["harmony"]["thinned"] = {"before": len(new), "after": len(kept), "max_per_s": cfg.harmony_max_per_s}
            new = kept
        notes += new
    assert fingerprint([n for n in notes if is_data(n, cfg)]) == data_fp, "harmony step changed data notes"

    # 2. form cues (never touch data notes)
    if cfg.form:
        notes, info["form"] = add_form(notes, root / cfg.form, cfg)
        assert fingerprint([n for n in notes if is_data(n, cfg)]) == data_fp, "form step changed data notes"

    # 3. pitched groups through FluidSynth
    by_group = {g: [n for n in notes if group_of(n.voice, cfg) == g] for g in ("melody", "bands", "events", "harmony")}
    stems: dict[str, tuple[str, np.ndarray]] = {}     # stem name -> (group, audio)
    engines = set()
    t0 = time.time()
    for g, gn in by_group.items():
        if not gn:
            continue
        if g in cfg.split:
            for v in sorted({n.voice for n in gn}):
                a, eng = render_parts(_parts([n for n in gn if n.voice == v], cfg, g), duration,
                                      reverb=cfg.reverb, tail=cfg.tail)
                stems[f"{g}_{_slug(v)}"] = (g, a)
                engines.add(eng)
        else:
            a, eng = render_parts(_parts(gn, cfg, g), duration, reverb=cfg.reverb, tail=cfg.tail)
            stems[g] = (g, a)
            engines.add(eng)
    ctrl_audio = None
    spec0 = cfg.fidelity[0] if cfg.fidelity else None
    if cfg.control and spec0 is not None:
        ctrl_lead = shuffled_control(notes, spec0.voices, cfg.seed)
        others = [n for n in by_group["melody"] if n.voice not in spec0.voices]
        ctrl_audio, _ = render_parts(_parts(ctrl_lead + others, cfg, "melody"), duration, reverb=cfg.reverb,
                                     tail=cfg.tail)
    timings["fluidsynth"] = time.time() - t0

    # 4. the three voices from controls.json
    t0 = time.time()
    n_total = int(round((duration + cfg.tail) * SR))
    ctl = _controls(out, duration)
    tune_src = cfg.tune
    if tune_src == "auto":
        tune_src = "harmony" if by_group["harmony"] else ("bands" if by_group["bands"] else "lead")
    tune_notes = {"harmony": [n for n in by_group["harmony"] if n.voice in MODEL_VOICES] or by_group["harmony"],
                  "bands": by_group["bands"], "lead": by_group["melody"]}[tune_src]
    chords = chord_timeline(tune_notes, duration) if tune_notes else None
    info["tune_to"] = {"source": tune_src, "segments": len(chords or [])}
    beds = cfg.beds or [Bed(g) for g in VOICE_GROUPS]
    bed_audio: dict[str, np.ndarray] = {}
    bed_peak: dict[str, float] = {}
    bed_info = []
    if ctl is not None:
        from ..voices import render_events, render_voice
        tc, curves, _ = ctl
        for k, b in enumerate(beds):
            seed = cfg.seed * 1000 + k
            if b.events_voice:
                ev = [n for n in data_notes if n.voice == b.events_voice]
                if not ev:
                    continue
                vel = np.array([n.velocity for n in ev], float)
                inten = 0.35 + 0.65 * (vel - vel.min()) / (np.ptp(vel) + 1e-9)
                a = render_events([n.start for n in ev], inten, duration + cfg.tail, pan=[n.pan for n in ev],
                                  group=b.group, sources=b.sources or ("thunder",), seed=seed, length=b.length)
                bed_info.append({"group": b.group, "kind": "events", "voice": b.events_voice, "hits": len(ev)})
            else:
                lvl = curves.get(b.curve or b.group)
                if lvl is None or not np.any(lvl > 1e-4):
                    bed_info.append({"group": b.group, "curve": b.curve or b.group, "skipped": "all zeros"})
                    continue
                br = curves.get("brightness") if b.brightness else None
                a = render_voice(b.group, tc, np.clip(lvl, 0, 1), duration + cfg.tail, brightness=br, seed=seed,
                                 sources=list(b.sources) if b.sources else None, tune_to=chords,
                                 tune_mix=b.tune_mix if chords else 0.0, width=b.width, density=b.density)
                bed_info.append({"group": b.group, "curve": b.curve or b.group, "sources": b.sources,
                                 "tune_mix": b.tune_mix, "mean_level": float(lvl.mean())})
            a = a[:n_total] * np.float32(10 ** (b.gain_db / 20))
            prev = bed_audio.get(b.group)
            bed_audio[b.group] = a if prev is None else prev + a
            peak_level = 1.0 if b.events_voice else float(np.max(lvl))
            bed_peak[b.group] = max(bed_peak.get(b.group, 0.0), peak_level * 10 ** (b.gain_db / 20))
    info["beds"] = bed_info
    timings["voices"] = time.time() - t0

    # 5. gains: fixed per group, relative to the melody's active RMS
    def fit(a):
        a = a[:n_total]
        return np.pad(a, ((0, n_total - len(a)), (0, 0))) if len(a) < n_total else a

    stems = {k: (g, fit(a)) for k, (g, a) in stems.items()}
    if ctrl_audio is not None:
        ctrl_audio = fit(ctrl_audio)
    gains_db = {**DEFAULT_GAINS_DB, **cfg.gains_db}
    mel = [a for k, (g, a) in stems.items() if g == "melody"]
    ref = active_rms(sum(mel)) if mel else max(active_rms(a) for _, a in stems.values())
    lvl_info = {}
    scaled: dict[str, tuple[str, np.ndarray]] = {}
    for k, (g, a) in stems.items():
        r = active_rms(a if g not in cfg.split else sum(x for kk, (gg, x) in stems.items() if gg == g))
        gain = (ref * 10 ** (gains_db[g] / 20) / r) if (r > 0 and g != "melody") else 1.0
        scaled[k] = (g, a * np.float32(gain))
        lvl_info[k] = {"group": g, "gain_db": db(gain), "target_db_vs_melody": gains_db[g]}
    for g, a in bed_audio.items():   # voices: loudest stretch -> gains_db below the melody, x the curve's peak
        r = active_rms(fit(a))
        if r <= 0:
            continue
        gain = ref * 10 ** (gains_db[g] / 20) / r * bed_peak.get(g, 1.0)
        scaled[g] = (g, fit(a) * np.float32(gain))
        lvl_info[g] = {"group": g, "gain_db": db(gain), "target_db_vs_melody_at_level1": gains_db[g],
                       "curve_peak": bed_peak.get(g, 1.0)}

    # 5b. lead stays on top: one sidechain gain for the accompaniment bus (events are data hits: never ducked)
    duck_info = None
    if cfg.duck_db is not None and "melody" in scaled:
        acc_names = [k_ for k_, (g, _) in scaled.items() if g in cfg.duck_groups]
        if acc_names:
            acc = sum(scaled[k_][1] for k_ in acc_names)
            gd = duck_gain(scaled["melody"][1], acc, cfg.duck_db, cfg.duck_max_db)
            for k_ in acc_names:
                g, a = scaled[k_]
                scaled[k_] = (g, a * gd[:, None])
            duck_info = {"margin_db": cfg.duck_db, "max_db": cfg.duck_max_db, "groups": sorted({scaled[k_][0] for k_ in acc_names}),
                         "mean_gr_db": float(-20 * np.log10(np.mean(gd))), "max_gr_db": float(-20 * np.log10(gd.min())),
                         "share_of_time_ducked_over_1db": float(np.mean(gd < 10 ** (-1 / 20)))}

    # 6. loudness: one factor for every stem (stems still sum to the pre-limiter mix), then the limiter
    pre = sum(a for _, a in scaled.values())
    L0 = lufs(pre)
    k = 10 ** ((cfg.lufs - L0) / 20) if np.isfinite(L0) else 1.0
    # ... but never push the mix limiter past ~4.5 dB, or any stem's peaks more than 3 dB into its own limiter
    stem_pk = max(float(np.abs(a).max()) for _, a in scaled.values())
    k = min(k, 10 ** (3 / 20) * 0.97 / max(stem_pk, 1e-9),
            10 ** (4.5 / 20) * 10 ** (-1 / 20) / max(float(np.abs(pre).max()), 1e-9))
    final = {}
    stem_gr = {}
    for name, (g, a) in scaled.items():   # a stem whose transients would clip gets a peak limiter
        a = a * np.float32(k)
        if np.abs(a).max() > 0.97:
            a, stem_gr[name] = limiter(a, ceiling_db=20 * np.log10(0.97))
        final[name] = (g, a)
    pre = sum(a for _, a in final.values()) if stem_gr else pre * np.float32(k)
    if ctrl_audio is not None:   # same gain (and the same peak limiter) as the melody stem
        ctrl_audio = ctrl_audio * np.float32(k)
        if np.abs(ctrl_audio).max() > 0.97:
            ctrl_audio, _ = limiter(ctrl_audio, ceiling_db=20 * np.log10(0.97))
    mix, gr = limiter(pre)
    L1 = lufs(mix)
    mix_ctrl = None
    if ctrl_audio is not None and "melody" in final:   # melody gain is 1, so the control needs none
        alt = pre - final["melody"][1] + ctrl_audio
        mix_ctrl, _ = limiter(alt)

    # 7. write
    for f in out.glob("*.wav"):
        if not f.name.startswith("orig_"):
            f.unlink()
    stem_list = []
    for name, (g, a) in sorted(final.items(), key=lambda kv: (GROUPS.index(kv[1][0]), kv[0])):
        pk = float(np.abs(a).max())
        if pk > 0.99:
            print(f"[arrange] warning: stem {name} peaks at {pk:.2f}; clipped in the WAV")
        write_wav(out / f"{name}.wav", a)
        lvl_info[name].update({"active_rms_db": db(active_rms(a)), "peak_db": db(pk)})
        stem_list.append({"name": name, "file": f"{name}.wav", "group": g, "label": _label(name, g, cfg)})
    write_wav(out / "mix.wav", mix)
    if ctrl_audio is not None:
        write_wav(out / "control_shuffled.wav", ctrl_audio)
        write_wav(out / "mix_control.wav", mix_ctrl)

    # 8. timeline + score (data notes exactly as read)
    tl = sorted(notes, key=lambda n: (n.start, n.voice, n.pitch))
    tl_json = [json_safe(asdict(n)) for n in tl]
    (out / "timeline.json").write_text(json.dumps(tl_json))
    assert fingerprint([n for n in load_notes(out / "timeline.json") if is_data(n, cfg)]) == data_fp

    # 9. fidelity before/after
    t0 = time.time()
    fid = {}
    use_mix = cfg.mix_fidelity if mix_fidelity is None else mix_fidelity
    for spec in cfg.fidelity:
        if not fid_notes(notes, spec):
            continue
        rng = pyin_range(notes, spec)
        lead_stem = final.get(spec.stem, final.get("melody", (None, None)))[1]
        r = {"voices": list(spec.voices), "pitch_map": spec.pm,
             "final_stem": fidelity(lead_stem, notes, spec, track=pyin_track(lead_stem, *rng))}
        if spec.orig_stem and (out / spec.orig_stem).exists():
            import soundfile as sf
            o, osr = sf.read(str(out / spec.orig_stem), dtype="float32", always_2d=True)
            r["before_stem"] = fidelity(o, notes, spec, track=pyin_track(o, *rng, sr=osr))
        if use_mix and not spec.solo:
            r["final_mix"] = fidelity(mix, notes, spec, track=pyin_track(mix, *rng))
        if spec is spec0 and ctrl_audio is not None:
            ctrl_lead = shuffled_control(notes, spec.voices, cfg.seed)
            truth = fid_notes(notes, spec)
            r["control_stem"] = fidelity(ctrl_audio, ctrl_lead, spec, truth_notes=truth,
                                         track=pyin_track(ctrl_audio, *rng))
        fid[spec.label] = r
    timings["fidelity"] = time.time() - t0

    # 10. reports
    timings["total"] = time.time() - t_all
    loud = {"ducking": duck_info, "stem_limiter_gr_db": stem_gr, "lufs_pre_norm": L0, "lufs_target": cfg.lufs, "lufs_mix": L1, "limiter_max_gr_db": gr,
            "mix_peak_db": db(float(np.abs(mix).max())), "norm_gain_db": db(k)}
    rep = {**info, "engines": sorted(engines), "levels": lvl_info, "loudness": loud, "fidelity": fid,
           "timings_s": {k_: round(v, 1) for k_, v in timings.items()},
           "data_notes": len(data_notes), "data_notes_unchanged": True,
           "model_notes": sum(1 for n in notes if not is_data(n, cfg)), "notes": cfg.notes}
    (out / "arrange_report.json").write_text(json.dumps(json_safe(rep), indent=1))
    _update_track(out, track, cfg, stem_list, fid, loud, info, ctrl_audio is not None, notes)
    if plot:
        try:
            _plot(out, final, mix, notes, cfg, duration)
        except Exception as e:  # pragma: no cover
            print(f"[arrange] plot failed: {e}")
    if verbose:
        print(json.dumps(json_safe({"id": cfg.id, "loudness": loud, "timings": rep["timings_s"],
                                    "fidelity": {k_: {kk: (v.get("r2"), v.get("pitch_within_half_semitone"))
                                                      for kk, v in r.items() if isinstance(v, dict) and "n" in v}
                                                 for k_, r in fid.items()}}), indent=1))
    return rep


def _slug(v: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in v.lower()).strip("-")


def _label(name: str, g: str, cfg: Arrangement) -> str:
    base = {"melody": "Data melody", "bands": "Bands (data)", "events": "Events (data)",
            "harmony": "Harmony and bass (model-written)", "geophony": "Geophony (recorded Earth sounds)",
            "anthrophony": "Anthrophony (recorded human sounds)", "biophony": "Biophony (recorded living sounds)"}[g]
    if name != g:
        return f"{base}: {name.split('_', 1)[1].replace('-', ' ')}"
    return base


GM = {73: "flute", 71: "clarinet", 48: "strings", 49: "slow strings", 89: "warm pad", 52: "choir", 46: "harp",
      45: "pizzicato strings", 0: "piano", 32: "acoustic bass", 33: "finger bass", 115: "wood block",
      11: "vibraphone", 42: "cello", 68: "oboe", 12: "marimba", 88: "new-age pad", 70: "bassoon", 61: "brass"}


def _update_track(out, track, cfg, stem_list, fid, loud, info, has_ctrl, notes):
    t = dict(track)
    t["stems"] = stem_list
    t["mix"] = "mix.wav"
    voices = sorted({n.voice for n in notes})
    t["voice_groups"] = {v: group_of(v, cfg) for v in voices}
    t["lead_voices"] = list(cfg.fidelity[0].voices) if cfg.fidelity and cfg.fidelity[0].stem == "melody" else []
    if t["lead_voices"]:
        t["lead_voice"] = t["lead_voices"][0]
        t["pitch_map"] = cfg.fidelity[0].pm
    if has_ctrl:
        t["control"] = {"file": "control_shuffled.wav", "replaces": "melody",
                        "label": "Same lead notes, shuffled in time (same arrangement)", "mix": "mix_control.wav"}
    if track.get("fidelity"):
        t["fidelity_builder"] = track["fidelity"]
    if cfg.fidelity:
        s0 = fid[cfg.fidelity[0].label]
        fs, fm, fc = s0.get("final_stem", {}), s0.get("final_mix", {}), s0.get("control_stem", {})
        if not t["lead_voices"]:   # no single lead (5669): no headline R², per-series checks in detail
            fs, fm, fc = {}, {}, {}
        t["fidelity"] = {"r2": fs.get("r2"), "pearson_r": fs.get("pearson_r"), "mae": fs.get("mae"), "n": fs.get("n"),
                         "control_r2": fc.get("r2"), "mix_r2": fm.get("r2"),
                         "pitch_within_half_semitone": fs.get("pitch_within_half_semitone"),
                         "mix_pitch_within_half_semitone": fm.get("pitch_within_half_semitone"),
                         "method": "note-level register-bounded pyin on the final rendered audio",
                         "series": cfg.fidelity[0].label,
                         "detail": {k: {kk: vv for kk, vv in v.items() if kk not in ("pitch_map",)} for k, v in fid.items()}}
    progs = {}
    for v in voices:
        g = group_of(v, cfg)
        if g in ("melody", "bands", "events") and v not in cfg.drums:
            progs[v] = GM.get(cfg.programs.get(v, DEFAULT_PROGRAMS[g]), f"GM {cfg.programs.get(v, DEFAULT_PROGRAMS[g])}")
    t["arrangement"] = {"engine": "FluidSynth + FluidR3_GM (MIT)", "instruments": progs,
                        "harmony": ({"model": info["harmony"]["model"], "seed": info["harmony"]["seed"],
                                     "notes": info["harmony"]["notes"]} if "harmony" in info else None),
                        "form": ({"sections": info["form"]["sections"]} if "form" in info else None),
                        "voices": [b for b in info.get("beds", []) if "skipped" not in b],
                        "loudness_lufs": round(loud["lufs_mix"], 1), "notes": cfg.notes}
    legend = list(t.get("legend") or [])
    have = " ".join(str(x.get("voice", "")).lower() for x in legend)
    if "harmony" in info and "harmony" not in have:
        legend.append({"voice": "harmony and bass (model)", "meaning": "written by the Anticipatory Music Transformer "
                       "around the data melody; it never changes a data note and carries no data"})
    if "form" in info:
        legend.append({"voice": "cymbal swells (form)", "meaning": "section changes found in the video frames "
                       "(DINOv2 novelty); cues only, no data"})
    t["legend"] = legend
    for old_s, new_s in cfg.text_subs:
        for key in ("description", "summary"):
            if isinstance(t.get(key), str):
                t[key] = t[key].replace(old_s, new_s)
        for item in t["legend"]:
            for key in ("voice", "meaning"):
                if isinstance(item.get(key), str):
                    item[key] = item[key].replace(old_s, new_s)
    (out / "track.json").write_text(json.dumps(json_safe(t), indent=1, ensure_ascii=False))


def _plot(out, final, mix, notes, cfg, duration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    hop = int(0.1 * SR)
    for name, (g, a) in final.items():
        m = a.mean(1)
        k = len(m) // hop
        e = 10 * np.log10((m[: k * hop].reshape(k, hop) ** 2).mean(1) + 1e-10)
        ax[0].plot(np.arange(k) * 0.1, e, label=name, lw=1.4 if g == "melody" else 0.9)
    ax[0].set_ylim(-70, 0); ax[0].set_ylabel("RMS dBFS (0.1 s)"); ax[0].legend(fontsize=7, ncol=4)
    ax[0].set_title(f"{cfg.id}: stem levels at mix gain")
    from scipy.signal import spectrogram
    f, tt, S = spectrogram(mix.mean(1), SR, nperseg=2048, noverlap=1024)
    sel = f < 5000
    ax[1].pcolormesh(tt, f[sel], 10 * np.log10(S[sel] + 1e-12), shading="auto", cmap="magma", vmin=-110, vmax=-30)
    ax[1].set_ylabel("Hz"); ax[1].set_title("mix spectrogram (0-5 kHz)")
    for n in notes:
        g = group_of(n.voice, cfg)
        c = {"melody": "#d9480f", "bands": "#1c7ed6", "events": "#2b8a3e", "harmony": "#adb5bd"}[g]
        ax[2].add_patch(plt.Rectangle((n.start, n.pitch - 0.4), max(n.end - n.start, 0.05), 0.8, color=c,
                                      alpha=0.9 if g == "melody" else 0.45, lw=0))
    ps = [n.pitch for n in notes]
    ax[2].set_ylim(min(ps) - 2, max(ps) + 2); ax[2].set_xlim(0, duration)
    ax[2].set_ylabel("MIDI"); ax[2].set_xlabel("s"); ax[2].set_title("score: orange data melody, blue bands, green events, grey model/form")
    fig.tight_layout(); fig.savefig(out / "arrange.png", dpi=90); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="outputs/<id> directory")
    ap.add_argument("--id", help="arrangement id (default: directory name)")
    ap.add_argument("--no-harmony", action="store_true")
    ap.add_argument("--no-mix-fidelity", action="store_true")
    ap.add_argument("--model", help="override the AMT checkpoint (e.g. stanford-crfm/music-small-800k)")
    a = ap.parse_args(argv)
    from .arrangements import get
    cfg = get(a.id or a.out.name, a.out)
    if a.model and cfg.harmony is not None:
        cfg.harmony = {**cfg.harmony, "model": a.model}
    arrange(a.out, cfg, harmony=not a.no_harmony, mix_fidelity=False if a.no_mix_fidelity else None)


if __name__ == "__main__":
    main()
