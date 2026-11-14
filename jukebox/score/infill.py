"""Model-written harmony and bass around FIXED data notes (Anticipatory Music Transformer).

The data melody (voice "lead") goes to the model as anticipated *controls*: the model sees
each lead note DELTA=5 s before it sounds and writes accompaniment that leads into it
(Thickstun et al. 2023, "Anticipatory Music Transformer"). Lead notes are never edited or
returned; `infill` returns only NEW notes, so the data -> pitch mapping stays auditable.

Constraints are applied at sampling time, by masking the note token after the model has
chosen the onset and duration (so the remaining probability mass is still the model's):
- instruments: only the requested GM programs, in passes (default: piano/strings "harmony", then "bass")
- register: harmony stays `below` semitones under every lead note it overlaps; bass in E1..E3
- consonance: no minor 2nd / tritone / major 7th against an overlapping lead note
- key: pitch classes of the (inferred) major key of the lead

    from jukebox.score.infill import infill
    new = infill(lead_notes, duration=66.5, model="stanford-crfm/music-small-800k", seed=0)
"""
from __future__ import annotations

import math
import time as _time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from anticipation.config import DELTA, MAX_DUR, MAX_PITCH, MAX_TIME, TIME_RESOLUTION
from anticipation.vocab import (ADUR_OFFSET, ANOTE_OFFSET, ANTICIPATE, ATIME_OFFSET, AUTOREGRESS,
                                CONTROL_OFFSET, DUR_OFFSET, NOTE_OFFSET, REST, TIME_OFFSET)

from .mapping import SCALES, Note

DEFAULT_MODEL = "stanford-crfm/music-medium-800k"
FAST_MODEL = "stanford-crfm/music-small-800k"
# GM program (0-based) the model writes with -> our voice name
DEFAULT_PASSES = [{0: "harmony", 48: "harmony"},          # pass 1: piano or string ensemble
                  {32: "bass", 33: "bass"}]                # pass 2: acoustic / finger bass
DEFAULT_RANGES = {"harmony": (43, 88), "bass": (28, 52)}  # G2..E6, E1..E3
DISSONANT = (1, 6, 11)                                     # m2, tritone, M7 (mod 12)


@dataclass
class InfillStats:
    events: int = 0
    rests: int = 0
    seconds: float = 0.0
    tokens: int = 0
    slides: int = 0
    dropped: int = 0


LAST_STATS = InfillStats()


# ---------------------------------------------------------------- helpers

def infer_key(notes: list[Note]) -> tuple[int, str]:
    """Major key whose scale best covers the lead (duration-weighted), tonic-weighted tie-break."""
    w = np.zeros(12)
    for n in notes:
        w[n.pitch % 12] += max(n.end - n.start, 1e-3)
    best = None
    for root in range(12):
        pcs = {(root + s) % 12 for s in SCALES["major"]}
        outside = sum(w[p] for p in range(12) if p not in pcs)
        score = (outside, -w[root] - 0.5 * w[(root + 7) % 12])
        if best is None or score < best[0]:
            best = (score, root)
    return best[1], "major"


def _key_mask(key) -> np.ndarray:
    m = np.ones(12, bool)
    if key is not None:
        root, scale = key
        m[:] = False
        for s in SCALES[scale]:
            m[(root + s) % 12] = True
    return m


class _Lead:
    """Lead notes in ticks, for register/consonance queries."""

    def __init__(self, notes: list[Note]):
        self.s = np.array([round(n.start * TIME_RESOLUTION) for n in notes], np.int64)
        self.e = np.array([round(n.end * TIME_RESOLUTION) for n in notes], np.int64)
        self.p = np.array([n.pitch for n in notes], np.int64)
        self.v = np.array([n.velocity for n in notes], float)
        self.mid = (self.s + self.e) / 2

    def overlapping(self, t0: int, t1: int) -> np.ndarray:
        if len(self.p) == 0:
            return np.zeros(0, int)
        return np.nonzero((self.s < max(t1, t0 + 1)) & (self.e > t0))[0]

    def nearest(self, t: int) -> int | None:
        if len(self.p) == 0:
            return None
        return int(np.argmin(np.abs(self.mid - t)))


def _allowed_pitches(lead: _Lead, t0: int, t1: int, voice: str, ranges, key_mask, below: int,
                     avoid) -> np.ndarray:
    lo, hi = ranges[voice]
    ok = np.zeros(MAX_PITCH, bool)
    ok[lo:hi + 1] = True
    idx = lead.overlapping(t0, t1)
    ref = idx if len(idx) else ([lead.nearest(t0)] if lead.nearest(t0) is not None else [])
    if len(ref):
        cap = int(lead.p[ref].min()) - below
        ok[max(cap + 1, 0):] = False
    pcs = key_mask.copy()
    for i in idx:
        for iv in avoid:
            pcs[(lead.p[i] + iv) % 12] = False
            pcs[(lead.p[i] - iv) % 12] = False
    ok &= pcs[np.arange(MAX_PITCH) % 12]
    return ok


def _nucleus_sample(logits: torch.Tensor, top_p: float, gen: torch.Generator) -> int:
    probs = torch.softmax(logits, -1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, -1)
        drop = cum - sp > top_p          # keep the smallest prefix whose mass reaches top_p
        sp[drop] = 0
        probs = torch.zeros_like(probs).scatter(0, si, sp)
    return int(torch.multinomial(probs / probs.sum(), 1, generator=gen))


class _Context:
    """Token history with a KV cache; times are relativized to the window like anticipation.sample."""

    def __init__(self, model, z: int, max_len: int = 1020, keep: int = 600):
        self.model, self.z, self.max_len, self.keep = model, z, max_len, keep - keep % 3
        self.tokens: list[int] = []   # absolute-time triples (events and controls)
        self.start = 0                # window start in self.tokens (multiple of 3)
        self.offset = 0               # time offset subtracted from time tokens in the window
        self.past = None
        self.fed = 0                  # tokens of the window already in the cache
        self.slides = 0
        self.n_forward_tokens = 0

    def _rel(self, i: int) -> int:
        tok = self.tokens[i]
        return tok - self.offset if (i % 3 == 0) else tok

    def _window_min_time(self) -> int:
        times = [self.tokens[i] - (ATIME_OFFSET if self.tokens[i] >= CONTROL_OFFSET else TIME_OFFSET)
                 for i in range(self.start, len(self.tokens) - 2, 3)]
        return min(times) if times else 0

    def maybe_slide(self, current_time: int, lookahead: int):
        """Call at a triple boundary. Slide when the context or the relative clock would overflow."""
        n = len(self.tokens) - self.start
        too_long = 1 + n + 3 > self.max_len
        too_late = current_time + lookahead - self.offset >= MAX_TIME - 1
        if not (too_long or too_late):
            return
        keep = self.keep
        while True:
            self.start = max(0, len(self.tokens) - keep)
            self.start -= self.start % 3
            self.offset = min(self._window_min_time(), current_time)
            if current_time + lookahead - self.offset < MAX_TIME - 1 or keep <= 3:
                break
            keep //= 2
        self.past, self.fed = None, 0
        self.slides += 1

    def logits(self) -> torch.Tensor:
        new = [self._rel(i) for i in range(self.start + self.fed, len(self.tokens))]
        if self.past is None:
            new = [self.z] + new
        ids = torch.tensor([new], dtype=torch.long, device=self.model.device)
        with torch.no_grad():
            out, self.past = self.model(ids, self.past)
        self.fed = len(self.tokens) - self.start
        self.n_forward_tokens += len(new)
        return out[0, -1].float().cpu()


def _controls(notes: list[Note], programs: list[int]) -> list[tuple[int, int, int]]:
    out = []
    for n, program in sorted(zip(notes, programs), key=lambda x: (x[0].start, x[0].pitch)):
        t = round(n.start * TIME_RESOLUTION)
        d = int(np.clip(round((n.end - n.start) * TIME_RESOLUTION), 1, MAX_DUR - 1))
        out.append((ATIME_OFFSET + t, ADUR_OFFSET + d, ANOTE_OFFSET + int(program) * MAX_PITCH + int(n.pitch)))
    out.sort()
    return out


def _resolve_model(model, device):
    if isinstance(model, str):
        from .amt_model import load
        return load(model, device)
    return model


# ---------------------------------------------------------------- main API

def infill(notes: list[Note], duration: float, *, model=DEFAULT_MODEL, top_p: float = 0.95, seed: int = 0,
           instruments: dict[int, str] | list[dict[int, str]] | None = None, lead_voices=("lead",),
           lead_program: int = 73, key="auto", below: int = 3, avoid=DISSONANT,
           ranges: dict[str, tuple[int, int]] | None = None, velocity_scale: dict[str, float] | None = None,
           pans: dict[str, float] | None = None, min_dur: float = 0.08, device: str = "cpu",
           max_events: int | None = None, allow_rest: bool = False,
           instrument_bias: dict[int, float] | None = None, min_ticks: int = 5, mono_voices=("bass",),
           verbose: bool = False) -> list[Note]:
    """Write accompaniment around the fixed lead notes. Returns NEW notes only (value=nan).

    notes        all score notes; those whose voice is in `lead_voices` become controls, others are ignored
    duration     seconds; nothing new starts at or after it, and new notes are clipped to it
    model        HF repo id / local dir of an AMT checkpoint, or a loaded `amt_model.GPT2`-like callable
    instruments  {GM program: voice} the model may write, or a LIST of such dicts = sequential passes:
                 each pass sees the lead plus every earlier pass as controls. Default: a harmony pass
                 (piano or strings, model's choice) then a bass pass (acoustic or finger bass).
                 (A single joint pass rarely produces bass: the model commits to one instrument.)
    lead_program GM program the lead is presented to the model as (73 = flute, a typical melody part)
    key          "auto" (infer major key from the lead), None (no key filter) or (root_pc, scale_name)
    below        harmony stays at least this many semitones under each overlapping lead note
    avoid        intervals (mod 12) forbidden against overlapping lead notes
    allow_rest   let the model emit REST padding. Off by default: with only controls in context the
                 model tends to "wait" with rests for the whole piece (dur-0 + REST every second)
    instrument_bias  {GM program: log-prob bonus} at the note token
    min_ticks    drop model notes shorter than this many 10 ms ticks. With REST and drums masked the
                 model sometimes spends a near-zero duration (meant for a drum hit) on a pitched note
    mono_voices  voices made monophonic afterwards (default bass: the model likes chordal bass)
    """
    global LAST_STATS
    t_start = _time.time()
    passes = instruments if isinstance(instruments, list) else [instruments] if instruments else DEFAULT_PASSES
    lead = [n for n in notes if n.voice in lead_voices]
    if key == "auto":
        key = infer_key(lead) if lead else None
    m = _resolve_model(model, device)
    gen = torch.Generator().manual_seed(int(seed))
    stats = InfillStats()
    new: list[Note] = []
    for inst in passes:
        ctrl_notes = lead + new
        ctrl_progs = [lead_program] * len(lead) + [n.source["program"] for n in new]
        new += _infill_pass(m, lead, ctrl_notes, ctrl_progs, duration, dict(inst), gen, stats, top_p=top_p,
                            seed=seed, key=key, below=below, avoid=avoid, ranges=ranges,
                            velocity_scale=velocity_scale, pans=pans, min_dur=min_dur, max_events=max_events,
                            allow_rest=allow_rest, ibias=dict(instrument_bias or {}), min_ticks=min_ticks,
                            mono_voices=tuple(mono_voices))
    new.sort(key=lambda n: (n.start, n.voice, n.pitch))
    stats.events, stats.seconds = len(new), _time.time() - t_start
    LAST_STATS = stats
    if verbose:
        print(f"[infill] {getattr(m, 'name', '?')}: {len(new)} notes ({stats.rests} rests, {stats.dropped} too short) in {stats.seconds:.1f}s, "
              f"{stats.tokens} tokens fed, {stats.slides} window slides, key={key}, "
              f"programs={sorted(Counter(n.source['program'] for n in new).items())}")
    return new


def _dedup(events):
    """Same program + pitch sounding twice: drop exact re-strikes at the same onset (keep the
    longest), and cut an earlier note where the same pitch re-enters."""
    best = {}
    for e in events:
        k = (e[0], e[2], e[3])
        if k not in best or e[1] > best[k][1]:
            best[k] = e
    ev = sorted(best.values())
    out = []
    last = {}  # (prog, pitch) -> index in out
    for t, d, prog, pitch in ev:
        j = last.get((prog, pitch))
        if j is not None and out[j][0] + out[j][1] > t:
            pt, pd, pp, pq = out[j]
            out[j] = (pt, t - pt, pp, pq)
        last[(prog, pitch)] = len(out)
        out.append((t, d, prog, pitch))
    return out


def _monophonic(events):
    """One note at a time (lowest wins at a shared onset; a new onset cuts the sounding note)."""
    first = {}
    for e in sorted(events, key=lambda e: (e[0], e[3])):
        first.setdefault(e[0], e)
    out = []
    for e in sorted(first.values()):
        if out and out[-1][0] + out[-1][1] > e[0]:
            t, d, p, q = out[-1]
            out[-1] = (t, e[0] - t, p, q)
        out.append(e)
    return out


def _infill_pass(m, lead, ctrl_notes, ctrl_progs, duration, instruments, gen, stats, *, top_p, seed, key,
                 below, avoid, ranges, velocity_scale, pans, min_dur, max_events, allow_rest, ibias,
                 min_ticks, mono_voices):
    ranges = {**DEFAULT_RANGES, **(ranges or {})}
    for v in instruments.values():
        ranges.setdefault(v, (36, 96))
    vscale = {"harmony": 0.75, "bass": 0.9, **(velocity_scale or {})}
    pans = {"harmony": -0.25, "bass": 0.0, **(pans or {})}
    kmask = _key_mask(key)
    L = _Lead(lead)
    model_name = getattr(m, "name", type(m).__name__)

    end_tick = int(round(duration * TIME_RESOLUTION))
    delta = DELTA * TIME_RESOLUTION
    controls = _controls(ctrl_notes, ctrl_progs)
    ctx = _Context(m, ANTICIPATE if controls else AUTOREGRESS)
    max_events = max_events or int(40 * duration) + 50

    events: list[tuple[int, int, int, int]] = []  # (time, dur, program, pitch)
    rests = 0
    ci = 0
    current = 0
    while len(events) + rests < max_events:
        # 1. reveal every control whose time is within DELTA of the present
        while ci < len(controls) and current >= controls[ci][0] - ATIME_OFFSET - delta:
            ctx.tokens.extend(controls[ci])
            ci += 1
        ctx.maybe_slide(current, delta + MAX_DUR)

        # 2. onset: not in the past, inside the vocab
        lg = ctx.logits()
        mask = torch.full_like(lg, -math.inf)
        lo_rel = max(current - ctx.offset, 0)
        mask[TIME_OFFSET + lo_rel:TIME_OFFSET + MAX_TIME] = 0
        rel_t = _nucleus_sample(lg + mask, top_p, gen)
        t = rel_t - TIME_OFFSET + ctx.offset
        if t >= end_tick:
            break
        ctx.tokens.append(rel_t + ctx.offset)  # stored with absolute time

        # 3. duration (dur 0 only pairs with REST)
        lg = ctx.logits()
        mask = torch.full_like(lg, -math.inf)
        mask[DUR_OFFSET + (0 if allow_rest else 1):DUR_OFFSET + MAX_DUR] = 0
        dtok = _nucleus_sample(lg + mask, top_p, gen)
        d = dtok - DUR_OFFSET
        ctx.tokens.append(dtok)

        # 4. note: only allowed instruments, register and pitch classes
        lg = ctx.logits()
        t1 = min(t + max(d, 1), end_tick)
        mask = torch.full_like(lg, -math.inf)
        if allow_rest:
            mask[REST] = 0
        for prog, voice in instruments.items():
            ok = _allowed_pitches(L, t, t1, voice, ranges, kmask, below if voice != "bass" else 0, avoid)
            ids = NOTE_OFFSET + prog * MAX_PITCH + torch.from_numpy(np.nonzero(ok)[0])
            mask[ids] = float(ibias.get(prog, 0.0))
        if not torch.isfinite(mask).any():
            mask[REST] = 0
        ntok = _nucleus_sample(lg + mask, top_p, gen)
        ctx.tokens.append(ntok)
        current = t
        if ntok == REST:
            rests += 1
            continue
        note = ntok - NOTE_OFFSET
        events.append((t, d, note // MAX_PITCH, note % MAX_PITCH))

    stats.rests += rests
    stats.tokens += ctx.n_forward_tokens
    stats.slides += ctx.slides
    stats.dropped += sum(d < min_ticks for _, d, _, _ in events)
    events = _dedup([e for e in events if e[1] >= min_ticks])
    for prog, voice in instruments.items():
        if voice in mono_voices:
            mine = [e for e in events if e[2] == prog]
            events = [e for e in events if e[2] != prog] + _monophonic(mine)
    out = []
    for t, d, prog, pitch in events:
        voice = instruments[prog]
        s = t / TIME_RESOLUTION
        e = min(s + max(d / TIME_RESOLUTION, min_dur), duration)
        if s >= duration or e <= s:
            continue
        j = L.nearest(t)
        lv = L.v[j] if j is not None else 70.0
        vel = int(np.clip(round(lv * vscale.get(voice, 0.8)), 25, 115))
        out.append(Note(float(s), float(e), int(pitch), vel, voice, float("nan"), float(pans.get(voice, 0.0)),
                        {"model": model_name, "program": int(prog), "seed": int(seed), "top_p": float(top_p),
                         "key": list(key) if key else None}))
    return out


# ---------------------------------------------------------------- checks

def clash_report(lead: list[Note], new: list[Note], below: int = 3, avoid=DISSONANT) -> dict:
    """How often new notes clash with / rise above the overlapping lead note (0 is the goal)."""
    L = _Lead(lead)
    pairs = clashes = above = 0
    for n in new:
        idx = L.overlapping(round(n.start * TIME_RESOLUTION), round(n.end * TIME_RESOLUTION))
        for i in idx:
            pairs += 1
            iv = abs(int(n.pitch) - int(L.p[i])) % 12
            clashes += int(iv in avoid)
            if n.voice != "bass":
                above += int(n.pitch > L.p[i] - below)
    # share of bass onsets whose pitch class is also struck by the harmony at that onset (chord tone)
    harm = {(round(n.start, 2), n.pitch % 12) for n in new if n.voice == "harmony"}
    bass = [n for n in new if n.voice == "bass"]
    chord_tone = sum((round(n.start, 2), n.pitch % 12) in harm for n in bass)
    return {"new_notes": len(new), "overlap_pairs": pairs,
            "bass_on_harmony_pc": chord_tone / len(bass) if bass else 0.0, "dissonant_pairs": clashes, "above_lead": above,
            "per_voice": {v: sum(n.voice == v for n in new) for v in sorted({n.voice for n in new})}}
