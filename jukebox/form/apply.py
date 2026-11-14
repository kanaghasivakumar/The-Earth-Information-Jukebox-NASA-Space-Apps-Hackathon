"""Apply musical form to a score WITHOUT touching the data.

    from jukebox.form.apply import apply_form
    new_notes, form_json = apply_form(notes, form)          # form: sections.Form or form.json dict

Rules
- Data voices (lead, band*, events, or any note with source["data"] truthy, or voices listed in
  `protect`) are returned as exact copies: same start, end, pitch, velocity, value, pan.
- Non-data voices (harmony, bass, AMT infill, ...) may be
    * shortened before a section boundary (phrase break, `break_s` of silence),
    * transposed per section label by whole SCALE steps (stays in key), and/or
    * moved to a per-section instrument (voice renamed "<voice>|<label>", program in form_json).
- New cue notes are added in non-data voices only:
    * "form cymbal": a crescendo roll into each boundary plus a crash on it (GM drums),
    * "form motif": the section's motif at its start; a RETURN replays the first statement exactly.
"""
from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np

from ..score.mapping import SCALES, Note

DATA_VOICES = ("lead", "events")
DATA_PREFIXES = ("band",)
CYMBAL_VOICE = "form cymbal"
MOTIF_VOICE = "form motif"
# per-label scale-step transposition and General MIDI program for non-data voices
DEFAULT_STEPS = [0, 2, -2, 4, -3, 1, -1, 3]
DEFAULT_PROGRAMS = [48, 52, 61, 71, 46, 89, 11, 73]   # strings, choir, brass, clarinet, harp, pad, vibes, flute


def is_data(n: Note, protect=()) -> bool:
    return (n.voice in DATA_VOICES or n.voice.startswith(DATA_PREFIXES) or n.voice in protect
            or bool((n.source or {}).get("data")))


def _sections(form) -> list[dict]:
    if hasattr(form, "sections") and not isinstance(form, dict):
        return [asdict(s) for s in form.sections]
    if isinstance(form, dict) and "sections" in form:
        return list(form["sections"])
    # plain [(start, end, label)]
    out, first = [], {}
    for i, (a, b, lab) in enumerate(form):
        ret = lab in first and out[-1]["label"] != lab
        out.append({"index": i, "start": float(a), "end": float(b), "label": lab, "is_return": ret,
                    "returns_to": first.get(lab)})
        first.setdefault(lab, i)
    return out


def scale_shift(pitch: int, steps: int, root: int, scale: str) -> int:
    """Move a pitch by whole scale steps (snapping to the scale first)."""
    pcs = sorted({(root + s) % 12 for s in SCALES[scale]})
    allowed = np.array([p for p in range(0, 128) if p % 12 in pcs])
    i = int(np.abs(allowed - pitch).argmin()) + int(steps)
    return int(allowed[np.clip(i, 0, len(allowed) - 1)])


def motif_for(label: str, root: int, scale: str, low: int = 72) -> list[tuple[int, float, float, int]]:
    """Deterministic 4-note motif per label: (pitch, beat offset, beats, velocity).
    Same idea as eic-vae/music/memory.py::_default_motif: a broken triad from a label-dependent degree."""
    pcs = sorted({(root + s) % 12 for s in SCALES[scale]})
    allowed = [p for p in range(low, low + 25) if p % 12 in pcs]
    n = len(pcs)
    base = (ord(label[0]) - ord("A")) * 2 % n
    degs = [base, base + 2, base + 4, base + 2]
    return [(allowed[d % len(allowed)], off, dur, vel)
            for d, off, dur, vel in zip(degs, (0.0, 0.5, 1.0, 1.5), (0.45, 0.45, 0.9, 0.45), (72, 64, 78, 62))]


def apply_form(notes: list[Note], form, *, root: int = 2, scale: str = "major_pent", tempo: float = 100.0,
               cymbal: bool = True, phrase_break: bool = True, break_s: float = 0.25,
               transpose: bool = True, instrument_change: bool = False, motifs: bool = True,
               steps: list[int] | None = None, programs: list[int] | None = None,
               protect: tuple[str, ...] = (), swell_s: float = 1.5) -> tuple[list[Note], dict]:
    secs = _sections(form)
    steps = steps or DEFAULT_STEPS
    programs = programs or DEFAULT_PROGRAMS
    labels = []
    for s in secs:
        if s["label"] not in labels:
            labels.append(s["label"])
    lab_step = {lab: steps[i % len(steps)] for i, lab in enumerate(labels)}
    lab_prog = {lab: programs[i % len(programs)] for i, lab in enumerate(labels)}
    bounds = [s["start"] for s in secs[1:]]
    beat = 60.0 / tempo

    def section_at(time: float) -> dict:
        for s in secs:
            if s["start"] <= time < s["end"]:
                return s
        return secs[-1] if time >= secs[-1]["start"] else secs[0]

    out: list[Note] = []
    changed = {"transposed": 0, "shortened": 0, "rerouted": 0}
    voice_programs: dict[str, int] = {}
    for n in notes:
        m = copy.deepcopy(n)
        if is_data(n, protect):
            out.append(m)            # exact copy: pitch, timing, velocity untouched
            continue
        s = section_at(n.start)
        if transpose and lab_step[s["label"]]:
            m.pitch = scale_shift(m.pitch, lab_step[s["label"]], root, scale)
            changed["transposed"] += 1
        if phrase_break:
            for b in bounds:
                if m.start < b - break_s < m.end or (m.start < b < m.end):
                    new_end = max(m.start + 0.05, b - break_s)
                    if new_end < m.end:
                        m.end = new_end
                        changed["shortened"] += 1
                    break
        if instrument_change:
            m.voice = f"{n.voice}|{s['label']}"
            voice_programs[m.voice] = lab_prog[s["label"]]
            changed["rerouted"] += 1
        m.source = {**(n.source or {}), "form_section": s["index"], "form_label": s["label"]}
        out.append(m)

    cues: list[Note] = []
    if cymbal:
        for s in secs[1:]:
            b = s["start"]
            hits = max(3, int(swell_s / 0.125))
            for j in range(hits):          # 16th-note-ish roll, crescendo into the boundary
                tt = b - swell_s + j * swell_s / hits
                if tt < 0:
                    continue
                vel = int(20 + 70 * (j / hits) ** 2)
                cues.append(Note(tt, tt + 0.1, 51, vel, CYMBAL_VOICE, None, 0.0,
                                 {"form": "swell", "section": s["index"]}))
            cues.append(Note(b, b + 1.5, 49 if not s.get("is_return") else 57, 100, CYMBAL_VOICE, None, 0.0,
                             {"form": "crash", "section": s["index"], "return": bool(s.get("is_return"))}))
    if motifs:
        first_motif: dict[str, list] = {}
        for s in secs:
            lab = s["label"]
            mot = first_motif.get(lab) or motif_for(lab, root, scale)
            first_motif.setdefault(lab, mot)
            for p, off, dur, vel in mot:
                st = s["start"] + off * beat
                en = min(st + dur * beat, s["end"])
                if st >= s["end"]:
                    break
                cues.append(Note(st, en, p, vel, MOTIF_VOICE, None, 0.0,
                                 {"form": "motif", "label": lab, "section": s["index"],
                                  "recall": bool(s.get("is_return"))}))
    out += cues
    out.sort(key=lambda n: (n.start, n.voice))

    form_json = {
        "sections": secs,
        "labels": labels,
        "recurrences": {lab: [s["index"] for s in secs if s["label"] == lab] for lab in labels},
        "returns": [{"section": s["index"], "label": s["label"], "time": s["start"], "first_section": s["returns_to"]}
                    for s in secs if s.get("is_return")],
        "cues": {"boundaries": bounds, "cymbal_voice": CYMBAL_VOICE if cymbal else None,
                 "motif_voice": MOTIF_VOICE if motifs else None, "phrase_break_s": break_s if phrase_break else 0},
        "non_data_changes": changed,
        "label_scale_steps": lab_step if transpose else {},
        "programs": {**voice_programs, MOTIF_VOICE: 73, CYMBAL_VOICE: 0},
        "drums": [CYMBAL_VOICE],
        "data_voices_protected": sorted({n.voice for n in notes if is_data(n, protect)}),
    }
    return out, form_json
