"""Form layer: data notes must never change; sections/regions behave on synthetic videos."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jukebox.form.apply import CYMBAL_VOICE, MOTIF_VOICE, apply_form, is_data  # noqa: E402
from jukebox.form.embed import embed_frames  # noqa: E402
from jukebox.form.sections import analyse, foote_novelty, self_similarity  # noqa: E402
from jukebox.score.mapping import Note, PitchMap, melody_from_curve  # noqa: E402


def _abab_frames(seg=12, h=32, w=64, seed=0):
    rng = np.random.default_rng(seed)
    cols = [(200, 40, 40), (40, 60, 200), (200, 40, 40), (40, 60, 200)]
    out = []
    for c in cols:
        for _ in range(seg):
            f = np.clip(np.array(c)[None, None] + rng.normal(0, 8, (h, w, 3)), 0, 255)
            out.append(f.astype(np.uint8))
    return np.stack(out)


def _score():
    t = np.arange(48) * 0.5
    pm = PitchMap(0, 1, low=60, span=24, root=2)
    lead = melody_from_curve(t, np.sin(t) * .5 + .5, pm, "lead")
    band = melody_from_curve(t, np.cos(t) * .5 + .5, pm, "band 60N..30N", step=2.0)
    harm = [Note(float(s), float(s) + 1.9, 50 + (i % 5) * 2, 60, "harmony", float("nan")) for i, s in enumerate(t[::4])]
    tagged = [Note(1.0, 2.0, 61, 70, "infill", 0.3, source={"data": True})]
    return lead + band + harm + tagged


def _key(n):
    return (n.voice, n.start, n.end, n.pitch, n.velocity, n.pan, None if n.value != n.value else n.value)


def test_data_notes_unchanged_for_all_options():
    notes = _score()
    data_before = sorted(_key(n) for n in notes if is_data(n))
    form = [(0.0, 6.0, "A"), (6.0, 12.0, "B"), (12.0, 18.0, "A"), (18.0, 24.0, "C")]
    for kw in [{}, {"instrument_change": True}, {"transpose": True, "steps": [5, 7, -7]},
               {"phrase_break": True, "break_s": 1.0}]:
        new, fj = apply_form(notes, form, **kw)
        data_after = sorted(_key(n) for n in new if is_data(n))
        assert data_after == data_before, kw
        assert {"lead", "band 60N..30N", "infill"} <= set(fj["data_voices_protected"])
    # inputs are not mutated
    assert sorted(_key(n) for n in notes if is_data(n)) == data_before


def test_non_data_voices_do_change_and_cues_added():
    notes = _score()
    form = [(0.0, 6.0, "A"), (6.0, 12.0, "B"), (12.0, 18.0, "A"), (18.0, 24.0, "C")]
    new, fj = apply_form(notes, form, instrument_change=True)
    assert any(n.voice == CYMBAL_VOICE for n in new)
    motif = [n for n in new if n.voice == MOTIF_VOICE]
    first = [n.pitch for n in motif if n.source["section"] == 0]
    ret = [n.pitch for n in motif if n.source["section"] == 2]
    assert first == ret and first, "a return must recall the first statement exactly"
    assert fj["returns"] == [{"section": 2, "label": "A", "time": 12.0, "first_section": 0}]
    assert any(n.voice.startswith("harmony|") for n in new)
    json.dumps(fj)  # serialisable


def test_novelty_finds_abab_boundaries_and_returns():
    frames = _abab_frames()
    emb = embed_frames(frames, method="handcrafted")
    t = np.arange(len(frames)) * 0.5
    form = analyse(emb.X, t, kernel_s=4.0, min_section_s=3.0)
    starts = [s.start_frame for s in form.sections[1:]]
    assert all(min(abs(b - x) for x in starts) <= 1 for b in (12, 24, 36)), starts
    labels = [s.label for s in form.sections]
    assert labels == ["A", "B", "A", "B"], labels
    assert [s.index for s in form.returns] == [2, 3]


def test_checkerboard_peak_location():
    X = np.r_[np.zeros((20, 3)) + [1, 0, 0], np.zeros((20, 3)) + [0, 1, 0]]
    nov = foote_novelty(self_similarity(X), 5)
    assert abs(int(np.argmax(nov)) - 20) <= 1
