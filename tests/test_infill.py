"""Tests for jukebox.score.infill (AMT harmony around fixed data notes) and render.fluid.

Fast tests use a deterministic stand-in model; tests marked `slow` load the real
music-small-800k checkpoint (skipped when it is not in $HF_HOME).

    PYTHONPATH=. python -m pytest tests/test_infill.py -q            # all
    PYTHONPATH=. python -m pytest tests/test_infill.py -q -m "not slow"
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("HF_HOME", "/projects/e32706/omb8654/hf_cache")

from anticipation.config import MAX_TIME
from anticipation.vocab import TIME_OFFSET, VOCAB_SIZE

from jukebox.score import infill as I
from jukebox.score.mapping import Note

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "3912" / "timeline.json"
SMALL = "stanford-crfm/music-small-800k"


class DummyAMT:
    """Deterministic pseudo-random logits that prefer short time steps (dense output)."""
    name = "dummy"
    device = torch.device("cpu")

    def __call__(self, ids, past=None):
        n = (past or 0) + ids.shape[1]
        g = torch.Generator().manual_seed(int(ids[0, -1]) * 7919 + n)
        lg = torch.randn(VOCAB_SIZE, generator=g)
        lg[TIME_OFFSET:TIME_OFFSET + MAX_TIME] -= 0.02 * torch.arange(MAX_TIME)
        return lg.view(1, 1, -1), n


def demo_notes() -> list[Note]:
    return [Note(**d) for d in json.loads(DEMO.read_text())]


def lead_of(notes):
    return [n for n in notes if n.voice == "lead"]


def small_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(SMALL, "pytorch_model.bin")
        return isinstance(p, str) and Path(p).exists()
    except Exception:
        return False


# ------------------------------------------------------------------ fast (stand-in model)

def test_lead_notes_pass_through_unchanged():
    notes = demo_notes()
    before = copy.deepcopy([asdict(n) for n in notes])
    new = I.infill(notes, 66.5, model=DummyAMT(), seed=0)
    assert [asdict(n) for n in notes] == before          # inputs not mutated
    assert new, "no accompaniment generated"
    ids = {id(n) for n in notes}
    assert all(id(n) not in ids for n in new)            # only NEW notes are returned
    assert {n.voice for n in new} <= {"harmony", "bass"}
    assert all(math.isnan(n.value) for n in new)
    merged = lead_of(notes) + new                         # what callers write out
    assert [asdict(n) for n in lead_of(merged)] == [d for d in before if d["voice"] == "lead"]


def test_outputs_within_duration():
    notes = demo_notes()
    for dur in (10.0, 66.5):
        new = I.infill(notes, dur, model=DummyAMT(), seed=1)
        assert new
        for n in new:
            assert 0 <= n.start < dur
            assert n.start < n.end <= dur + 1e-9
            assert 0 <= n.pitch < 128 and 1 <= n.velocity <= 127


def test_seed_is_reproducible():
    notes = demo_notes()
    a = I.infill(notes, 30, model=DummyAMT(), seed=7)
    b = I.infill(notes, 30, model=DummyAMT(), seed=7)
    c = I.infill(notes, 30, model=DummyAMT(), seed=8)
    key = lambda ns: [(n.start, n.end, n.pitch, n.voice, n.velocity) for n in ns]
    assert key(a) == key(b)
    assert key(a) != key(c)


def test_constraints_register_and_consonance():
    notes = demo_notes()
    new = I.infill(notes, 66.5, model=DummyAMT(), seed=2)
    rep = I.clash_report(lead_of(notes), new)
    assert rep["overlap_pairs"] > 0
    assert rep["dissonant_pairs"] == 0
    assert rep["above_lead"] == 0
    key = I.infer_key(lead_of(notes))
    assert key == (2, "major")                           # lead is D major pentatonic
    pcs = {(2 + s) % 12 for s in (0, 2, 4, 5, 7, 9, 11)}
    assert all(n.pitch % 12 in pcs for n in new)
    for n in new:
        lo, hi = I.DEFAULT_RANGES[n.voice]
        assert lo <= n.pitch <= hi


def test_long_piece_slides_window():
    """> 100 s exercises the relative-time window (AMT's vocab only spans 100 s)."""
    lead = [Note(t, t + 0.5, 62 + 2 * (int(t) % 4), 70, "lead", float(t)) for t in np.arange(0, 150, 0.5)]
    new = I.infill(lead, 150.0, model=DummyAMT(), seed=0, instruments={0: "harmony"})
    assert I.LAST_STATS.slides >= 1
    assert max(n.start for n in new) > 100
    assert all(0 <= n.start < 150 for n in new)


def test_no_lead_still_works():
    new = I.infill([], 5.0, model=DummyAMT(), seed=0, key=None)
    assert all(n.voice in ("harmony", "bass") for n in new)


# ------------------------------------------------------------------ real checkpoint

@pytest.mark.slow
@pytest.mark.skipif(not small_available(), reason="music-small-800k not cached")
def test_real_model_small():
    notes = demo_notes()
    before = [asdict(n) for n in notes]
    new = I.infill(notes, 12.0, model=SMALL, seed=0)
    again = I.infill(notes, 12.0, model=SMALL, seed=0)
    assert [asdict(n) for n in notes] == before
    assert new and all(0 <= n.start < 12.0 and n.end <= 12.0 for n in new)
    assert [(n.start, n.pitch, n.voice) for n in new] == [(n.start, n.pitch, n.voice) for n in again]
    rep = I.clash_report(lead_of(notes), new)
    assert rep["dissonant_pairs"] == 0 and rep["above_lead"] == 0


@pytest.mark.slow
@pytest.mark.skipif(not small_available(), reason="music-small-800k not cached")
def test_kv_cache_matches_full_forward():
    from jukebox.score.amt_model import load
    m = load(SMALL)
    ids = torch.randint(0, VOCAB_SIZE, (1, 120), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        full, _ = m(ids)
        a, p = m(ids[:, :70])
        b, p = m(ids[:, 70:119], p)
        c, _ = m(ids[:, 119:], p)
    assert torch.allclose(full, torch.cat([a, b, c], 1), atol=1e-3)


# ------------------------------------------------------------------ demo artefact + renderer

@pytest.mark.skipif(not (ROOT / "demo/3912_harmony/timeline.json").exists(), reason="demo not generated")
def test_demo_timeline_keeps_lead():
    src = [d for d in json.loads(DEMO.read_text()) if d["voice"] == "lead"]
    out = json.loads((ROOT / "demo/3912_harmony/timeline.json").read_text())
    assert [d for d in out if d["voice"] == "lead"] == src
    assert {d["voice"] for d in out} >= {"lead", "harmony", "bass"}
    assert all(d["value"] is None for d in out if d["voice"] in ("harmony", "bass"))


def test_fluid_render_or_fallback():
    from jukebox.render import fluid
    notes = [Note(0.0, 0.5, 60, 90, "lead", 1.0), Note(0.5, 1.0, 64, 90, "bass", float("nan"))]
    audio = fluid.render_notes(notes, 1.0, {"lead": 0, "bass": 32}, tail=0.5)
    assert audio.dtype == np.float32 and audio.shape == (int(1.5 * 44100), 2)
    assert np.abs(audio).max() > 1e-3
