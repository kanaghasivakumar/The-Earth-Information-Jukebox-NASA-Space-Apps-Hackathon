"""Tests for jukebox.render.arrange (the integrator: data notes + harmony + voices -> stems + mix).

    PYTHONPATH=. python -m pytest tests/test_arrange.py -q

Uses a deterministic stand-in AMT model and the procedural stand-in voice corpus, so it runs in
well under a minute without the HF cache or the corpora. FluidSynth is used when available,
otherwise the additive-synth fallback.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from anticipation.config import MAX_TIME
from anticipation.vocab import TIME_OFFSET, VOCAB_SIZE

from jukebox.render import arrange as A
from jukebox.render.synth import SR
from jukebox.voices import grains

pytestmark = pytest.mark.filterwarnings("ignore:no corpus at")
PM = {"vmin": 0.0, "vmax": 10.0, "low": 60, "span": 24, "root": 2, "scale": "major_pent", "polarity": 1}
DUR = 6.0


class DummyAMT:
    """Deterministic pseudo-random logits (same idea as tests/test_infill.py)."""
    name = "dummy"
    device = torch.device("cpu")

    def __call__(self, ids, past=None):
        n = (past or 0) + ids.shape[1]
        g = torch.Generator().manual_seed(int(ids[0, -1]) * 7919 + n)
        lg = torch.randn(VOCAB_SIZE, generator=g)
        lg[TIME_OFFSET:TIME_OFFSET + MAX_TIME] -= 0.02 * torch.arange(MAX_TIME)
        return lg.view(1, 1, -1), n


@pytest.fixture(autouse=True)
def standin_corpus(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("JUKEBOX_CORPORA", str(tmp_path_factory.mktemp("no_corpora")))
    grains.load_corpus.cache_clear()
    yield
    grains.load_corpus.cache_clear()


def make_track(d: Path) -> Path:
    from jukebox.score.mapping import PitchMap, melody_from_curve, timeline
    d.mkdir(parents=True, exist_ok=True)
    t = np.arange(0, DUR, 0.5)
    vals = 5 + 4 * np.sin(t / 1.3)
    pm = A.PitchMap(**PM)
    lead = melody_from_curve(t, vals, pm, "lead")
    band = melody_from_curve(t[::4], vals[::4] * 0.8, PitchMap(0, 10, low=40, span=12, root=2), "band x")
    (d / "timeline.json").write_text(json.dumps(timeline(lead + band)))
    tc = np.linspace(0, DUR, 25)
    (d / "controls.json").write_text(json.dumps({"t": tc.tolist(), "curves": {
        "geophony": np.linspace(0.2, 1, 25).tolist(), "anthrophony": [0.0] * 25,
        "biophony": np.linspace(1, 0.1, 25).tolist(), "brightness": [0.5] * 25}}))
    (d / "track.json").write_text(json.dumps({"id": "t", "title": "test", "duration_s": DUR, "units": "u",
                                              "legend": [], "stems": [], "fidelity": {"r2": 0.9}}))
    (d / "report.json").write_text(json.dumps({"lead_pitchmap": PM}))
    return d


def cfg(**kw) -> A.Arrangement:
    base = dict(id="t", lead_voices=("lead",), harmony={"top_p": 0.9}, tail=0.5,
                beds=[A.Bed("geophony"), A.Bed("anthrophony"), A.Bed("biophony")],
                fidelity=[A.FidSpec(("lead",), PM, "lead")], mix_fidelity=False)
    base.update(kw)
    return A.Arrangement(**base)


def run(d: Path, c=None, **kw):
    return A.arrange(d, c or cfg(), model=DummyAMT(), plot=False, verbose=False, **kw)


def data_rows(path: Path, c) -> list[dict]:
    return A.fingerprint([n for n in A.load_notes(path) if A.is_data(n, c)])


def test_lead_unchanged_and_harmony_added(tmp_path):
    d = make_track(tmp_path / "t")
    before = json.loads((d / "timeline.json").read_text())
    rep = run(d)
    after = json.loads((d / "timeline.json").read_text())
    data_after = [n for n in after if n["voice"] in ("lead", "band x")]
    key = lambda n: (n["start"], n["voice"], n["pitch"])
    assert sorted(data_after, key=key) == sorted(before, key=key)          # exact JSON identity
    model = [n for n in after if n["voice"] in ("harmony", "bass")]
    assert model and all(n["value"] is None for n in model)                 # model notes carry no data
    assert rep["data_notes_unchanged"] and rep["model_notes"] == len(model)
    lead = [n for n in after if n["voice"] == "lead"]
    for m in model:                                                         # harmony stays below the lead
        if m["voice"] == "harmony":
            assert m["pitch"] < min(n["pitch"] for n in lead)


def test_stems_aligned_and_track_contract(tmp_path):
    d = make_track(tmp_path / "t")
    rep = run(d)
    track = json.loads((d / "track.json").read_text())
    n = int(round((DUR + 0.5) * SR))
    groups = {s["group"] for s in track["stems"]}
    assert {"melody", "bands", "harmony", "geophony", "biophony"} <= groups
    assert "anthrophony" not in groups                                       # all-zero curve is skipped
    assert all(g in A.GROUPS for g in groups)
    total = np.zeros((n, 2))
    for s in track["stems"]:
        a, sr = sf.read(str(d / s["file"]), always_2d=True)
        assert sr == SR and a.shape == (n, 2), s
        total += a
    mix, _ = sf.read(str(d / "mix.wav"), always_2d=True)
    ctrl, _ = sf.read(str(d / "control_shuffled.wav"), always_2d=True)
    assert mix.shape == ctrl.shape == (n, 2)
    # stems sum to the pre-limiter mix (equal where the limiter did nothing)
    c = np.corrcoef(total.ravel(), mix.ravel())[0, 1]
    assert c > 0.99, c
    assert abs(rep["loudness"]["lufs_mix"] - (-16.0)) < 1.5 or rep["loudness"]["limiter_max_gr_db"] > 0
    f = track["fidelity"]
    assert f["r2"] is not None and f["r2"] > 0.8 and f["control_r2"] < f["r2"]
    assert track["voice_groups"]["lead"] == "melody" and track["voice_groups"]["harmony"] == "harmony"


def test_deterministic_seed(tmp_path):
    a = make_track(tmp_path / "a")
    b = make_track(tmp_path / "b")
    run(a)
    run(b)
    for f in ("melody.wav", "harmony.wav", "geophony.wav", "biophony.wav", "mix.wav"):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    assert (a / "timeline.json").read_text() == (b / "timeline.json").read_text()
    c = make_track(tmp_path / "c")
    run(c, cfg(seed=1))
    assert (a / "geophony.wav").read_bytes() != (c / "geophony.wav").read_bytes()


def test_rerun_uses_cache_and_keeps_source(tmp_path):
    d = make_track(tmp_path / "t")
    run(d)
    first = (d / "timeline.json").read_text()
    rep = run(d)
    assert rep["harmony"]["cached"] is True
    assert (d / "timeline.json").read_text() == first
    assert json.loads((d / "track_data.json").read_text())["stems"] == []    # builder's track.json kept


def test_tampering_with_data_notes_is_caught(tmp_path, monkeypatch):
    d = make_track(tmp_path / "t")
    import jukebox.score.infill as I

    def bad_infill(notes, duration, **kw):
        notes[0].pitch += 1          # a "harmony" step that edits a data note in place
        return []
    monkeypatch.setattr(I, "infill", bad_infill)
    with pytest.raises(AssertionError):
        A.arrange(d, cfg(), plot=False, verbose=False, model=DummyAMT())


def test_form_cues_never_touch_data(tmp_path):
    d = make_track(tmp_path / "t")
    form = tmp_path / "form.json"
    form.write_text(json.dumps({"sections": [
        {"index": 0, "start": 0.0, "end": 3.0, "label": "A", "is_return": False, "returns_to": None},
        {"index": 1, "start": 3.0, "end": DUR, "label": "B", "is_return": False, "returns_to": None}]}))
    before = data_rows(d / "timeline.json", cfg())
    run(d, cfg(form=str(form)), root=tmp_path)
    after = A.load_notes(d / "timeline.json")
    assert data_rows(d / "timeline.json", cfg()) == before
    assert any(n.voice == "form cymbal" for n in after)


def test_lufs_and_limiter():
    t = np.arange(int(3 * SR)) / SR
    x = 0.1 * np.sin(2 * np.pi * 997 * t)
    stereo = np.stack([x, x], 1)
    assert abs(A.lufs(stereo) - (-20.0)) < 0.3              # BS.1770: -20 dBFS 997 Hz in both channels
    loud = (stereo * 15).astype(np.float32)
    y, gr = A.limiter(loud)
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6 and abs(gr - 20 * np.log10(1.5 / 10 ** (-1 / 20))) < 0.2


def test_thin_notes_only_reduces_model_notes():
    from jukebox.score.mapping import Note
    new = [Note(i * 0.1, i * 0.1 + 0.2, 50 + i % 5, 70, "harmony", math.nan) for i in range(40)]
    kept = A.thin_notes(new, 1.5)
    assert 0 < len(kept) < len(new) and all(k in new for k in kept)
