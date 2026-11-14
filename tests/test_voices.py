"""Voice-layer contract tests.

    python -m pytest tests/test_voices.py -q

Runs against the real corpora when present (JUKEBOX_CORPORA or the default assets path) and
always against the procedural stand-in corpus, so it also passes on a machine without assets.
"""
from __future__ import annotations

import numpy as np
import pytest

from jukebox.voices import GROUPS, SR, render_events, render_voice
from jukebox.voices import grains

pytestmark = pytest.mark.filterwarnings("ignore:no corpus at")


@pytest.fixture(params=["real", "standin"])
def corpus_mode(request, monkeypatch, tmp_path):
    if request.param == "standin":
        monkeypatch.setenv("JUKEBOX_CORPORA", str(tmp_path))
    elif not any((grains.corpora_root() / g).glob("*.flac") for g in GROUPS):
        pytest.skip("real corpora not fetched (run scripts/fetch_corpora.py)")
    grains.load_corpus.cache_clear()
    yield request.param
    grains.load_corpus.cache_clear()


T = np.linspace(0, 4, 9)
RAMP = np.linspace(0.2, 1.0, 9)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


@pytest.mark.parametrize("group", GROUPS)
def test_shape_dtype(group, corpus_mode):
    x = render_voice(group, T, RAMP, 4.0)
    assert x.shape == (4 * SR, 2) and x.dtype == np.float32
    assert np.isfinite(x).all() and np.abs(x).max() <= 0.9
    assert rms(x) > 1e-3
    y = render_voice(group, T, RAMP, 2.5, sr=22050)
    assert y.shape == (int(round(2.5 * 22050)), 2) and y.dtype == np.float32


@pytest.mark.parametrize("group", GROUPS)
def test_silence_at_level_zero(group, corpus_mode):
    z = render_voice(group, T, np.zeros_like(T), 4.0)
    assert z.shape == (4 * SR, 2) and not np.any(z)
    z = render_voice(group, T, 0.0, 4.0, tune_to=[50, 57])
    assert not np.any(z)
    # a curve that drops to 0 is silent where it is 0
    lv = np.where(T < 2, 0.8, 0.0)
    x = render_voice(group, T, lv, 4.0)
    assert rms(x[: SR]) > 1e-3
    assert np.abs(x[int(2.6 * SR):]).max() < 1e-6


@pytest.mark.parametrize("group", GROUPS)
def test_deterministic(group, corpus_mode):
    kw = dict(brightness=np.linspace(0, 1, 9), pan=np.linspace(-1, 1, 9), tune_to=[50, 54, 57])
    a = render_voice(group, T, RAMP, 3.0, seed=3, **kw)
    b = render_voice(group, T, RAMP, 3.0, seed=3, **kw)
    c = render_voice(group, T, RAMP, 3.0, seed=4, **kw)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("group", GROUPS)
def test_level_drives_loudness_and_pan(group, corpus_mode):
    lo = render_voice(group, T, 0.15, 4.0, seed=1)
    hi = render_voice(group, T, 0.9, 4.0, seed=1)
    assert rms(hi) > 2 * rms(lo)
    left = render_voice(group, T, 0.8, 4.0, pan=-1.0, width=0.0, seed=1)
    assert rms(left[:, 0]) > 5 * rms(left[:, 1])


def test_tune_to_timeline_and_events(corpus_mode):
    prog = [(0, 2, [50, 54, 57]), (2, 4, [47, 54, 59])]
    x = render_voice("geophony", T, 0.7, 4.0, tune_to=prog, tune_mix=0.6)
    assert x.shape == (4 * SR, 2) and np.isfinite(x).all() and rms(x) > 1e-3
    e = render_events([0.5, 2.0], [1.0, 0.5], 4.0, pan=[-0.5, 0.5])
    assert e.shape == (4 * SR, 2) and rms(e[int(0.5 * SR): int(1.5 * SR)]) > 1e-3
    assert not np.any(render_events([], [], 1.0))


def test_no_clicks(corpus_mode):
    """Hann-windowed overlap-add: no isolated sample-to-sample jumps far above the local level.
    Checked on the smooth stand-in corpus, since real rain/thunder grains contain genuine transients."""
    if corpus_mode == "real":
        pytest.skip("windowing check uses the stand-in corpus")
    x = render_voice("anthrophony", T, RAMP, 4.0, seed=2).mean(1)
    d = np.abs(np.diff(x))
    n = int(0.05 * SR)
    blk = d[: len(d) // n * n].reshape(-1, n)
    ratio = blk.max(1) / (np.sqrt((blk ** 2).mean(1)) + 1e-6)
    assert ratio.max() < 12


def test_bad_group():
    with pytest.raises(ValueError):
        render_voice("technophony", T, RAMP, 1.0)
