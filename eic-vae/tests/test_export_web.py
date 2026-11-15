import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

import export_web
from export_web import POINTS, export, normalise, probe_duration, resample, runs


def test_normalise_is_clipped_to_unit_range():
    out = normalise(np.linspace(-5, 50, 200))
    assert len(out) == 200
    assert min(out) >= 0 and max(out) <= 1


def test_normalise_constant_signal_does_not_divide_by_zero():
    out = normalise(np.ones(50))
    assert len(out) == 50
    assert all(np.isfinite(out))


def test_resample_caps_length():
    assert len(resample(np.arange(1000.0))) == POINTS
    assert len(resample(np.arange(10.0))) == 10


def test_runs_are_contiguous_and_cover_the_total():
    labels = np.array([1, 1, 2, 2, 2, 0])
    timestamps = np.array([0., 1., 2., 3., 4., 5.])
    out = runs(labels, timestamps, 6.0)
    assert out == [[0.0, 2.0, 1], [2.0, 5.0, 2], [5.0, 6.0, 0]]
    for a, b in zip(out[:-1], out[1:]):
        assert a[1] == b[0]
    assert out[-1][1] == 6.0


@pytest.mark.skipif(shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_probe_duration_reads_a_generated_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10", "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True)
    assert abs(probe_duration(clip) - 2.0) < 0.2


def test_duration_probe_falls_back_when_container_reports_na(monkeypatch):
    outputs = iter(["N/A\n", "7.5\n"])
    monkeypatch.setattr(export_web.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=next(outputs), stderr=""))
    assert probe_duration(Path("x")) == 7.5


@pytest.mark.skipif(shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_export_poster_exists_for_a_half_second_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=0.5:size=64x64:rate=10", "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True)
    results = tmp_path / "results"
    (results / "ambient").mkdir(parents=True)
    with wave.open(str(results / "ambient" / "earth_jukebox.wav"), "wb") as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 2 * 8000)
    np.save(results / "timestamps.npy", np.array([0.0, 0.25]))
    np.save(results / "latent_means.npy", np.zeros((2, 4)))
    np.save(results / "ambient" / "cluster_labels.npy", np.array([0, 1]))
    export(results, clip, tmp_path / "web", styles=["ambient"])
    assert (tmp_path / "web" / "video" / "poster.jpg").is_file()
    assert (tmp_path / "web" / "tracks.json").is_file()
