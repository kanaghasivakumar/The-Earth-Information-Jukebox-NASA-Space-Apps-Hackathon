import json
import os
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "JOBS", tmp_path)  # job folders go to tmp
    monkeypatch.setattr(server, "_jobs", {})  # isolate the in-memory table
    started = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            started.append(args)

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Thread", FakeThread)  # never run the generator
    monkeypatch.setattr(
        server.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="12.5\n", stderr=""))  # ffprobe: 12.5 s
    c = TestClient(server.app)
    c.started = started
    return c


def post(client, name="clip.mp4", data=b"x" * 100, headers=None, **form):
    return client.post("/api/jobs", files={"file": (name, data, "video/mp4")}, data=form, headers=headers)


def test_health_lists_styles(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "jazz" in r.json()["styles"]


def test_rejects_unsupported_extension(client):
    r = post(client, name="x.exe")
    assert r.status_code == 422
    assert "Unsupported file type" in r.json()["detail"]


def test_rejects_unknown_style(client):
    r = post(client, styles="nope")
    assert r.status_code == 422
    assert "Unknown style" in r.json()["detail"]


def test_rejects_tempo_out_of_range(client):
    for tempo in ("5", "999"):
        assert post(client, tempo=tempo).status_code == 422


def test_rejects_bad_scale(client):
    assert post(client, scale="lydian").status_code == 422


def test_accepts_every_key_the_composer_supports(client):
    for k in [*server.KEYS, "bb"]:
        assert post(client, key=k).status_code == 202, k
        server._jobs.clear()  # stay under the queue cap


def test_rejects_keys_the_composer_does_not_support(client):
    for k in ("Cb", "E#", "Fb", "B#", "H"):
        assert post(client, key=k).status_code == 422, k


def test_accepts_valid_job_and_queues_it(client, tmp_path):
    r = post(client, styles="ambient,jazz", tempo="90", key="d", scale="dorian")
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert len(job_id) == 12 and all(c in "0123456789abcdef" for c in job_id)
    assert len(client.started) == 1
    assert client.started[0][3] == ["--tempo", "90", "--key", "D", "--scale", "dorian"]
    assert server._jobs[job_id]["status"] == "queued"
    assert (tmp_path / job_id / "upload.mp4").is_file()


def test_oversize_upload_returns_413_and_removes_folder(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MAX_BYTES", 10)
    r = post(client, data=b"x" * 100)
    assert r.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_unreadable_video_returns_422_and_removes_folder(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "probe_duration", lambda path: None)
    r = post(client)
    assert r.status_code == 422
    assert "could not be read as a video" in r.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_rejects_videos_over_ten_minutes(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        server.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="601\n", stderr=""))
    r = post(client)
    assert r.status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_get_job_unknown_is_404(client):
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_get_job_restores_completed_job_from_disk(client, tmp_path):
    web = tmp_path / "abc123def456" / "web"
    web.mkdir(parents=True)
    (web / "tracks.json").write_text(json.dumps([{"id": "ambient"}]))
    (web / "signals.json").write_text(json.dumps({"frames": 1}))
    r = client.get("/api/jobs/abc123def456")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["source"]["tracks"] == [{"id": "ambient"}]
    assert body["source"]["video"] == "/api/media/abc123def456/web/video/hurricane.mp4"


def test_get_job_returns_live_job(client):
    job = {"job_id": "aaaaaaaaaaaa", "status": "processing", "stage": "x", "progress": 0.3}
    server._jobs["aaaaaaaaaaaa"] = job
    r = client.get("/api/jobs/aaaaaaaaaaaa")
    assert r.status_code == 200
    assert r.json() == job


def test_stream_kills_a_process_that_exceeds_the_deadline(tmp_path):
    t0 = time.monotonic()
    code, _ = server._stream([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
                             on_line=lambda l: None, timeout=1)
    assert code < 0
    assert time.monotonic() - t0 < 5


def test_stream_collects_output_lines(tmp_path):
    seen = []
    code, tail = server._stream([sys.executable, "-c", "print('a'); print('b'); print('c')"], cwd=tmp_path,
                                on_line=seen.append)
    assert code == 0
    assert len(seen) == 3
    assert tail == ["a", "b", "c"]


def test_queue_cap_returns_429(client, tmp_path):
    for i in range(server.MAX_ACTIVE_JOBS):
        server._jobs[f"{i:012x}"] = {"job_id": f"{i:012x}", "status": "queued"}
    r = post(client)
    assert r.status_code == 429
    assert list(tmp_path.iterdir()) == []


def test_sweep_removes_old_finished_jobs_but_keeps_active_ones(client, tmp_path):
    old, active = tmp_path / "aaaaaaaaaaaa", tmp_path / "bbbbbbbbbbbb"
    long_ago = time.time() - 3 * 24 * 3600
    for folder in (old, active):
        folder.mkdir()
        (folder / "f.txt").write_text("x")
        os.utime(folder / "f.txt", (long_ago, long_ago))
        os.utime(folder, (long_ago, long_ago))
    server._jobs["bbbbbbbbbbbb"] = {"job_id": "bbbbbbbbbbbb", "status": "processing"}
    server._sweep()
    assert not old.exists()
    assert active.exists()


def test_get_job_returns_a_copy(client):
    server._jobs["cccccccccccc"] = {"job_id": "cccccccccccc", "status": "processing"}
    assert server.get_job("cccccccccccc") is not server._jobs["cccccccccccc"]


def test_trim_finished_removes_upload_cache_and_wavs(tmp_path):
    folder = tmp_path / "dddddddddddd"
    (folder / "cache").mkdir(parents=True)
    (folder / "results" / "ambient").mkdir(parents=True)
    (folder / "web").mkdir()
    (folder / "upload.mp4").write_bytes(b"x")
    (folder / "cache" / "x.npz").write_bytes(b"x")
    (folder / "results" / "ambient" / "earth_jukebox.wav").write_bytes(b"x")
    (folder / "web" / "tracks.json").write_text("[]")
    server._trim_finished(folder)
    assert not (folder / "upload.mp4").exists()
    assert not (folder / "cache").exists()
    assert not (folder / "results" / "ambient" / "earth_jukebox.wav").exists()
    assert (folder / "web" / "tracks.json").is_file()


def test_oversize_content_length_is_rejected_before_reading(client, tmp_path):
    r = client.post("/api/jobs", content=b"x", headers={
        "content-length": str(server.MAX_BYTES + 5 * 1024 * 1024),
        "content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_cross_origin_post_is_forbidden(client):
    assert post(client, headers={"Origin": "https://evil.example"}).status_code == 403
    assert post(client, headers={"Origin": "http://localhost:3000"}).status_code == 202
    assert post(client).status_code == 202


def test_failure_messages_do_not_leak_paths():
    secret = server._friendly(subprocess.CalledProcessError(1, ["ffmpeg", "-i", "/Users/someone/secret/path.mp4"]))
    assert "/Users" not in secret
    assert "/Users" not in server._friendly(RuntimeError("/Users/someone/secret"))
    assert "/Users" not in server._friendly(subprocess.TimeoutExpired(["ffmpeg", "/Users/someone/x"], 5))


def test_every_failed_upload_path_removes_its_folder(client, tmp_path, monkeypatch):
    def boom(path):
        raise RuntimeError("boom")
    monkeypatch.setattr(server, "probe_duration", boom)
    with pytest.raises(RuntimeError):  # TestClient re-raises server errors; a real server answers 500
        post(client)
    assert list(tmp_path.iterdir()) == []
