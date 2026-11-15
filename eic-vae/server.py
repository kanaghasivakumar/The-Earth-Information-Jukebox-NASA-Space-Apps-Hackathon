#!/usr/bin/env python3
"""Small local API so the Earth Jukebox web app can turn an uploaded visualization into music.

Run from the repo root:
    .venv/bin/python -m uvicorn main:app --app-dir eic-vae/eic-sonification/backend --port 8000

POST /api/jobs            multipart `file` (+ optional `styles` comma separated, `tempo`, `key`, `scale`) -> {job_id}
GET  /api/jobs/{job_id}   status, stage and, when done, the playable source
GET  /api/media/{job_id}/ generated audio, video preview and JSON

Jobs run one at a time in a background thread and live under eic-vae/jobs/.
Nothing trains or modifies the VAE; it reuses checkpoints/best_vae.pt.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from export_web import export, probe_duration
from music.composer import KEYS
from music.styles import available_styles

log = logging.getLogger('earth-jukebox')

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / 'jobs'
JOBS.mkdir(exist_ok=True)
ALLOWED = {'.mp4', '.mov', '.m4v', '.webm', '.avi', '.gif'}
MAX_BYTES = 300 * 1024 * 1024
JOB_TIMEOUT_S = 20 * 60      # kill a job that runs longer than this
MAX_ACTIVE_JOBS = 3          # queued + processing
JOB_TTL_S = 24 * 3600        # finished jobs are deleted after a day
MAX_KEPT_JOBS = 20
TIMEOUT_MESSAGE = 'The generator took too long and was stopped.'
JOB_ID_RE = re.compile(r'[0-9a-f]{12}')
ALLOWED_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:3001']

app = FastAPI(title='Earth Jukebox API')


@app.middleware('http')
async def guard_uploads(request, call_next):
    if request.method == 'POST' and request.url.path == '/api/jobs':
        # A multipart POST is a CORS "simple request", so refuse other web pages before reading anything.
        origin = request.headers.get('origin')
        if origin is not None and origin not in ALLOWED_ORIGINS:
            return JSONResponse({'detail': 'This origin is not allowed.'}, status_code=403)
        # Reject obviously oversize bodies before they are buffered. Chunked uploads without a
        # Content-Length are still limited by the byte counter in create_job.
        declared = request.headers.get('content-length')
        if declared and declared.isdigit() and int(declared) > MAX_BYTES + 1024 * 1024:
            return JSONResponse({'detail': 'That file is over 300 MB. Trim or compress it and try again.'}, status_code=413)
    return await call_next(request)


# Added after the guard so CORS is outermost and its headers also reach the guard's error responses.
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=['*'], allow_headers=['*'])
app.mount('/api/media', StaticFiles(directory=JOBS), name='media')

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_one_at_a_time = threading.Semaphore(1)
_pending: set[str] = set()  # job folders still being uploaded; the sweeper must not touch them


def _set(job_id: str, **fields):
    with _lock:
        _jobs[job_id].update(fields, updated=time.time())


def _stream(cmd, cwd, on_line, timeout=JOB_TIMEOUT_S):
    """Run `cmd`, feeding each output line to on_line. Kills it after `timeout` seconds.
    Returns (returncode, last_12_lines). Always leaves no child process behind."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            stdin=subprocess.DEVNULL)
    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    tail: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            tail = (tail + [line.rstrip()])[-12:]
            on_line(line)
        return proc.wait(), tail
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def _trim_finished(folder: Path):
    """Free disk space once a job is over: the upload, frame cache and full-quality wavs are not needed."""
    for f in folder.glob('upload*'):
        f.unlink(missing_ok=True)
    shutil.rmtree(folder / 'cache', ignore_errors=True)
    for wav in (folder / 'results').glob('*/earth_jukebox.wav'):
        wav.unlink(missing_ok=True)


def _friendly(exc: Exception) -> str:
    log.exception('job failed')  # full detail stays in the server log
    if isinstance(exc, subprocess.TimeoutExpired):
        return 'Preparing the audio and video took too long.'
    if isinstance(exc, subprocess.CalledProcessError):
        return 'Could not prepare the audio or video from that file.'
    if isinstance(exc, ValueError):
        return str(exc)[:200]  # our own ValueErrors are user-facing sentences
    if isinstance(exc, RuntimeError) and str(exc) == TIMEOUT_MESSAGE:
        return str(exc)
    return 'The generator could not process that file. Try a shorter or different clip.'


def _run(job_id: str, upload: Path, styles: list[str], extra: list[str], tempo=None, scale=None):
    folder = JOBS / job_id
    results, web = folder / 'results', folder / 'web'
    try:
        with _one_at_a_time:
            _set(job_id, status='processing', stage='Reading frames and running the model', progress=0.1)
            cmd = [sys.executable, str(ROOT / 'generate_music.py'), '--data', str(upload), '--styles', *styles,
                   '--memory', '--output-dir', str(results), '--cache-dir', str(folder / 'cache'), *extra]
            written = 0

            def on_line(line: str):
                nonlocal written
                if line.startswith('Wrote '):
                    written += 1
                    _set(job_id, stage=f'Composing and rendering styles ({written} of {len(styles)})',
                         progress=0.15 + 0.6 * written / len(styles))

            code, tail = _stream(cmd, ROOT, on_line)
            if code < 0:
                raise RuntimeError(TIMEOUT_MESSAGE)
            if code != 0:
                detail = next((l for l in reversed(tail) if l and not l.startswith(' ')), 'The generator stopped.')
                log.error('generator output: %s', detail)
                raise RuntimeError('generator failed')
            _set(job_id, stage='Preparing audio and video for the browser', progress=0.8)
            data = export(results, upload, web, prefix=f'/api/media/{job_id}/web', styles=styles,
                          on_stage=lambda m: _set(job_id, stage=m), tempo=tempo, scale=scale)
            source = {
                'id': job_id,
                'video': f'/api/media/{job_id}/web/video/hurricane.mp4',
                'poster': f'/api/media/{job_id}/web/video/poster.jpg',
                'signals': data['signals'],
                'tracks': data['tracks'],
            }
            _trim_finished(folder)
            _set(job_id, status='completed', stage='Ready', progress=1.0, source=source)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as plain text
        _trim_finished(folder)
        _set(job_id, status='failed', stage='Failed', error=_friendly(exc))


def _sweep():
    """Delete old finished job folders and forget jobs whose folder is gone."""
    now = time.time()
    with _lock:
        active = {i for i, j in _jobs.items() if j['status'] in ('queued', 'processing')}
        finished = []
        for folder in JOBS.iterdir():
            if not folder.is_dir() or not JOB_ID_RE.fullmatch(folder.name) or folder.name in active or folder.name in _pending:
                continue
            newest = max([folder.stat().st_mtime, *(p.stat().st_mtime for p in folder.rglob('*'))])
            if now - newest > JOB_TTL_S:
                shutil.rmtree(folder, ignore_errors=True)
                _jobs.pop(folder.name, None)
            else:
                finished.append((newest, folder))
        finished.sort(key=lambda t: t[0])
        for _, folder in finished[:max(0, len(finished) - MAX_KEPT_JOBS)]:
            shutil.rmtree(folder, ignore_errors=True)
            _jobs.pop(folder.name, None)
        for i in [i for i, j in _jobs.items() if j['status'] in ('completed', 'failed') and not (JOBS / i).exists()]:
            del _jobs[i]


_sweep()


@app.get('/api/health')
def health():
    return {'status': 'ok', 'styles': available_styles()}


@app.post('/api/jobs', status_code=202)
def create_job(file: UploadFile = File(...), styles: str = Form(''), tempo: int | None = Form(None),
               key: str | None = Form(None), scale: str | None = Form(None)):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(422, f'Unsupported file type {suffix or "(none)"}. Use MP4, MOV, WebM, AVI or GIF.')
    chosen = [s for s in re.split(r'[,\s]+', styles) if s] or available_styles()
    bad = [s for s in chosen if s not in available_styles()]
    if bad:
        raise HTTPException(422, f'Unknown style: {", ".join(bad)}')

    extra: list[str] = []
    if tempo is not None:
        if not 30 <= tempo <= 240:
            raise HTTPException(422, 'Tempo must be between 30 and 240 BPM.')
        extra += ['--tempo', str(tempo)]
    if key:
        norm = key[0].upper() + key[1:]
        if norm not in KEYS:
            raise HTTPException(422, 'Key must be one of ' + ', '.join(KEYS) + '.')
        extra += ['--key', norm]
    if scale:
        if scale not in ('major', 'minor', 'dorian', 'pentatonic'):
            raise HTTPException(422, 'Scale must be major, minor, dorian or pentatonic.')
        extra += ['--scale', scale]

    _sweep()
    job_id = uuid.uuid4().hex[:12]
    folder = JOBS / job_id
    with _lock:
        _pending.add(job_id)
    try:
        folder.mkdir()
        upload = folder / f'upload{suffix}'
        size = 0
        with upload.open('wb') as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, 'That file is over 300 MB. Trim or compress it and try again.')
                out.write(chunk)

        duration = probe_duration(upload)
        if duration is None:
            raise HTTPException(422, 'That file could not be read as a video.')
        if duration > 600:
            raise HTTPException(422, 'Videos longer than 10 minutes are not supported yet.')

        with _lock:
            if sum(1 for j in _jobs.values() if j['status'] in ('queued', 'processing')) >= MAX_ACTIVE_JOBS:
                raise HTTPException(429, 'The generator is busy. Try again in a few minutes.')
            _jobs[job_id] = {'job_id': job_id, 'status': 'queued', 'stage': 'Waiting for a free slot', 'progress': 0.02,
                             'filename': file.filename, 'duration': duration, 'created': time.time(), 'updated': time.time()}
    except BaseException:
        # Every failure path (validation, disconnect, unexpected error) removes the half-made job folder.
        shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        with _lock:
            _pending.discard(job_id)
    threading.Thread(target=_run, args=(job_id, upload, chosen, extra, tempo, scale or None), daemon=True).start()
    return {'job_id': job_id}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    with _lock:
        job = dict(_jobs[job_id]) if job_id in _jobs else None
    if job is None:
        done = JOBS / job_id / 'web' / 'tracks.json'
        if done.is_file():  # server restarted; rebuild from disk
            tracks = json.loads(done.read_text())
            signals = json.loads((JOBS / job_id / 'web' / 'signals.json').read_text())
            return {'job_id': job_id, 'status': 'completed', 'stage': 'Ready', 'progress': 1.0, 'source': {
                'id': job_id, 'video': f'/api/media/{job_id}/web/video/hurricane.mp4',
                'poster': f'/api/media/{job_id}/web/video/poster.jpg', 'signals': signals, 'tracks': tracks}}
        raise HTTPException(404, 'Unknown job.')
    return job
