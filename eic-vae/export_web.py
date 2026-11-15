#!/usr/bin/env python3
"""Export saved Earth Jukebox results into web-ready media and JSON.

Reads results/<style>/ and the shared latent arrays, writes to the Next.js
frontend's public/jukebox folder:

  audio/<style>.mp3        128 kbps transcode of earth_jukebox.wav
  video/hurricane.mp4      720p H.264 copy of the source animation (muted)
  video/poster.jpg         first-frame poster
  tracks.json              per-style metadata, loudness envelope, region labels
  signals.json             shared visual-change signal (latent movement)

Nothing here trains or re-runs the VAE. Durations come from ffprobe.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from music.styles import available_styles, get_style

ROOT = Path(__file__).resolve().parent
POINTS = 400  # resolution of every curve sent to the browser


def run(cmd, timeout=600):
    subprocess.run(cmd, check=True, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)


def _ffprobe_duration(path: Path, entries: list[str]) -> float | None:
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', *entries, '-of', 'default=nw=1:nk=1', str(path)],
            check=True, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL).stdout
        value = float(out.strip().splitlines()[0])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def probe_duration(path: Path) -> float | None:
    """Duration in seconds, or None when the file cannot be read as a video.

    Some containers (WebM, GIF) report no format duration, so fall back to the video
    stream and finally to OpenCV's frame count / fps.
    """
    for entries in (['-show_entries', 'format=duration'],
                    ['-select_streams', 'v:0', '-show_entries', 'stream=duration']):
        value = _ffprobe_duration(path, entries)
        if value is not None:
            return value
    cap = cv2.VideoCapture(str(path))
    try:
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if frames > 0 and fps > 0:
        return float(frames / fps)
    return None


def resample(values: np.ndarray, n: int = POINTS) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) <= n:
        return values
    edges = np.linspace(0, len(values), n + 1).astype(int)
    return np.array([values[a:max(b, a + 1)].mean() for a, b in zip(edges[:-1], edges[1:])])


def normalise(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, 2), np.percentile(values, 98)
    return np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1).round(4).tolist()


def loudness_envelope(wav_path: Path) -> np.ndarray:
    """RMS of the rendered audio per window, so the curve is what you hear."""
    with wave.open(str(wav_path), 'rb') as wav:
        channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        raw = wav.readframes(frames)
    dtype = {2: np.int16, 4: np.int32}[width]
    data = np.frombuffer(raw, dtype=dtype).reshape(-1, channels).astype(np.float32).mean(axis=1)
    edges = np.linspace(0, len(data), POINTS + 1).astype(int)
    return np.array([np.sqrt(np.mean(data[a:b] ** 2)) if b > a else 0 for a, b in zip(edges[:-1], edges[1:])])


def runs(labels: np.ndarray, timestamps: np.ndarray, total: float):
    """Contiguous latent-region runs as [start, end, id] in seconds of source video."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            end = timestamps[i] if i < len(timestamps) else total
            out.append([round(float(timestamps[start]), 2), round(float(end), 2), int(labels[start])])
            start = i
    return out


def export(results: Path, video: Path, out: Path, prefix: str = '/jukebox', styles=None,
           skip_media: bool = False, on_stage=lambda msg: None, tempo: int | None = None, scale: str | None = None) -> dict:
    """Write web-ready media and JSON for one set of generated styles.

    `prefix` is the URL path the folder `out` will be served from.
    Returns {'tracks': [...], 'signals': {...}}.
    """
    from concurrent.futures import ThreadPoolExecutor

    results, out = Path(results), Path(out)
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    (out / 'video').mkdir(parents=True, exist_ok=True)

    timestamps = np.load(results / 'timestamps.npy')
    latents = np.load(results / 'latent_means.npy')
    movement = np.linalg.norm(np.diff(latents, axis=0, prepend=latents[:1]), axis=1)
    source_duration = probe_duration(Path(video))
    if source_duration is None:
        raise ValueError('Could not read the video duration.')

    wanted = list(styles or available_styles())
    names = [n for n in wanted if (results / n / 'earth_jukebox.wav').is_file()]
    if not names:
        raise ValueError('No audio was produced.')
    if styles and len(names) != len(wanted):
        raise ValueError('Some styles could not be rendered.')
    if not skip_media:
        on_stage('Preparing the video preview')
        run(['ffmpeg', '-nostdin', '-y', '-i', str(video), '-vf', 'scale=560:560:force_original_aspect_ratio=decrease,pad=560:560:(ow-iw)/2:(oh-ih)/2:black,fps=20',
             '-an', '-c:v', 'libx264', '-crf', '34', '-preset', 'medium', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', str(out / 'video' / 'hurricane.mp4')])
        poster = out / 'video' / 'poster.jpg'
        poster_cmd = ['-i', str(video), '-frames:v', '1',
                      '-vf', 'scale=560:560:force_original_aspect_ratio=decrease,pad=560:560:(ow-iw)/2:(oh-ih)/2:black',
                      '-q:v', '4', str(poster)]
        run(['ffmpeg', '-nostdin', '-y', '-ss', f'{min(1.0, source_duration / 2):.2f}', *poster_cmd])
        if not poster.is_file():  # seeking past the last frame writes nothing; retry from the first frame
            run(['ffmpeg', '-nostdin', '-y', *poster_cmd])
        if not poster.is_file():
            raise RuntimeError('poster missing')
        on_stage('Encoding audio')
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda n: run(['ffmpeg', '-nostdin', '-y', '-i', str(results / n / 'earth_jukebox.wav'),
                                         '-codec:a', 'libmp3lame', '-b:a', '128k', str(out / 'audio' / f'{n}.mp3')]), names))

    tracks = []
    for name in names:
        folder = results / name
        style = get_style(name)
        mp3 = out / 'audio' / f'{name}.mp3'
        labels_file = folder / 'cluster_labels.npy'
        labels = np.load(labels_file) if labels_file.is_file() else None
        mid = folder / 'earth_jukebox.mid'
        if mid.is_file():
            (out / 'audio' / f'{name}.mid').write_bytes(mid.read_bytes())
        tracks.append({
            'id': name,
            'tempo': int(tempo if tempo is not None else style.tempo_bpm),
            'scale': scale or style.default_scale,
            'duration': round(probe_duration(mp3), 2),
            'audio': f'{prefix}/audio/{name}.mp3',
            'midi': f'{prefix}/audio/{name}.mid',
            'instruments': [{'role': i.role, 'sound': i.timbre} for i in style.instruments],
            'loudness': normalise(loudness_envelope(folder / 'earth_jukebox.wav')),
            'regions': runs(labels, timestamps, source_duration) if labels is not None else [],
        })

    signals = {
        'sourceDuration': round(source_duration, 2),
        'firstTimestamp': float(timestamps[0]),
        'lastTimestamp': float(timestamps[-1]),
        'frames': int(len(timestamps)),
        'movement': normalise(resample(movement)),
    }
    # signals first, tracks last: the server treats tracks.json as "job finished", and each file lands atomically.
    for name, payload in (('signals.json', signals), ('tracks.json', tracks)):
        target = out / name
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, separators=(',', ':')))
        os.replace(tmp, target)
    return {'tracks': tracks, 'signals': signals}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default=str(ROOT / 'results'))
    p.add_argument('--video', default=str(next((ROOT / 'data' / 'raw').glob('*.webm'))))
    p.add_argument('--out', default=str(ROOT / 'eic-sonification' / 'frontend' / 'public' / 'jukebox'))
    p.add_argument('--skip-media', action='store_true', help='only rewrite JSON')
    args = p.parse_args()
    data = export(Path(args.results), Path(args.video), Path(args.out), skip_media=args.skip_media)
    print(f"wrote {len(data['tracks'])} tracks, source {data['signals']['sourceDuration']:.1f}s -> {args.out}")


if __name__ == '__main__':
    main()
