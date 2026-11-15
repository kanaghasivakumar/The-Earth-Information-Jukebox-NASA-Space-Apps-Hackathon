"""Build the local, static Earth Jukebox listening page from saved outputs."""
from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import numpy as np
from music.styles import get_style

STYLE_COPY = {
    'ambient': ('Open sky', 'Soft, spacious tones leave room to follow the changing visual pattern.'),
    'cinematic': ('Wide-angle score', 'A sweeping arrangement follows the changing visual pattern.'),
    'classical': ('Orchestral lens', 'A familiar ensemble turns the same signals into a measured score.'),
    'jazz': ('Improvised current', 'A nimble, syncopated reading gives the motion a playful pulse.'),
    'electronic': ('Signal pulse', 'Bright synthesized textures trace shifts in the learned pattern.'),
}


def _duration(path: Path) -> float:
    try:
        with wave.open(str(path), 'rb') as audio:
            return audio.getnframes() / audio.getframerate()
    except (OSError, wave.Error, ZeroDivisionError):
        return 0.0


def _normalize(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not values.size:
        return []
    lo, hi = np.percentile(values, [5, 95])
    return np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1).tolist()


def _cached_flow(cache_dir: Path, timestamps: np.ndarray) -> list[float]:
    """Recover optical-flow magnitude only from an exact timestamp cache match."""
    for cache in cache_dir.glob('*.npz') if cache_dir.is_dir() else ():
        try:
            with np.load(cache) as saved:
                if 'timestamps' not in saved or 'flow' not in saved:
                    continue
                cached_times = np.asarray(saved['timestamps'], dtype=float).reshape(-1)
                flow_vectors = saved['flow']
                if len(cached_times) != len(timestamps) or not np.allclose(cached_times, timestamps, rtol=0, atol=1e-6):
                    continue
                magnitude = np.sqrt(np.sum(np.asarray(flow_vectors, dtype=float) ** 2, axis=1)).mean(axis=(1, 2))
                return _normalize(magnitude)
        except (OSError, ValueError, KeyError):
            continue
    return []


def build_showcase(output_dir: Path | str = 'results', project_root: Path | str | None = None,
                   source_video: Path | str | None = None) -> Path:
    """Build ``earth_jukebox_showcase.html`` beside WAV style folders."""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    results = Path(output_dir)
    if not results.is_absolute():
        results = root / results
    results = results.resolve()
    timestamps_path, latent_path = results / 'timestamps.npy', results / 'latent_means.npy'
    timestamps = np.load(timestamps_path) if timestamps_path.exists() else np.array([])
    latent = np.load(latent_path) if latent_path.exists() else np.empty((0, 0))
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if len(latent) != len(timestamps):
        timestamps = np.arange(len(latent), dtype=float)
    selected_dims = np.load(results / 'selected_dims.npy').astype(int).tolist() if (results / 'selected_dims.npy').exists() else []
    controls = latent[:, selected_dims] if len(latent) and selected_dims else latent[:, :2]
    if len(controls) > 1:
        delta = np.linalg.norm(np.diff(controls, axis=0), axis=1)
        movement = np.r_[delta[0], delta]
        lo, hi = np.percentile(movement, [5, 95])
        movement = np.clip((movement - lo) / max(hi - lo, 1e-9), 0, 1)
    else:
        movement = np.zeros(len(timestamps))
    patterns = []
    if len(latent):
        dims = selected_dims[:2] or list(range(min(2, latent.shape[1])))
        for dim in dims:
            series = latent[:, dim]
            lo, hi = np.percentile(series, [5, 95])
            patterns.append(np.clip((series - lo) / max(hi - lo, 1e-9), 0, 1).tolist())
    flow = []
    signals_path = results / 'visual_signals.npz'
    if signals_path.exists():
        with np.load(signals_path) as saved:
            if 'flow_magnitude' in saved and len(saved['flow_magnitude']) == len(timestamps):
                flow = np.asarray(saved['flow_magnitude'], dtype=float).tolist()
    if not flow:
        flow = _cached_flow(root / 'data/cache', timestamps)
    manifest_path = results / 'showcase_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else None
    if manifest is None:
        tracks = [{'id': 'default', 'folder': '.', 'style': 'cinematic'}] if (results / 'earth_jukebox.wav').is_file() else []
        for name in ('ambient', 'cinematic', 'classical', 'jazz', 'electronic'):
            if (results / name / 'earth_jukebox.wav').is_file():
                tracks.append({'id': name, 'folder': name, 'style': name})
    else:
        tracks = manifest.get('tracks', [])
    styles = []
    for track in tracks:
        name, folder_name, track_id = track.get('style', 'cinematic'), track.get('folder', '.'), track.get('id', 'default')
        folder = results / folder_name
        audio = folder / 'earth_jukebox.wav'
        if not audio.is_file():
            continue
        title, description = STYLE_COPY.get(name, ('Original score', 'The original single-score composition, shaped by saved visual signals.'))
        labels_path = folder / 'cluster_labels.npy'
        labels_enabled = track.get('memory_enabled', labels_path.exists())
        labels = np.load(labels_path).astype(int).tolist() if labels_enabled and labels_path.exists() else []
        memory = bool(labels)
        preset = get_style(name)
        instrument_names = track.get('instruments') or [item.timbre.title() for item in preset.instruments]
        caption_profile = manifest.get('caption_profile') if manifest else None
        if not track.get('instruments') and caption_profile and caption_profile.get('cues', {}).get('sst'):
            instrument_names = ['Warm Pad (caption interpretation)' if item.role == 'pad' else item.timbre.title() for item in preset.instruments]
        instruments = ' · '.join(instrument_names)
        styles.append({'id': track_id, 'name': 'Original' if track_id == 'default' else name.title(),
                       'title': title, 'description': description,
                       'audio': f'{folder_name}/earth_jukebox.wav' if folder_name != '.' else 'earth_jukebox.wav',
                       'duration': _duration(audio),
                       'memory': memory, 'labels': labels, 'instruments': instruments})
    manifest_source = manifest.get('source_video') if manifest else None
    selected_source = source_video or manifest_source
    if selected_source:
        video = Path(selected_source).resolve()
    else:
        source_candidates = sorted(path for path in (root / 'data/raw').glob('*')
                                   if path.suffix.lower() in {'.webm', '.mp4', '.mov', '.m4v'})
        video = source_candidates[0] if len(source_candidates) == 1 else Path('/__no_source_video__')
    video_url = Path(os.path.relpath(video, results)).as_posix() if video.is_file() else None
    payload = {'styles': styles, 'timestamps': timestamps.tolist(), 'movement': movement.tolist(),
               'patterns': patterns, 'flow': flow, 'video': video_url,
               'captionProfile': manifest.get('caption_profile') if manifest else None,
               'musicPlan': manifest.get('music_plan') if manifest else None,
               'sourceStart': float(timestamps[0]) if len(timestamps) else 0,
               'sourceDuration': float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0}
    template = (root / 'showcase/template.html').read_text(encoding='utf-8')
    rendered = template.replace('/* SHOWCASE_DATA */', json.dumps(payload, separators=(',', ':')).replace('</', '<\\/'))
    destination = results / 'earth_jukebox_showcase.html'
    destination.write_text(rendered, encoding='utf-8')
    return destination
