"""Diagnostics for comparing style scores and inspecting musical memory."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_PALETTE = [
    (42, 105, 190), (222, 103, 55), (62, 150, 100), (158, 83, 174),
    (216, 164, 47), (40, 157, 163), (190, 73, 112), (105, 105, 105),
]


def _event_value(event, name, default=None):
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def _correlation(left, right):
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return ''
    left, right = left[valid], right[valid]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return ''
    return float(np.corrcoef(left, right)[0, 1])


def composition_metrics(
    piano_roll,
    duration: float,
    *,
    style: str,
    memory_schedule=None,
    cluster_labels=None,
    musical_features=None,
    visual_features=None,
    timing_mode='match_video',
    source_duration=None,
) -> dict:
    """Return comparable, deterministic summary values for one score."""
    notes = [note for track in piano_roll for note in track]
    pitched = [note for track in piano_roll[:4] for note in track]
    pitches = [int(note[2]) for note in pitched]
    note_durations = [max(0.0, float(note[1]) - float(note[0])) for note in notes]
    velocities = [int(note[3]) for note in notes]
    duration = max(float(duration), 1e-9)
    schedule = list(memory_schedule or [])
    repetitions = sum(bool(_event_value(event, 'is_return', False)) or
                      int(_event_value(event, 'reuse_count', 0) or 0) > 0 for event in schedule)
    variations = sum(bool(_event_value(event, 'is_variation', False)) for event in schedule)
    labels = np.asarray(cluster_labels if cluster_labels is not None else [], dtype=int)
    transitions = int(np.count_nonzero(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
    movement_correlation = ''
    flow_correlation = ''
    if musical_features is not None and visual_features is not None:
        music_times = np.asarray(musical_features.get('timestamps', []), dtype=float)
        video_times = np.asarray(visual_features.get('timestamps', []), dtype=float)
        if len(music_times) and len(video_times):
            if timing_mode == 'fixed' and source_duration is not None:
                video_query = video_times[0] + music_times / duration * float(source_duration)
            else:
                video_query = video_times[0] + music_times
            movement = np.interp(video_query, video_times, visual_features.get('movement', np.zeros(len(video_times))))
            flow = np.interp(video_query, video_times, visual_features.get('flow_magnitude', np.zeros(len(video_times))))
            movement_correlation = _correlation(musical_features.get('musical_energy', []), movement)
            flow_correlation = _correlation(musical_features.get('percussion_intensity', []), flow)
    return {
        'style': str(style),
        'total_note_count': len(notes),
        'average_note_duration_seconds': float(np.mean(note_durations)) if note_durations else 0.0,
        'pitch_min': min(pitches) if pitches else '',
        'pitch_max': max(pitches) if pitches else '',
        'pitch_range_semitones': max(pitches) - min(pitches) if pitches else 0,
        'rhythmic_density_notes_per_second': len(notes) / duration,
        'average_note_velocity': float(np.mean(velocities)) if velocities else 0.0,
        'motif_repetitions': int(repetitions),
        'motif_variations': int(variations),
        'cluster_transitions': transitions,
        'music_energy_vs_latent_movement_correlation': movement_correlation,
        'percussion_vs_optical_flow_correlation': flow_correlation,
    }


def write_style_comparison(rows, output_path) -> Path:
    """Write one row per style as a portable CSV comparison table."""
    rows = list(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'style', 'total_note_count', 'average_note_duration_seconds',
        'pitch_min', 'pitch_max', 'pitch_range_semitones',
        'rhythmic_density_notes_per_second', 'average_note_velocity',
        'motif_repetitions', 'motif_variations', 'cluster_transitions',
        'music_energy_vs_latent_movement_correlation', 'percussion_vs_optical_flow_correlation',
    ]
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_memory_plot(
    latent_trajectory,
    timestamps,
    cluster_labels,
    output_path,
    *,
    memory_schedule=None,
) -> Path:
    """Plot a latent path plus its time-aligned cluster and motif history.

    Two-dimensional latent vectors are shown directly. Higher-dimensional
    vectors are projected onto their first two principal components using SVD.
    """
    z = np.asarray(latent_trajectory, dtype=float)
    t = np.asarray(timestamps, dtype=float).reshape(-1)
    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    if z.ndim != 2 or len(z) != len(t) or len(z) != len(labels):
        raise ValueError('latent_trajectory, timestamps, and cluster_labels must have matching rows')
    if not len(t):
        raise ValueError('cannot plot an empty latent trajectory')
    if not np.isfinite(z).all() or not np.isfinite(t).all():
        raise ValueError('latent_trajectory and timestamps must be finite')

    if z.shape[1] == 2:
        xy = z.copy()
        projection_name = 'latent dimensions 1 and 2'
    elif z.shape[1] > 2:
        centered = z - np.mean(z, axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        xy = centered @ right[:2].T
        projection_name = 'first two principal components'
    else:
        xy = np.column_stack((z[:, 0], np.zeros(len(z))))
        projection_name = 'latent dimension 1 (dimension 2 is zero)'

    image = Image.new('RGB', (1400, 900), 'white')
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 70, 100, 1010, 660
    draw.text((70, 25), 'Musical memory: latent regions and motif returns', fill=(20, 20, 20))
    draw.text((70, 62), f'Projection: {projection_name}', fill=(70, 70, 70))
    draw.rectangle((left, top, right, bottom), outline=(100, 100, 100), width=1)
    lo = np.min(xy, axis=0)
    hi = np.max(xy, axis=0)
    span = np.maximum(hi - lo, 1e-8)
    pad = span * .08
    lo -= pad
    hi += pad

    def pixel(point):
        x = left + (point[0] - lo[0]) / max(hi[0] - lo[0], 1e-9) * (right - left)
        y = bottom - (point[1] - lo[1]) / max(hi[1] - lo[1], 1e-9) * (bottom - top)
        return int(x), int(y)

    # Two-sigma covariance ellipses provide a visual guide to each recurring
    # region without presenting cluster labels as physical categories.
    for cluster in sorted(set(map(int, labels))):
        points = xy[labels == cluster]
        center = points.mean(axis=0)
        covariance = np.cov(points, rowvar=False) if len(points) > 1 else np.eye(2) * 1e-6
        eigenvalues, eigenvectors = np.linalg.eigh(np.atleast_2d(covariance))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 1e-8)
        eigenvectors = eigenvectors[:, order]
        theta = np.linspace(0, 2 * np.pi, 81)
        circle = np.vstack((2 * np.sqrt(eigenvalues[0]) * np.cos(theta),
                            2 * np.sqrt(eigenvalues[1]) * np.sin(theta)))
        ellipse = center[:, None] + eigenvectors @ circle
        polygon = [pixel(ellipse[:, i]) for i in range(ellipse.shape[1])]
        draw.line(polygon, fill=_PALETTE[cluster % len(_PALETTE)], width=2)

    # Show the path chronologically; colors identify the stabilized cluster.
    for i in range(1, len(xy)):
        draw.line((pixel(xy[i - 1]), pixel(xy[i])), fill=(185, 185, 185), width=1)
    for i, point in enumerate(xy):
        x, y = pixel(point)
        color = _PALETTE[int(labels[i]) % len(_PALETTE)]
        radius = 4 if i in (0, len(xy) - 1) else 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(35, 35, 35))
    if len(xy) > 1:
        x, y = pixel(xy[0]); draw.text((x + 5, y - 12), 'start', fill=(20, 20, 20))
        x, y = pixel(xy[-1]); draw.text((x + 5, y - 12), 'end', fill=(20, 20, 20))
    draw.text((left + 420, bottom + 12), 'Component 1', fill=(70, 70, 70))
    draw.text((left + 5, top + 5), 'Component 2', fill=(70, 70, 70))

    unique = sorted(set(map(int, labels)))
    legend_y = top + 10
    for cluster in unique:
        color = _PALETTE[cluster % len(_PALETTE)]
        draw.rectangle((1040, legend_y, 1055, legend_y + 14), fill=color)
        count = int(np.count_nonzero(labels == cluster))
        draw.text((1065, legend_y), f'cluster {cluster}  ({count} frames)', fill=(30, 30, 30))
        legend_y += 25

    time_lo, time_hi = float(t[0]), float(t[-1])
    time_span = max(time_hi - time_lo, 1e-9)
    timeline_left, timeline_right = 70, 1320
    transition_y = 725
    motif_y0, motif_y1 = 790, 824
    draw.text((70, 690), 'Cluster transitions', fill=(20, 20, 20))
    draw.line((timeline_left, transition_y, timeline_right, transition_y), fill=(150, 150, 150), width=2)
    transition_indices = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for i in transition_indices:
        x = timeline_left + (t[i] - time_lo) / time_span * (timeline_right - timeline_left)
        draw.line((int(x), transition_y - 10, int(x), transition_y + 10), fill=(190, 45, 45), width=2)
    draw.text((70, 752), f'{len(transition_indices)} stabilized cluster transitions', fill=(70, 70, 70))

    draw.text((70, 770), 'Active motif (color = cluster; R = a returning motif)', fill=(20, 20, 20))
    draw.rectangle((timeline_left, motif_y0, timeline_right, motif_y1), outline=(120, 120, 120))
    schedule = list(memory_schedule or [])
    if not schedule:
        # A cluster-state band is still useful before motif events are supplied.
        schedule = []
        changes = np.r_[0, transition_indices, len(t)]
        for a, b in zip(changes[:-1], changes[1:]):
            schedule.append({'start': float(t[a]), 'end': float(t[b - 1]), 'cluster_id': int(labels[a])})
    for event in schedule:
        start = float(_event_value(event, 'start', time_lo))
        end = float(_event_value(event, 'end', time_hi))
        cluster = int(_event_value(event, 'cluster_id', 0))
        x0 = timeline_left + (start - time_lo) / time_span * (timeline_right - timeline_left)
        x1 = timeline_left + (end - time_lo) / time_span * (timeline_right - timeline_left)
        x0 = int(np.clip(x0, timeline_left, timeline_right - 1))
        x1 = int(np.clip(max(x1, x0 + 2), timeline_left + 1, timeline_right))
        color = _PALETTE[cluster % len(_PALETTE)]
        draw.rectangle((x0, motif_y0 + 1, x1, motif_y1 - 1), fill=color)
        if bool(_event_value(event, 'is_return', False)) or int(_event_value(event, 'reuse_count', 0) or 0) > 0:
            draw.text((x0 + 1, motif_y0 + 7), 'R', fill='white')

    draw.text((70, 850), f'Video time: {time_lo:.2f}–{time_hi:.2f} seconds', fill=(70, 70, 70))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
