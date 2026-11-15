import csv

import numpy as np

from music.diagnostics import composition_metrics, save_memory_plot, write_style_comparison


def test_style_metrics_and_csv(tmp_path):
    roll = [
        [(0.0, 0.5, 60, 80)],
        [(0.0, 2.0, 48, 60)],
        [],
        [(0.0, 0.8, 36, 70)],
        [(0.5, 0.6, 42, 50)],
    ]
    schedule = [
        {'is_return': False, 'reuse_count': 0, 'is_variation': False},
        {'is_return': True, 'reuse_count': 1, 'is_variation': True},
        {'is_return': False, 'reuse_count': 2, 'is_variation': False},
    ]
    metrics = composition_metrics(
        roll, 2.0, style='ambient', memory_schedule=schedule,
        cluster_labels=[0, 0, 1, 1],
    )
    assert metrics['total_note_count'] == 4
    assert metrics['pitch_range_semitones'] == 24
    assert metrics['motif_repetitions'] == 2
    assert metrics['cluster_motif_returns'] == 1
    assert metrics['motif_variations'] == 1
    assert metrics['cluster_transitions'] == 1
    path = write_style_comparison([metrics], tmp_path / 'style_comparison.csv')
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['style'] == 'ambient'
    assert rows[0]['total_note_count'] == '4'


def test_memory_plot_direct_and_pca_projection(tmp_path):
    times = np.arange(6, dtype=float)
    labels = np.array([0, 0, 1, 1, 0, 0])
    events = [
        {'start': 0, 'end': 2, 'cluster_id': 0, 'is_return': False},
        {'start': 2, 'end': 4, 'cluster_id': 1, 'is_return': False},
        {'start': 4, 'end': 5, 'cluster_id': 0, 'is_return': True},
    ]
    for dimensions, name in ((2, '2d.png'), (5, 'pca.png')):
        z = np.random.default_rng(4).normal(size=(len(times), dimensions))
        path = save_memory_plot(z, times, labels, tmp_path / name, memory_schedule=events)
        assert path.is_file()
        assert path.stat().st_size > 1000


def test_scientific_alignment_metrics_are_time_aligned():
    times = np.arange(4, dtype=float)
    visual = {'timestamps': times, 'movement': times, 'flow_magnitude': times[::-1]}
    musical = {
        'timestamps': times,
        'musical_energy': times,
        'percussion_intensity': times[::-1],
    }
    metrics = composition_metrics(
        [[], [], [], [], []], 3.0, style='cinematic',
        musical_features=musical, visual_features=visual,
    )
    assert metrics['music_energy_vs_latent_movement_correlation'] == 1.0
    assert metrics['percussion_vs_optical_flow_correlation'] == 1.0
