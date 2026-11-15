import csv
import json

import numpy as np

import generate_music
from music.director import DirectorResult, MusicPlan


def test_yaml_style_memory_defaults_and_cli_override(tmp_path):
    config_path = tmp_path / 'jukebox.yml'
    config_path.write_text(
        'music:\n  styles: [ambient, jazz]\n  seed: 5\n'
        'memory:\n  enabled: true\n  clusters: 3\n'
        'output:\n  directory: results\n',
        encoding='utf-8',
    )
    config = generate_music._load_config(str(config_path))
    parser = generate_music._make_parser(config)
    args = parser.parse_args(['--config', str(config_path)])
    assert args.styles == ['ambient', 'jazz']
    assert args.memory and args.clusters == 3
    override_parser = generate_music._make_parser(config, ['--style', 'classical'])
    override = override_parser.parse_args(['--config', str(config_path), '--style', 'classical'])
    assert override.style == 'classical' and override.styles is None


def test_memory_plot_schedule_is_aligned_to_source_time():
    schedule = [{'start': 2.0, 'end': 4.0, 'cluster_id': 1}]
    timestamps = np.array([10.0, 110.0])
    fixed = generate_music._schedule_video_times(schedule, timestamps, 8.0, 'fixed')
    matched = generate_music._schedule_video_times(schedule, timestamps, 8.0, 'match_video')
    assert (fixed[0]['start'], fixed[0]['end']) == (35.0, 60.0)
    assert (matched[0]['start'], matched[0]['end']) == (12.0, 14.0)


def test_multi_style_generation_runs_one_shared_inference(tmp_path, monkeypatch):
    times = np.arange(0, 8, .25)
    z = np.zeros((len(times), 5), dtype=float)
    z[:, 0] = np.r_[np.zeros(len(times) // 2), np.ones(len(times) - len(times) // 2) * 4]
    z[:, 1] = np.sin(times)
    z[:, 2] = np.cos(times * .5)
    ext = {
        'latent_means': z,
        'latent_logvars': np.zeros_like(z),
        'timestamps': times,
        'flow_magnitude': np.r_[np.zeros(len(times) // 2), np.ones(len(times) - len(times) // 2)],
    }
    calls = []

    def fake_extract(*args, **kwargs):
        calls.append((args, kwargs))
        return ext

    monkeypatch.setattr(generate_music, 'extract_latents', fake_extract)
    output = tmp_path / 'multi'
    generate_music.main([
        '--data', 'not-read-by-stub', '--output-dir', str(output),
        '--styles', 'ambient', 'jazz', '--memory', '--clusters', '2',
        '--minimum-cluster-duration', '0', '--cluster-hysteresis', '0',
        '--batch-size', '1',
    ])
    assert len(calls) == 1
    for name in ('ambient', 'jazz'):
        folder = output / name
        assert (folder / 'earth_jukebox.mid').is_file()
        assert (folder / 'earth_jukebox.wav').is_file()
        assert (folder / 'earth_jukebox_memory.png').is_file()
    with (output / 'style_comparison.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert [row['style'] for row in rows] == ['ambient', 'jazz']
    assert 'music_energy_vs_latent_movement_correlation' in rows[0]
    assert 'percussion_vs_optical_flow_correlation' in rows[0]


def test_music_director_respects_manual_style_tempo_scale_and_instrument(tmp_path, monkeypatch):
    times = np.arange(0, 12, .25)
    latent = np.column_stack((np.sin(times), np.cos(times * .7)))
    ext = {
        'latent_means': latent,
        'latent_logvars': np.zeros_like(latent),
        'timestamps': times,
        'flow_magnitude': np.abs(np.sin(times * .4)),
    }
    plan = MusicPlan(
        style='cinematic', tempo_bpm=88, scale='minor', mood='dramatic',
        explanation='A broad cinematic arc follows the changing visual pattern.',
        mappings=[{'source': 'optical_flow', 'target': 'percussion_density', 'strength': .8}],
    )
    director = DirectorResult(plan=plan, source='model', cache_key='cached-key', image_input_used=False)
    monkeypatch.setattr(generate_music, 'extract_latents', lambda *args, **kwargs: ext)
    monkeypatch.setattr(generate_music, 'resolve_music_plan', lambda *args, **kwargs: director)
    captured = {}

    def fake_compose(features, output, **kwargs):
        captured.update(kwargs)
        captured['features'] = features
        return {
            'duration': 12., 'bars': 5, 'piano_roll': [[] for _ in range(5)],
            'musical_features': {}, 'memory_schedule': [],
        }

    monkeypatch.setattr(generate_music, 'compose_midi', fake_compose)
    monkeypatch.setattr(generate_music, 'save_plots', lambda *args, **kwargs: None)
    monkeypatch.setattr(generate_music, '_render_wav', lambda *args, **kwargs: None)
    monkeypatch.setattr(generate_music, 'composition_metrics', lambda *args, **kwargs: {'style': 'jazz'})
    monkeypatch.setattr(generate_music, 'write_style_comparison', lambda *args, **kwargs: None)
    monkeypatch.setattr(generate_music, 'build_showcase', lambda *args, **kwargs: tmp_path / 'showcase.html')

    output = tmp_path / 'results'
    generate_music.main([
        '--data', 'stored-animation.webm', '--output-dir', str(output), '--auto-style',
        '--description', 'A tropical storm visualization.', '--style', 'jazz',
        '--tempo', '124', '--scale', 'major', '--instrument', 'melody=tenor sax',
    ])

    assert captured['tempo'] == 124
    assert captured['scale'] == 'major'
    assert captured['style'].name == 'jazz'
    assert captured['style'].instruments[0].timbre == 'tenor sax'
    assert 'director_percussion_density' in captured['features']
    manifest = json.loads((output / 'showcase_manifest.json').read_text(encoding='utf-8'))
    assert manifest['music_plan']['selected_style'] == 'jazz'
    assert manifest['tracks'][0]['tempo_bpm'] == 124
    assert manifest['tracks'][0]['scale'] == 'major'
