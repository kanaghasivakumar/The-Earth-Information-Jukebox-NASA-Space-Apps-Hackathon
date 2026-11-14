#!/usr/bin/env python3
"""Generate Earth Jukebox scores from one saved VAE trajectory."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from music.composer import compose_midi
from music.diagnostics import composition_metrics, save_memory_plot, write_style_comparison
from music.features import extract_features
from music.inference import extract_latents
from music.memory import LatentMemory, Motif, MotifNote
from music.plots import save_plots
from music.render import render_basic_wav
from music.styles import available_styles, get_style


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    raw = config_path.read_text(encoding='utf-8')
    if config_path.suffix.lower() == '.json':
        data = json.loads(raw)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit('YAML configuration requires PyYAML; install the project requirements.') from exc
        data = yaml.safe_load(raw)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit('The configuration file must contain a YAML/JSON mapping.')
    return data


def _mapping(config: dict, section: str) -> dict:
    value = config.get(section, {})
    if not isinstance(value, dict):
        raise SystemExit(f'Configuration section {section!r} must be a mapping.')
    return value


def _make_motif_library(style, cluster_count: int, scale_length: int) -> dict[int, Motif]:
    """Create one recognizable, style-shaped scale motif per cluster."""
    params = style.motif
    degrees = tuple(int(value) % scale_length for value in params.scale_degrees)
    note_count = max(1, min(len(degrees), int(round(len(degrees) * params.activity_density))))
    selected = sorted(set(np.rint(np.linspace(0, len(degrees) - 1, note_count)).astype(int).tolist()))
    phrase_beats = params.phrase_bars * 4
    spacing = phrase_beats / max(1, len(selected))
    library = {}
    for cluster_id in range(cluster_count):
        # A stable cluster-dependent transposition makes each visual region
        # recognizable while preserving the style's original interval contour.
        shift = (cluster_id * max(1, params.nearby_variation_steps + 1)) % scale_length
        notes = []
        for position, degree_index in enumerate(selected):
            offset = position * spacing
            if position % 2:
                offset += style.rhythm.syncopation * .5
            offset = min(offset, phrase_beats - .125)
            duration = max(.25, spacing * params.legato)
            accent = .72 if position % 2 else 1.0
            velocity = int(round(style.dynamics.floor_velocity +
                                 (style.dynamics.peak_velocity - style.dynamics.floor_velocity) * accent))
            notes.append(MotifNote((degrees[degree_index] + shift) % scale_length,
                                   offset, duration, int(np.clip(velocity, 1, 127))))
        notes.sort(key=lambda note: note.beat_offset)
        library[cluster_id] = Motif(tuple(notes), name=f'{style.name}-cluster-{cluster_id}')
    return library


def _make_parser(config: dict, argv=()) -> argparse.ArgumentParser:
    music = _mapping(config, 'music')
    memory = _mapping(config, 'memory')
    output = _mapping(config, 'output')
    style_default = music.get('style')
    styles_default = music.get('styles')
    if any(arg == '--style' or arg.startswith('--style=') or arg == '--styles' for arg in argv):
        style_default = None
        styles_default = None
    # A config may declare either a single style or a set of styles.
    if styles_default:
        style_default = None
    parser = argparse.ArgumentParser(
        description='Turn one NASA visualization into one or more music styles using a saved VAE.'
    )
    parser.add_argument('--config', help='optional YAML or JSON configuration file')
    parser.add_argument('--checkpoint', default=music.get('checkpoint', 'checkpoints/best_vae.pt'))
    parser.add_argument('--data', '--video', dest='data', default=music.get('data', music.get('video', 'data/raw')),
                        help='single animation file or folder containing one animation source')
    parser.add_argument('--output-dir', default=output.get('directory', 'results'))
    style_group = parser.add_mutually_exclusive_group()
    style_group.add_argument('--style', choices=available_styles(), default=style_default)
    style_group.add_argument('--styles', nargs='+', choices=available_styles(), default=styles_default,
                             help='generate several styles after one shared VAE inference pass')
    parser.add_argument('--tempo', type=int, default=music.get('tempo'))
    parser.add_argument('--key', default=music.get('key', 'C'))
    parser.add_argument('--scale', choices=('major', 'minor', 'dorian', 'pentatonic'), default=music.get('scale'))
    parser.add_argument('--timing-mode', '--duration-mode', choices=('match_video', 'fixed'),
                        default=music.get('timing_mode', music.get('duration_mode', 'match_video')))
    parser.add_argument('--target-duration', type=float, default=music.get('target_duration'))
    parser.add_argument('--smoothing', type=int, default=music.get('smoothing', 1))
    parser.add_argument('--movement-weight', type=float, default=music.get('movement_weight', 1.0))
    parser.add_argument('--flow-weight', type=float, default=music.get('flow_weight', 1.0))
    parser.add_argument('--seed', type=int, default=music.get('seed', 0))
    parser.add_argument('--soundfont', default=output.get('soundfont'))
    parser.add_argument('--device', default=music.get('device', 'cpu'))
    parser.add_argument('--batch-size', type=int, default=music.get('batch_size', 8))
    parser.add_argument('--memory', action=argparse.BooleanOptionalAction,
                        default=memory.get('enabled', False), help='enable latent-region motif memory')
    parser.add_argument('--clusters', type=int, default=memory.get('clusters', 4))
    parser.add_argument('--motif-bars', type=int, default=memory.get('motif_bars'))
    parser.add_argument('--variation-strength', type=float, default=memory.get('variation_strength', .25))
    parser.add_argument('--minimum-cluster-duration', type=float,
                        default=memory.get('minimum_cluster_duration', 2.0))
    parser.add_argument('--cluster-hysteresis', type=float, default=memory.get('hysteresis', .5))
    return parser


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.tempo is not None and args.tempo <= 0:
        parser.error('--tempo must be positive')
    if args.smoothing <= 0:
        parser.error('--smoothing must be a positive integer')
    if not np.isfinite(args.movement_weight) or not np.isfinite(args.flow_weight) or min(args.movement_weight, args.flow_weight) < 0 or args.movement_weight + args.flow_weight <= 0:
        parser.error('movement/flow weights must be finite, nonnegative, and sum to a positive value')
    if args.timing_mode == 'fixed' and (args.target_duration is None or not np.isfinite(args.target_duration) or args.target_duration <= 0):
        parser.error('--timing-mode fixed requires a positive --target-duration')
    if args.timing_mode == 'match_video' and args.target_duration is not None:
        parser.error('--target-duration is only valid with --timing-mode fixed')
    if args.clusters < 1:
        parser.error('--clusters must be positive')
    if args.motif_bars is not None and args.motif_bars < 1:
        parser.error('--motif-bars must be positive')
    if not np.isfinite(args.variation_strength) or not 0 <= args.variation_strength <= 1:
        parser.error('--variation-strength must be in [0, 1]')
    if not np.isfinite(args.minimum_cluster_duration) or args.minimum_cluster_duration < 0:
        parser.error('--minimum-cluster-duration must be nonnegative')
    if not np.isfinite(args.cluster_hysteresis) or args.cluster_hysteresis < 0:
        parser.error('--cluster-hysteresis must be nonnegative')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')


def _render_wav(midi_path: Path, score, duration: float, output: Path, args) -> None:
    soundfont = args.soundfont
    fluidsynth = shutil.which('fluidsynth')
    rendered = False
    if fluidsynth and soundfont and Path(soundfont).is_file():
        try:
            subprocess.run([fluidsynth, '-ni', soundfont, str(midi_path), '-F', str(output), '-r', '44100'],
                           check=True, capture_output=True)
            with wave.open(str(output), 'rb') as wav:
                rendered = wav.getframerate() == 44100 and wav.getnchannels() == 2
        except (OSError, subprocess.CalledProcessError, wave.Error):
            rendered = False
    if not rendered:
        render_basic_wav(score['piano_roll'], duration, output, sample_rate=44100, seed=args.seed,
                         instrument_specs=score.get('instrument_specs'))
        if fluidsynth and soundfont and Path(soundfont).is_file():
            print(f'FluidSynth rendering failed for {midi_path.parent.name}; used the built-in basic synth.')
        else:
            print('Using the built-in basic synth for WAV; install FluidSynth and provide --soundfont with a local SF2 for richer instrument rendering. No SoundFont was downloaded.')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config')
    pre_args, _ = pre.parse_known_args(argv)
    config = _load_config(pre_args.config)
    parser = _make_parser(config, argv)
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    # With no explicit style/memory request, compose using the original score
    # path and default tempo/key/scale for backward-compatible outputs.
    if args.styles:
        requested = list(args.styles)
    elif args.style:
        requested = [args.style]
    else:
        requested = [None]
    multi_style = len(requested) > 1 or bool(args.styles) or bool(args.style)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).resolve().parent

    # The expensive VAE pass and shared visual-feature extraction happen once,
    # regardless of how many styles are requested.
    ext = extract_latents(args.checkpoint, args.data, output_root, batch_size=args.batch_size,
                          device=args.device, cache_dir=base / 'data/cache')
    z = ext['latent_means']
    variance = np.var(z, axis=0)
    dims = np.argsort(variance)[-2:][::-1].tolist()
    controls = z[:, dims]
    features = extract_features(ext['timestamps'], controls, ext['latent_logvars'],
                                smoothing=args.smoothing, flow_magnitude=ext['flow_magnitude'])
    features['activity'] = np.clip(args.movement_weight * features['movement'] +
                                   args.flow_weight * features['flow_magnitude'], 0, None)
    if np.max(features['activity']) > 0:
        features['activity'] /= np.max(features['activity'])
    np.save(output_root / 'selected_dims.npy', np.asarray(dims))

    rows = []
    for requested_style in requested:
        style_name = requested_style or 'cinematic'
        style_overrides = _mapping(config, 'music').get('style_options', {})
        if not isinstance(style_overrides, dict):
            parser.error('music.style_options must be a mapping')
        music_config = get_style(style_name, **style_overrides)
        if args.motif_bars is not None:
            music_config = music_config.with_overrides(motif={'phrase_bars': args.motif_bars})
        tempo = args.tempo if args.tempo is not None else (100 if requested_style is None else music_config.tempo_bpm)
        scale = args.scale or ('major' if requested_style is None else music_config.default_scale)
        composer_style = None if requested_style is None and not args.memory else music_config
        style_dir = output_root / style_name if multi_style else output_root
        style_dir.mkdir(parents=True, exist_ok=True)

        latent_memory = None
        if args.memory:
            scale_length = {'major': 7, 'minor': 7, 'dorian': 7, 'pentatonic': 5}[scale]
            motif_library = _make_motif_library(music_config, args.clusters, scale_length)
            latent_memory = LatentMemory(seed=args.seed).fit(
                z, ext['timestamps'], n_clusters=args.clusters,
                min_dwell=args.minimum_cluster_duration,
                hysteresis=args.cluster_hysteresis,
                motif_library=motif_library,
            )
        midi_path = style_dir / 'earth_jukebox.mid'
        score = compose_midi(
            features, midi_path, tempo=tempo, key=args.key, scale=scale,
            timing_mode=args.timing_mode, target_duration=args.target_duration,
            weights=[args.movement_weight, args.flow_weight], seed=args.seed,
            latent_means=z, selected_dims=dims, style=composer_style,
            memory=latent_memory, variation_strength=args.variation_strength,
        )
        save_plots(features, z, style_dir, score['piano_roll'], score['musical_features'],
                   timing_mode=args.timing_mode, duration=score['duration'])
        if latent_memory is not None:
            save_memory_plot(z, ext['timestamps'], latent_memory.labels,
                             style_dir / 'earth_jukebox_memory.png',
                             memory_schedule=score.get('memory_schedule'))
            np.save(style_dir / 'cluster_labels.npy', latent_memory.labels)
            np.save(style_dir / 'cluster_labels_raw.npy', latent_memory.labels_raw)
        wav_path = style_dir / 'earth_jukebox.wav'
        _render_wav(midi_path, score, score['duration'], wav_path, args)
        metrics = composition_metrics(
            score['piano_roll'], score['duration'], style=style_name,
            memory_schedule=score.get('memory_schedule'),
            cluster_labels=latent_memory.labels if latent_memory is not None else None,
            musical_features=score.get('musical_features'), visual_features=features,
            timing_mode=args.timing_mode,
            source_duration=float(ext['timestamps'][-1] - ext['timestamps'][0]) if len(ext['timestamps']) > 1 else None,
        )
        rows.append(metrics)
        (style_dir / 'earth_jukebox_description.txt').write_text(
            f'NASA Earth Jukebox style: {style_name}. The latent dimensions are learned patterns, not named physical quantities. Musical mappings are artistic interpretations. '
            + ('Latent-region musical memory is enabled.' if latent_memory is not None else 'Latent-region musical memory is disabled.')
            + (' Match-video mode follows timestamp seconds.' if args.timing_mode == 'match_video' else f' Fixed mode resamples controls to {score["duration"]:.2f} musical seconds.')
            + '\n', encoding='utf-8')
        print(f'Wrote {midi_path} ({score["duration"]:.2f}s; {score["bars"]} bars; style={style_name}; memory={latent_memory is not None}).')

    write_style_comparison(rows, output_root / 'style_comparison.csv')
    print(f'Shared VAE inference completed once. Style metrics: {output_root / "style_comparison.csv"}.')
    print(f'Selected latent dimensions: {dims}. The model was not trained or modified.')


if __name__ == '__main__':
    main()
