#!/usr/bin/env python3
"""Generate Earth Jukebox scores from one saved VAE trajectory."""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from music.composer import compose_midi
from music.diagnostics import composition_metrics, save_memory_plot, write_style_comparison
from music.director import (
    MODEL_DEFAULT,
    InstrumentChoice,
    apply_instrument_choices,
    apply_plan_mappings,
    resolve_music_plan,
    select_representative_frames,
    summarize_visual_features,
)
from music.features import extract_features
from music.inference import extract_latents
from music.memory import LatentMemory, Motif, MotifNote
from music.plots import save_plots
from music.render import render_basic_wav
from music.showcase import build_showcase
from music.scene import interpret_caption, tempo_delta
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


def _schedule_video_times(schedule, timestamps, musical_duration, timing_mode):
    """Convert the composer schedule back to source-video seconds for plotting."""
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if not len(timestamps):
        return list(schedule or [])
    video_start = float(timestamps[0])
    video_duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    musical_duration = max(float(musical_duration), 1e-9)
    converted = []
    for event in schedule or []:
        item = dict(event)
        for key in ('start', 'end'):
            value = float(item.get(key, 0.0))
            if timing_mode == 'fixed':
                item[key] = video_start + value / musical_duration * video_duration
            else:
                item[key] = video_start + value
        converted.append(item)
    return converted


def _director_frame_records(dataset, limit: int = 5) -> list[dict]:
    """Encode at most five evenly spaced representative frames for opt-in vision input."""
    if dataset is None or limit <= 0:
        return []
    source = Path(dataset.sources[0]) if getattr(dataset, 'sources', None) else None
    rgbs = []
    labels = []
    if source and source.is_file() and source.suffix.lower() in {'.mp4', '.mov', '.m4v', '.avi', '.webm'}:
        capture = cv2.VideoCapture(str(source))
        try:
            if capture.isOpened():
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if frame_count > 0:
                    if not np.isfinite(fps) or fps <= 0:
                        fps = 30.0
                    indices = np.unique(np.linspace(0, frame_count - 1, min(limit, frame_count)).round().astype(int))
                    for index in indices:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                        ok, bgr = capture.read()
                        if not ok:
                            continue
                        timestamp = float(index / fps)
                        percent = 0.0 if frame_count <= 1 else 100.0 * index / (frame_count - 1)
                        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        labels.append(f'{percent:.0f}% of video · {timestamp:.1f}s')
        finally:
            capture.release()
    if not rgbs and getattr(dataset, 'samples', None):
        samples = dataset.samples
        indices = np.unique(np.linspace(0, len(samples) - 1, min(limit, len(samples))).round().astype(int))
        for index in indices:
            sample = samples[int(index)]
            # Dataset sample rows are (source_id, timestamp, source_frame_index, RGB, flow).
            rgbs.append(np.asarray(sample[3], dtype=np.uint8))
            labels.append(f'representative frame · {float(sample[1]):.1f}s')

    records = []
    for rgb, label in zip(rgbs, labels):
        image = Image.fromarray(rgb).convert('RGB')
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=82, optimize=True)
        records.append({'label': label, 'bytes': buffer.getvalue()})
    return select_representative_frames(records, limit=limit)


def _parse_instrument_overrides(parser, values):
    choices = []
    for value in values or []:
        role, separator, timbre = value.partition('=')
        if not separator or not role.strip() or not timbre.strip():
            parser.error('--instrument values must use ROLE=TIMBRE, for example --instrument melody="flute lead"')
        try:
            choices.append(InstrumentChoice.model_validate({'role': role.strip(), 'timbre': timbre.strip()}))
        except Exception as exc:
            parser.error(f'invalid --instrument {value!r}: {exc}')
    if len({choice.role for choice in choices}) != len(choices):
        parser.error('provide at most one --instrument override per role')
    return choices


def _make_parser(config: dict, argv=()) -> argparse.ArgumentParser:
    music = _mapping(config, 'music')
    memory = _mapping(config, 'memory')
    output = _mapping(config, 'output')
    director = _mapping(config, 'director')
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
    parser.add_argument('--cache-dir', default=None, help='where decoded frame caches are stored (default: data/cache)')
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
    parser.add_argument('--caption', default=music.get('caption'),
                        help='optional short scene caption for transparent artistic score shaping')
    parser.add_argument('--caption-file', default=music.get('caption_file'),
                        help='optional UTF-8 text file containing the scene caption')
    parser.add_argument('--auto-style', action=argparse.BooleanOptionalAction,
                        default=director.get('enabled', False),
                        help='ask Gemma to choose a validated style and visual-to-music mapping from a description')
    parser.add_argument('--description', default=director.get('description'),
                        help='NASA scientific description for the optional AI Music Director')
    parser.add_argument('--description-file', default=director.get('description_file'),
                        help='UTF-8 file with the NASA scientific description')
    parser.add_argument('--model', default=director.get('model', MODEL_DEFAULT),
                        help='Gemini API model used by the optional AI Music Director')
    parser.add_argument('--offline', '--director-offline', dest='director_offline', action='store_true',
                        default=director.get('offline', False),
                        help='reuse a cached plan or use the registered-style fallback without an API call')
    parser.add_argument('--force-director', action='store_true', default=False,
                        help='ignore a cached plan and request a fresh one')
    parser.add_argument('--director-frames', action='store_true', default=director.get('include_frames', False),
                        help='opt in to sending up to five representative video frames with the description')
    parser.add_argument('--director-timeout', type=float, default=director.get('timeout_seconds', 30.0),
                        help='Gemini request timeout in seconds')
    parser.add_argument('--director-retries', type=int, default=director.get('retries', 2),
                        help='bounded retries for transient Gemini request failures')
    parser.add_argument('--instrument', action='append', default=None, metavar='ROLE=TIMBRE',
                        help='manual instrument override, repeatable; available values come from the selected style')
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
    if args.caption is not None and args.caption_file is not None:
        parser.error('provide only one of --caption and --caption-file (or music.caption/music.caption_file)')
    if args.auto_style:
        if args.description is not None and args.description_file is not None:
            parser.error('provide only one of --description and --description-file')
        if not (args.description or args.description_file):
            parser.error('--auto-style requires --description or --description-file (or director.description in config)')
        if not args.model.strip():
            parser.error('--model must not be empty')
        if not np.isfinite(args.director_timeout) or args.director_timeout <= 0:
            parser.error('--director-timeout must be positive')
        if args.director_retries < 0 or args.director_retries > 5:
            parser.error('--director-retries must be between 0 and 5')
    elif args.description is not None or args.description_file is not None or args.director_frames:
        parser.error('--description, --description-file, and --director-frames require --auto-style')
    args.parsed_instruments = _parse_instrument_overrides(parser, args.instrument)


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
    cli_caption = any(arg == '--caption' or arg.startswith('--caption=') for arg in argv)
    cli_caption_file = any(arg == '--caption-file' or arg.startswith('--caption-file=') for arg in argv)
    if cli_caption and not cli_caption_file:
        args.caption_file = None
    elif cli_caption_file and not cli_caption:
        args.caption = None
    _validate_args(parser, args)
    caption = args.caption
    if args.caption_file:
        try:
            caption = Path(args.caption_file).read_text(encoding='utf-8')
        except OSError as exc:
            parser.error(f'cannot read --caption-file: {exc}')
    scene_profile = interpret_caption(caption)
    scene_profile_data = scene_profile.to_dict(memory_enabled=args.memory) if scene_profile is not None else None
    director_description = args.description
    if args.description_file:
        try:
            director_description = Path(args.description_file).read_text(encoding='utf-8').strip()
        except OSError as exc:
            parser.error(f'cannot read --description-file: {exc}')
    if args.auto_style and not director_description:
        parser.error('the scientific description must not be empty')
    config_music = _mapping(config, 'music')
    configured_instruments = config_music.get('instruments') or []
    if args.parsed_instruments:
        manual_instruments = args.parsed_instruments
    else:
        try:
            manual_instruments = [InstrumentChoice.model_validate(item) for item in configured_instruments]
            if len({item.role for item in manual_instruments}) != len(manual_instruments):
                raise ValueError('music.instruments can specify each role only once')
        except Exception as exc:
            parser.error(f'invalid music.instruments configuration: {exc}')

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).resolve().parent

    # The expensive VAE pass and shared visual-feature extraction happen once,
    # regardless of how many styles are requested.
    ext = extract_latents(args.checkpoint, args.data, output_root, batch_size=args.batch_size,
                          device=args.device, cache_dir=Path(args.cache_dir) if args.cache_dir else base / 'data/cache')
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
    np.savez(output_root / 'visual_signals.npz', timestamps=features['timestamps'],
             movement=features['movement'], flow_magnitude=features['flow_magnitude'])

    director_result = None
    composer_features = features
    if args.auto_style:
        frame_records = _director_frame_records(ext.get('dataset'), limit=5) if args.director_frames else None
        fallback_style = args.style or (args.styles[0] if args.styles else 'cinematic')
        director_result = resolve_music_plan(
            director_description, features, style=fallback_style, tempo=args.tempo, scale=args.scale,
            api_key=os.environ.get('GEMINI_API_KEY'), model=args.model,
            cache_path=output_root / '.music_director_cache', offline=args.director_offline,
            force=args.force_director, frames=frame_records,
            timeout=args.director_timeout, retries=args.director_retries,
            generation_settings={
                'key': args.key, 'timing_mode': args.timing_mode,
                'target_duration': args.target_duration, 'seed': args.seed,
                'memory': args.memory, 'clusters': args.clusters,
                'motif_bars': args.motif_bars, 'variation_strength': args.variation_strength,
                'smoothing': args.smoothing, 'movement_weight': args.movement_weight,
                'flow_weight': args.flow_weight, 'batch_size': args.batch_size,
                'manual_instruments': [choice.model_dump(mode='json') for choice in manual_instruments],
            },
        )
        if director_result.warning:
            print(f'Music Director: {director_result.warning}')
        composer_features = apply_plan_mappings(features, director_result.plan.mappings)

    # Manual selections win over the plan; without a plan the old invocation is unchanged.
    if args.styles:
        requested = list(args.styles)
    elif args.style:
        requested = [args.style]
    elif director_result is not None:
        requested = [director_result.plan.style]
    else:
        requested = [None]
    multi_style = len(requested) > 1 or bool(args.styles) or bool(args.style) or director_result is not None

    plan_record = None
    if director_result is not None:
        plan = director_result.plan
        plan_record = plan.model_dump(mode='json')
        plan_record.update({
            'selected_model': args.model,
            'recommended_style': plan.style,
            'selected_style': requested[0] if len(requested) == 1 else None,
            'selected_styles': requested,
            'plan_source': director_result.source,
            'cache_key': director_result.cache_key,
            'image_input_used': director_result.image_input_used,
            'warning': director_result.warning,
            'manual_overrides': {
                'style': args.style or args.styles,
                'tempo_bpm': args.tempo,
                'scale': args.scale,
                'instruments': [choice.model_dump(mode='json') for choice in manual_instruments],
            },
            'description': director_description,
            'visual_feature_summary': summarize_visual_features(features),
        })
        (output_root / 'music_plan.json').write_text(json.dumps(plan_record, indent=2) + '\n', encoding='utf-8')
        chosen_style_label = ', '.join(requested) if requested and requested[0] is not None else plan.style
        explanation = (
            f"The {chosen_style_label} arrangement uses a {plan.mood} mood at {args.tempo or plan.tempo_bpm} BPM in {args.scale or plan.scale}. "
            f"{plan.explanation} These feature-to-music relationships are artistic choices using normalized visual signals; "
            "they do not represent measured rainfall, temperature, or the physical meaning of latent dimensions."
        )
        (output_root / 'music_explanation.txt').write_text(explanation + '\n', encoding='utf-8')

    rows = []
    showcase_tracks = []
    for requested_style in requested:
        style_name = requested_style or 'cinematic'
        style_overrides = _mapping(config, 'music').get('style_options', {})
        if not isinstance(style_overrides, dict):
            parser.error('music.style_options must be a mapping')
        music_config = get_style(style_name, **style_overrides)
        if args.motif_bars is not None:
            music_config = music_config.with_overrides(motif={'phrase_bars': args.motif_bars})
        if director_result is not None or manual_instruments:
            selected_choices = {}
            if director_result is not None and not (args.style or args.styles):
                selected_choices = {choice.role: choice for choice in director_result.plan.instruments}
            selected_choices.update({choice.role: choice for choice in manual_instruments})
            if selected_choices:
                try:
                    selected_specs = apply_instrument_choices(style_name, list(selected_choices.values()))
                except ValueError as exc:
                    parser.error(f'instrument selection is not supported for style {style_name!r}: {exc}')
                music_config = music_config.with_overrides(instruments=selected_specs)
        if scene_profile is not None and scene_profile.active:
            music_config = scene_profile.shape_style(music_config)
        default_tempo = 100 if requested_style is None else music_config.tempo_bpm
        default_tempo += 0 if director_result is not None else tempo_delta(scene_profile)
        tempo = args.tempo if args.tempo is not None else (
            director_result.plan.tempo_bpm if director_result is not None else default_tempo
        )
        scale = args.scale or (
            director_result.plan.scale if director_result is not None else
            ('major' if requested_style is None else music_config.default_scale)
        )
        composer_style = None if requested_style is None and not args.memory and not manual_instruments and not (scene_profile and scene_profile.active) else music_config
        style_dir = output_root / style_name if multi_style else output_root
        style_dir.mkdir(parents=True, exist_ok=True)
        showcase_tracks.append({'id': style_name if multi_style else 'default',
                                'folder': style_name if multi_style else '.',
                                'style': style_name, 'memory_enabled': bool(args.memory),
                                'instruments': [item.timbre for item in music_config.instruments],
                                'instrument_specs': [{'role': item.role, 'timbre': item.timbre}
                                                     for item in music_config.instruments],
                                'tempo_bpm': tempo, 'scale': scale})

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
            composer_features, midi_path, tempo=tempo, key=args.key, scale=scale,
            timing_mode=args.timing_mode, target_duration=args.target_duration,
            weights=[args.movement_weight, args.flow_weight], seed=args.seed,
            latent_means=z, selected_dims=dims, style=composer_style,
            memory=latent_memory, variation_strength=args.variation_strength,
        )
        save_plots(features, z, style_dir, score['piano_roll'], score['musical_features'],
                   timing_mode=args.timing_mode, duration=score['duration'])
        if latent_memory is not None:
            plot_schedule = _schedule_video_times(score.get('memory_schedule'), ext['timestamps'],
                                                  score['duration'], args.timing_mode)
            save_memory_plot(z, ext['timestamps'], latent_memory.labels,
                             style_dir / 'earth_jukebox_memory.png',
                             memory_schedule=plot_schedule)
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
            + (' Caption interpretation (artistic): ' + ' '.join(scene_profile_data['interpretations'])
               + ' Caption wording and embedded numbers do not measure rainfall or sea-surface temperature.'
               if scene_profile_data else '')
            + (' AI Music Director interpretation: ' + director_result.plan.explanation
               + ' Feature relationships are artistic choices using normalized visual signals.'
               if director_result is not None else '')
            + (' Match-video mode follows timestamp seconds.' if args.timing_mode == 'match_video' else f' Fixed mode resamples controls to {score["duration"]:.2f} musical seconds.')
            + '\n', encoding='utf-8')
        print(f'Wrote {midi_path} ({score["duration"]:.2f}s; {score["bars"]} bars; style={style_name}; memory={latent_memory is not None}).')

    write_style_comparison(rows, output_root / 'style_comparison.csv')
    dataset = ext.get('dataset')
    sources = getattr(dataset, 'sources', []) if dataset is not None else []
    source_video = sources[0] if sources else args.data
    if plan_record is not None:
        plan_record['effective_compositions'] = showcase_tracks
        (output_root / 'music_plan.json').write_text(json.dumps(plan_record, indent=2) + '\n', encoding='utf-8')
    (output_root / 'showcase_manifest.json').write_text(json.dumps({
        'tracks': showcase_tracks,
        'source_video': source_video,
        'caption_profile': scene_profile_data,
        'music_plan': plan_record,
    }, indent=2) + '\n', encoding='utf-8')
    showcase_path = build_showcase(output_root, project_root=base,
                                   source_video=source_video)
    print(f'Listening showcase: {showcase_path}.')
    print(f'Shared VAE inference completed once. Style metrics: {output_root / "style_comparison.csv"}.')
    print(f'Selected latent dimensions: {dims}. The model was not trained or modified.')


if __name__ == '__main__':
    main()
