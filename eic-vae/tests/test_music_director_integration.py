import numpy as np

from music.composer import compose_midi
from music.director import MusicPlan, apply_instrument_choices, apply_plan_mappings
from music.styles import get_style


def _visual_features():
    times = np.arange(0, 24, .125)
    x = np.linspace(0, 2 * np.pi, len(times))
    return {
        'timestamps': times,
        'movement': np.clip(.5 + .45 * np.sin(x), 0, 1),
        'velocity': np.clip(.5 + .45 * np.cos(x * .7), 0, 1),
        'flow_magnitude': np.clip(.5 + .45 * np.sin(x * 1.3), 0, 1),
        'activity': np.clip(.5 + .45 * np.sin(x * .6 + .2), 0, 1),
        'smoothness': np.clip(.5 + .45 * np.cos(x * 1.1), 0, 1),
        'latent_controls': np.column_stack((np.sin(x), np.cos(x * .8))),
    }


def test_validated_plan_changes_existing_composer_midi(tmp_path):
    features = _visual_features()
    plan = MusicPlan(
        style='jazz', tempo_bpm=106, scale='dorian', mood='curious',
        explanation='A nimble, curious reading follows the changing image motion.',
        instruments=[{'role': 'melody', 'timbre': 'tenor sax'}],
        mappings=[
            {'source': 'visual_smoothness', 'target': 'percussion_density', 'strength': .9},
            {'source': 'visual_activity', 'target': 'melody_contour', 'strength': .8},
        ],
    )
    planned_style = get_style(plan.style).with_overrides(
        instruments=apply_instrument_choices(plan.style, plan.instruments)
    )
    baseline = compose_midi(
        features, tmp_path / 'baseline.mid', tempo=plan.tempo_bpm,
        scale=plan.scale, style=get_style(plan.style), seed=14,
    )
    mapped_features = apply_plan_mappings(features, plan.mappings)
    directed = compose_midi(
        mapped_features, tmp_path / 'directed.mid', tempo=plan.tempo_bpm,
        scale=plan.scale, style=planned_style, seed=14,
    )

    assert (tmp_path / 'directed.mid').read_bytes() != (tmp_path / 'baseline.mid').read_bytes()
    assert directed['tempo'] == 106
    assert directed['scale'] == 'dorian'
    assert directed['instrument_programs']['melody'] == 66
    assert directed['piano_roll'] != baseline['piano_roll']
