import pytest
from music.styles import available_styles, get_style, STYLE_REGISTRY

def test_five_style_presets_have_distinct_configured_arrangements():
    names=available_styles()
    assert names==('ambient','cinematic','classical','jazz','electronic')
    assert set(STYLE_REGISTRY)==set(names)
    presets=[get_style(name) for name in names]
    assert len({s.tempo_bpm for s in presets})==5
    assert len({s.progression for s in presets})>1
    for style in presets:
        assert len(style.instruments)==5
        assert {i.role for i in style.instruments}=={'melody','pad','pluck','bass','drums'}
        assert style.motif.scale_degrees and style.rhythm.hat_subdivision>0
        assert style.chord_degrees==(0,2,4,6) if style.name=='jazz' else style.chord_degrees==(0,2,4)
        assert style.rhythm.bass_beats
    assert all(get_style(n).default_scale=='major' for n in names if n!='electronic')
    assert get_style('electronic').default_scale=='minor'
    assert get_style('jazz').rhythm.bass_beats==(0.,1.,2.,3.)

def test_style_overrides_are_validated_and_do_not_mutate_presets():
    ambient=get_style('ambient',tempo_bpm=84,bars_per_chord=1,dynamics={'peak_velocity':100},motif={'scale_degrees':(0,3,4)},chord_degrees=(0,2,4,6),rhythm={'bass_beats':(0.,1.,2.,3.)})
    assert ambient.tempo_bpm==84 and ambient.bars_per_chord==1
    assert ambient.dynamics.peak_velocity==100 and ambient.motif.scale_degrees==(0,3,4)
    assert ambient.chord_degrees==(0,2,4,6) and ambient.rhythm.bass_beats==(0.,1.,2.,3.)
    assert get_style('ambient').tempo_bpm==78
    with pytest.raises(ValueError): get_style('mystery')
    with pytest.raises(ValueError): get_style('ambient',tempo_bpm=0)
    with pytest.raises(ValueError): get_style('ambient',rhythm={'hat_subdivision':0})
