"""Immutable, typed presets for the standalone NASA music composer.

Use ``get_style('ambient')`` for a preset or pass keyword overrides, e.g.
``get_style('jazz', tempo_bpm=118, bars_per_chord=1)``. Nested parameter groups
can be overridden with mappings such as ``dynamics={'peak_velocity': 104}``.
All pitches in ``progression`` and ``motif`` are scale-degree indices, so a
style remains transposable to any supported key and scale.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

@dataclass(frozen=True)
class InstrumentSpec:
    role: str
    gm_program: int
    timbre: str

@dataclass(frozen=True)
class RhythmParameters:
    kick_beats: tuple[float, ...]
    snare_beats: tuple[float, ...]
    bass_beats: tuple[float, ...]
    hat_subdivision: float
    arpeggio_subdivision: float
    syncopation: float

@dataclass(frozen=True)
class DynamicsParameters:
    floor_velocity: int
    peak_velocity: int
    accent_gain: float
    crescendo_start: float
    crescendo_end: float
    section_contrast: float

@dataclass(frozen=True)
class MotifParameters:
    scale_degrees: tuple[int, ...]
    phrase_bars: int
    nearby_variation_steps: int
    legato: float
    activity_density: float

@dataclass(frozen=True)
class MusicStyle:
    name: str
    tempo_bpm: int
    default_scale: str
    progression: tuple[int, ...]
    chord_degrees: tuple[int, ...]
    bars_per_chord: int
    instruments: tuple[InstrumentSpec, ...]
    rhythm: RhythmParameters
    dynamics: DynamicsParameters
    motif: MotifParameters

    def with_overrides(self, **overrides: object) -> "MusicStyle":
        """Return a validated copy with top-level or nested mapping overrides."""
        nested={"rhythm":RhythmParameters,"dynamics":DynamicsParameters,"motif":MotifParameters}
        cooked=dict(overrides)
        for field, cls in nested.items():
            value=cooked.get(field)
            if isinstance(value, Mapping): cooked[field]=replace(getattr(self,field),**value)
        updated=replace(self,**cooked)
        _validate(updated)
        return updated

def _style(name, tempo, default_scale, progression, chord_degrees, bars_per_chord, programs, rhythm, dynamics, motif):
    instruments=tuple(InstrumentSpec(role,program,timbre) for role,program,timbre in programs)
    return MusicStyle(name,tempo,default_scale,tuple(progression),tuple(chord_degrees),bars_per_chord,instruments,rhythm,dynamics,motif)

_common_dynamics=DynamicsParameters(34,108,1.15,.12,.84,.72)
_styles={
    'ambient':_style('ambient',78,'major',(0,4,5,3),(0,2,4),2,
        [('melody',73,'soft flute'),('pad',89,'warm pad'),('pluck',10,'music box'),('bass',38,'rounded synth bass'),('drums',0,'soft percussion')],
        RhythmParameters((0.,2.),(1.,3.),(0.,2.),1.,1.,.08),_common_dynamics,
        MotifParameters((0,2,4,2),4,1,.86,.55)),
    'cinematic':_style('cinematic',92,'major',(0,4,5,3),(0,2,4),2,
        [('melody',73,'flute lead'),('pad',50,'strings'),('pluck',11,'vibraphone'),('bass',43,'contrabass'),('drums',0,'taiko-style percussion')],
        RhythmParameters((0.,2.),(1.,3.),(0.,2.),.5,1.,.2),
        DynamicsParameters(38,118,1.35,.08,.9,.95),MotifParameters((0,2,4,5,4,2),4,1,.92,.72)),
    'classical':_style('classical',96,'major',(0,3,4,0),(0,2,4),1,
        [('melody',40,'violin'),('pad',48,'string ensemble'),('pluck',0,'piano'),('bass',42,'cello'),('drums',0,'timpani')],
        RhythmParameters((0.,), (2.,),(0.,2.), .5, .5, .12),
        DynamicsParameters(30,102,1.12,.15,.8,.62),MotifParameters((0,2,4,3,2,1),2,1,.98,.68)),
    'jazz':_style('jazz',112,'major',(1,4,0,5),(0,2,4,6),1,
        [('melody',66,'tenor sax'),('pad',4,'electric piano'),('pluck',10,'music box'),('bass',32,'acoustic bass'),('drums',0,'brush kit')],
        RhythmParameters((0.,2.5),(1.,3.),(0.,1.,2.,3.),.5,.5,.42),
        DynamicsParameters(34,108,1.2,.18,.8,.58),MotifParameters((0,2,3,4,2,1),2,1,.72,.82)),
    'electronic':_style('electronic',124,'minor',(0,5,3,4),(0,2,4),2,
        [('melody',81,'saw lead'),('pad',90,'polysynth'),('pluck',9,'glockenspiel'),('bass',38,'synth bass'),('drums',0,'electronic kit')],
        RhythmParameters((0.,1.,2.,3.),(1.,3.),(0.,2.),.25,.5,.28),
        DynamicsParameters(38,116,1.28,.1,.88,.84),MotifParameters((0,0,4,2,5,4),4,1,.66,.95)),
}
STYLE_REGISTRY: Mapping[str,MusicStyle]=MappingProxyType(_styles)

def _validate(style: MusicStyle) -> None:
    if style.tempo_bpm<=0: raise ValueError('tempo_bpm must be positive')
    if style.default_scale not in ('major','minor','dorian','pentatonic'): raise ValueError('default_scale must be a supported scale')
    if style.bars_per_chord<=0: raise ValueError('bars_per_chord must be positive')
    if not style.name or not style.progression or any(degree<0 for degree in style.progression): raise ValueError('style name and nonnegative progression are required')
    if not style.chord_degrees or any(degree<0 for degree in style.chord_degrees): raise ValueError('chord_degrees must contain nonnegative scale degrees')
    if not style.instruments or any(not 0<=inst.gm_program<=127 for inst in style.instruments): raise ValueError('instrument programs must be GM values in [0,127]')
    if not 1<=style.dynamics.floor_velocity<=127 or not 1<=style.dynamics.peak_velocity<=127: raise ValueError('velocity bounds must be in [1,127]')
    if style.dynamics.floor_velocity>style.dynamics.peak_velocity: raise ValueError('floor_velocity cannot exceed peak_velocity')
    if not style.motif.scale_degrees or any(degree<0 for degree in style.motif.scale_degrees): raise ValueError('motif must contain nonnegative scale degrees')
    if style.motif.phrase_bars<=0 or style.motif.nearby_variation_steps<0: raise ValueError('invalid motif phrase settings')
    for value in (style.rhythm.hat_subdivision,style.rhythm.arpeggio_subdivision):
        if value<=0: raise ValueError('rhythm subdivisions must be positive')
    if any(beat<0 for beat in style.rhythm.bass_beats): raise ValueError('bass beats must be nonnegative')
    for value in (style.rhythm.syncopation,style.motif.legato,style.motif.activity_density,style.dynamics.crescendo_start,style.dynamics.crescendo_end,style.dynamics.section_contrast):
        if not 0<=value<=1: raise ValueError('style proportions must be in [0,1]')

for _preset in _styles.values(): _validate(_preset)

def available_styles() -> tuple[str,...]:
    """Return registered style names in stable display order."""
    return tuple(STYLE_REGISTRY)

def get_style(name: str='ambient', **overrides: object) -> MusicStyle:
    """Get a preset copy with optional validated configuration overrides."""
    try: style=STYLE_REGISTRY[name.lower()]
    except KeyError as exc: raise ValueError(f'unknown style {name!r}; choose from {", ".join(STYLE_REGISTRY)}') from exc
    return style.with_overrides(**overrides) if overrides else style
