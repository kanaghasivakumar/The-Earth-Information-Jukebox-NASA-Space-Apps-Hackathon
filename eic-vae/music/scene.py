"""Transparent caption-to-scene rules for optional artistic score shaping.

Caption text only enables named, qualitative cues. Numeric values in prose are
never parsed as rainfall or temperature measurements.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .styles import InstrumentSpec, MusicStyle


@dataclass(frozen=True)
class SceneProfile:
    caption: str
    storm: bool = False
    precipitation: bool = False
    storm_track: bool = False
    sst: bool = False
    half_hourly: bool = False

    @property
    def active(self) -> bool:
        return any((self.storm, self.precipitation, self.storm_track, self.sst))

    def shape_style(self, style: MusicStyle) -> MusicStyle:
        """Apply stable style changes for qualitative caption cues only."""
        changes = {}
        if self.storm:
            rhythm = style.rhythm
            changes['rhythm'] = {
                'kick_beats': tuple(sorted(set(rhythm.kick_beats + (0.0, 2.0)))),
                'snare_beats': tuple(sorted(set(rhythm.snare_beats + (1.0, 3.0)))),
                'hat_subdivision': max(.125, rhythm.hat_subdivision / 2),
            }
            changes['dynamics'] = {'section_contrast': max(.95, style.dynamics.section_contrast)}
        if self.precipitation:
            rhythm = changes.get('rhythm', {})
            rhythm.update({
                'kick_beats': tuple(sorted(set(rhythm.get('kick_beats', style.rhythm.kick_beats) + (0.0, 2.0)))),
                'snare_beats': tuple(sorted(set(rhythm.get('snare_beats', style.rhythm.snare_beats) + (1.0, 3.0)))),
                'hat_subdivision': min(rhythm.get('hat_subdivision', style.rhythm.hat_subdivision), .25),
            })
            changes['rhythm'] = rhythm
        if self.storm_track:
            changes['motif'] = {'nearby_variation_steps': style.motif.nearby_variation_steps + 1}
        shaped = style.with_overrides(**changes) if changes else style
        if self.sst:
            warm = tuple(InstrumentSpec(item.role, 89, 'warm pad (caption interpretation)')
                         if item.role == 'pad' else item for item in shaped.instruments)
            shaped = shaped.with_overrides(instruments=warm)
        return shaped

    def to_dict(self, *, memory_enabled: bool = False) -> dict:
        cues = {
            'storm': self.storm,
            'precipitation': self.precipitation,
            'storm_track': self.storm_track,
            'sst': self.sst,
            'half_hourly_observation': self.half_hourly,
        }
        interpretations = []
        if self.storm:
            interpretations.append('Storm or hurricane wording: a more dramatic, evolving, driving arrangement (slightly denser beat, faster hats, and stronger section contrast).')
        if self.precipitation:
            interpretations.append('Precipitation wording: a more active percussion pattern. This does not estimate rainfall amount or rate.')
        if self.storm_track:
            if memory_enabled:
                interpretations.append('Storm-track wording: the existing latent trajectory selects learned regions and returning motifs; this is used as a visual-motion proxy, not as a measured geographic storm track. Caption shaping also increases the style-shaped distinction between those region motifs.')
            else:
                interpretations.append('Storm-track wording: the melody responds more strongly to the existing selected latent trajectory, used only as a visual-motion proxy. It is not a measured geographic storm track.')
        if self.sst:
            interpretations.append('SST wording: a warmer pad timbre is chosen as an artistic harmonic color. It is not a sea-surface-temperature measurement.')
        if self.half_hourly:
            interpretations.append('Half-hourly observation wording: noted as temporal-resolution context only; it does not change the score or the input timestamps.')
        if not interpretations:
            interpretations.append('No named cue matched; the caption did not change the score.')
        return {
            'caption': self.caption,
            'cues': cues,
            'interpretations': interpretations,
            'measurement_note': 'Caption wording and embedded numbers are not rainfall or SST measurements. This text-only caption interface does not accept numeric measurement series, so it never maps caption numbers to physical quantities.',
            'artistic_note': 'These caption-based choices are artistic interpretations. They are separate from the learned latent-pattern and optical-flow signals used by the visual score.',
        }


def interpret_caption(caption: str | None) -> SceneProfile | None:
    """Classify the requested qualitative cues using deterministic word rules."""
    text = (caption or '').strip()
    if not text:
        return None
    lower = text.lower()
    storm = bool(re.search(r'\b(?:storms?|hurricanes?|cyclones?|typhoons?)\b', lower))
    precipitation = bool(re.search(r'\b(?:precipitation|precip|rainfall|rain|raining)\b', lower))
    storm_track = bool(re.search(r'\bstorm[ -]tracks?\b|\btracks? of (?:the )?(?:storm|hurricane)\b', lower))
    sst = bool(re.search(r'\bSSTs?\b|\bsea[ -]surface temperatures?\b', text, re.IGNORECASE))
    half_hourly = bool(re.search(r'\bhalf[ -]hourly\b|\bevery 30 minutes?\b|\b30[ -]minute(?:ly)?\b|\btwice an hour\b', lower))
    return SceneProfile(text, storm, precipitation, storm_track, sst, half_hourly)


def tempo_delta(profile: SceneProfile | None) -> int:
    """A fixed tempo lift supports the dramatic/driving storm arrangement."""
    return 8 if profile is not None and profile.storm else 0
