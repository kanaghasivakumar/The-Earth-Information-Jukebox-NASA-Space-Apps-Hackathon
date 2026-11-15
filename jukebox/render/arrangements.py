"""Per-track arrangement settings for `jukebox.render.arrange` (the 8 jukebox tracks).

    from jukebox.render.arrangements import get
    cfg = get("3912", Path("outputs/3912"))

Instrument choice: the data lead is a flute (GM 73) everywhere. The harmony builder measured that a
flute lead survives pyin under strings + bass at 99 %, while a vibraphone lead fell to 4-8 %.
Bands are soft pads or harp 15 dB below it; model harmony is AMT strings + bass, kept under the lowest
lead note and 13 dB down. Exceptions are noted per track (5058 trumpet, 5669 one instrument per type).
Pitch maps are read from each builder's report.json, so the round trip uses the builder's mapping.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .arrange import Arrangement, Bed, FidSpec

PM_KEYS = ("vmin", "vmax", "low", "span", "root", "scale", "polarity")
ROOT = Path(__file__).resolve().parents[2]
BANDS_3912 = "Average temperature of six latitude bands, 30 degrees each, from 90° North to 90° South. Pitch spans the full colour bar, 0 to 32 °C, so warm tropical bands are high and cold polar bands are low. Each band also sits in its own register, shifted by up to 6 semitones: northern bands a little higher, southern a little lower."
CONTROL_3912 = "Press A to swap the lead for the same temperatures played in random order. It sounds like a tune but no longer follows the ocean."


def _pm(d: dict) -> dict:
    return {k: d[k] for k in PM_KEYS if k in d}


def _report(out: Path) -> dict:
    p = Path(out) / "report.json"
    return json.loads(p.read_text()) if p.exists() else {}


def prepare_3912(out: Path) -> None:
    """3912 comes from jukebox.pipeline (M1), which writes no track.json / controls.json.
    Build them: geophony level from the movie's frame-to-frame activity (moving currents),
    brightness from the global-mean SST; anthrophony and biophony are not measured (zeros)."""
    out = Path(out)
    if (out / "track.json").exists() and (out / "controls.json").exists():
        return
    rep = _report(out)
    meta = json.loads((ROOT / "scripts" / "track_meta" / "3912.json").read_text())
    z = np.load(out / "features.npz")
    t, act, mean = z["timestamps"].astype(float), np.clip(z["activity"].astype(float), 0, 1), z["mean"].astype(float)
    k = np.hanning(5)
    act_s = np.convolve(np.pad(act, 2, mode="edge"), k / k.sum(), mode="valid")
    ok = np.isfinite(mean)
    m = np.interp(t, t[ok], mean[ok])
    geo = 0.18 + 0.82 * act_s
    bright = (m - m.min()) / (np.ptp(m) + 1e-9) * 0.6 + 0.2
    zero = [0.0] * len(t)
    ctrl = {"t": [round(float(x), 3) for x in t],
            "curves": {"geophony": [round(float(x), 4) for x in geo], "anthrophony": zero, "biophony": zero,
                       "brightness": [round(float(x), 4) for x in bright]},
            "note": "geophony = 0.18 + 0.82 x smoothed frame-to-frame activity of the movie (features.npz); "
                    "brightness = normalised global-mean SST"}
    (out / "controls.json").write_text(json.dumps(ctrl))
    fr, fc = rep.get("fidelity_real", {}), rep.get("fidelity_shuffled_control", {})
    track = {
        "id": 3912, "title": meta["title"], "svs_url": "https://svs.gsfc.nasa.gov/3912/",
        "video_url": meta["video_url"], "video_alt_url": meta.get("video_alt_url"),
        "duration_s": rep.get("duration_s", 66.5), "units": rep.get("units", "°C"),
        "description": ("Ocean surface currents swirl around the globe, coloured by sea surface temperature from the "
                        "ECCO2 ocean model (MIT/JPL) on a 0 to 32 °C colour bar, March 2007 to February 2008. The flute "
                        "melody is the global average sea surface temperature, frame by frame: it moves over about "
                        "0.9 °C. Six soft pads underneath are the six latitude bands. A hydrophone recording of the "
                        "Pacific swells with how much the currents move, and the model-written strings and bass "
                        "follow the flute without changing it."),
        "legend": [
            {"voice": "Lead (flute)", "meaning": "Global mean sea surface temperature. Higher pitch means warmer. "
             "The two-octave range covers 18.9 to 19.8 °C, so small changes are easy to hear."},
            {"voice": "Bands (soft pads)", "meaning": BANDS_3912},
            {"voice": "Ocean (geophony)", "meaning": "MBARI Pacific hydrophone and surf recordings; louder and denser "
             "when the currents in the movie move more (frame-to-frame activity), brighter when the ocean is warmer."},
            {"voice": "Shuffled control", "meaning": CONTROL_3912},
        ],
        "stems": [], "timeline": "timeline.json", "controls": "controls.json",
        "fidelity": {"r2": fr.get("r2"), "pearson_r": fr.get("pearson_r"), "mae": fr.get("mae"),
                     "control_r2": fc.get("r2")},
        "data_source": meta["data_source"],
        "band_pitch_map": meta.get("band_pitch_map"), "sweep": meta.get("sweep"),
        "summary": meta.get("summary"), "voice": "geophony",
    }
    (out / "track.json").write_text(json.dumps(track, indent=1, ensure_ascii=False))


def _3912(out):
    prepare_3912(out)
    r = _report(out)
    return Arrangement(
        id="3912", lead_voices=("lead",),
        programs={"lead": 73}, harmony={},
        form="demo/3912/form.json",
        beds=[Bed("geophony", sources=("hydrophone", "ocean", "water"), tune_mix=0.45, width=0.7)],
        fidelity=[FidSpec(("lead",), _pm(r["lead_pitchmap"]), "global mean SST", orig_stem="orig_lead.wav")],
        notes="corrected v2 timeline (recorded inversion flags); form sections from DINOv2 frame novelty")


def _5209(out):
    r = _report(out)
    bands = [f"band {z}" for z in ("64N-90N", "44N-64N", "24N-44N", "EQU-24N", "24S-EQU", "44S-24S", "64S-44S", "90S-64S")]
    return Arrangement(
        id="5209", lead_voices=("lead global",),
        programs={"lead global": 73, **{b: 46 for b in bands}}, cc7={b: 90 for b in bands},
        harmony={},
        beds=[Bed("geophony", sources=("wind", "ocean", "rain"), tune_mix=0.4, width=0.6)],
        fidelity=[FidSpec(("lead global",), _pm(r["lead_pitchmap"]), "global anomaly", orig_stem="orig_lead_global.wav",
                          win=0.18)],
        text_subs=[("the plucked lead plays the global mean and eight voices, strummed",
                    "the flute plays the global mean and eight harp voices, strummed")])


def _5176(out):
    r = _report(out)
    return Arrangement(
        id="5176", lead_voices=("lead global",), programs={"lead global": 73},
        harmony={},
        beds=[Bed("geophony", sources=("ocean", "hydrophone", "water"), tune_mix=0.45, width=0.7)],
        fidelity=[FidSpec(("lead global",), _pm(r["lead_pitchmap"]), "global SST anomaly",
                          orig_stem="orig_lead_global.wav", win=0.22)],
        text_subs=[("The plucked lead follows", "The flute lead follows")])


def _4950(out):
    r = _report(out)
    return Arrangement(
        id="4950", lead_voices=("melody canopy",), programs={"melody canopy": 73},
        harmony={},
        beds=[Bed("geophony", sources=("ocean", "water", "wind"), tune_mix=0.4, width=0.7, gain_db=-3),
              Bed("biophony", sources=("birds", "insects", "frogs"), tune_mix=0.2, width=0.8)],
        fidelity=[FidSpec(("melody canopy",), _pm(r["melody_pitchmap"]), "mean canopy height",
                          orig_stem="orig_melody_canopy.wav", win=0.22)],
        text_subs=[("The plucked melody follows", "The flute melody follows")])


def _5313(out):
    r = _report(out)
    zooms = ("zoom Eastern Mediterranean", "zoom India", "zoom Ukraine", "zoom Western Europe")
    return Arrangement(
        id="5313", lead_voices=("years", "change-sweep"), event_voices=("year-ticks",),
        programs={"years": 73, "change-sweep": 71, "sectors": 46, "year-ticks": 115, **{z: 89 for z in zooms}},
        cc7={"year-ticks": 80},
        harmony={},
        beds=[Bed("geophony", sources=("wind",), tune_mix=0.3, width=0.8),
              Bed("anthrophony", sources=("city", "traffic", "hum"), tune_mix=0.5, width=0.5),
              Bed("biophony", sources=("birds", "insects"), tune_mix=0.2, width=0.8)],
        gains_db={"events": -14.0},
        fidelity=[FidSpec(("years",), _pm(r["pitchmaps"]["years"]), "global light / 2012", orig_stem="orig_years.wav",
                          win=0.45, offset=0.10)],   # skip the wood-block year tick that shares each onset
        text_subs=[("pluck (years)", "flute (years)"), ("pluck arpeggio (sectors)", "harp arpeggio (sectors)"),
                   ("plucked runs (change sweep)", "clarinet runs (change sweep)"),
                   ("ticks (events)", "wood-block ticks (events)")])


def _5011(out):
    r = _report(out)
    return Arrangement(
        id="5011", lead_voices=("storm-belt latitude",), event_voices=("flashes",), drums=("flashes",),
        programs={"storm-belt latitude": 73}, cc7={"flashes": 100},
        harmony=None,   # mostly percussive: model harmony would blur the drum pattern
        beds=[Bed("geophony", events_voice="flashes", sources=("thunder",), length=2.0, gain_db=-2),
              Bed("geophony", sources=("rain", "wind"), tune_mix=0.3, width=0.7, gain_db=-6),
              Bed("biophony", sources=("insects",), tune_mix=0.2, width=0.8)],
        gains_db={"events": -5.0},
        fidelity=[FidSpec(("storm-belt latitude",), _pm(r["latitude_pitchmap"]), "storm-belt latitude",
                          orig_stem="orig_latitude.wav")],
        text_subs=[("The bell line follows", "The flute line follows"), ("bell (melody)", "flute (melody)")],
        notes="no model harmony: percussive track; thunder = render_events at every flash-cluster hit")


def _5669(out):
    r = _report(out)
    pms = r["pitchmap"]
    types = ("black carbon", "dust", "sea salt", "sulfate")
    files = {"black carbon": "orig_black-carbon.wav", "dust": "orig_dust.wav", "sea salt": "orig_sea-salt.wav",
             "sulfate": "orig_sulfate.wav"}
    return Arrangement(
        id="5669", lead_voices=(), split=("bands",),
        # chosen by solo-stem pitch recovery (contrabass/clarinet/oboe/harp 100 %; cello 50 %, choir 80 %)
        programs={"sea salt": 43, "dust": 71, "sulfate": 68, "black carbon": 46},
        harmony=None,   # a west-to-east sweep of a still image: no melodic lead to harmonise
        beds=[Bed("geophony", sources=("wind", "ocean"), tune_mix=0.35, width=0.7),
              Bed("anthrophony", sources=("city", "traffic"), tune_mix=0.45, width=0.5),
              Bed("biophony", sources=("birds", "insects"), tune_mix=0.2, width=0.8)],
        gains_db={"bands": 0.0, "geophony": -12.0, "anthrophony": -12.0, "biophony": -13.0},
        fidelity=[FidSpec((t,), _pm(pms[t]), t, stem=f"bands_{t.replace(' ', '-')}", orig_stem=files[t], solo=True,
                          win=0.2, truth_key="lat") for t in types],
        control=False, mix_fidelity=False,
        text_subs=[("sea salt is a low pad, dust a middle pad, sulfate a high pad and smoke (black carbon) a plucked note",
                    "sea salt is a low double bass, dust a clarinet, sulfate an oboe and smoke (black carbon) a harp"),
                   ("low pad (sea salt)", "double bass (sea salt)"), ("mid pad (dust)", "clarinet (dust)"),
                   ("high pad (sulfate)", "oboe (sulfate)"), ("pluck (black carbon)", "harp (black carbon)")],
        notes="no lead melody; each aerosol type is its own stem, checked as pitch -> latitude on solo notes")


def _5058(out):
    return Arrangement(
        id="5058", lead_voices=("city",), programs={"city": 56, "drone": 89}, gains_db={"bands": -17.0},
        harmony={}, harmony_max_per_s=1.5,
        duck_db=10.0,   # 0.12 s trumpet notes over a drone and city hum: keep the bed a bit further down
        beds=[Bed("anthrophony", sources=("traffic", "city", "hum"), tune_mix=0.3, width=0.5),
              Bed("biophony", sources=("frogs", "insects"), tune_mix=0.2, width=0.8),
              Bed("geophony", sources=("wind", "rain"), tune_mix=0.3, width=0.7)],
        # ian_lights.py: PitchMap(0, log-light max, low=55, span=24, G major pent); vmax refit from the
        # timeline (reproduces all 20 city pitches)
        fidelity=[FidSpec(("city",), {"vmin": 0.0, "vmax": 3.0698, "low": 55, "span": 24, "root": 7,
                                      "scale": "major_pent", "polarity": 1}, "city light runs",
                          orig_stem="orig_city.wav", win=0.11)],
        text_subs=[("pluck runs (city)", "trumpet runs (city)")],
        notes="lead is a trumpet, not a flute: the 0.12 s city notes are too short for the flute attack "
              "(note-level pitch recovery 60 % flute vs 100 % trumpet, solo stem)")


TRACKS = {"3912": _3912, "5209": _5209, "5176": _5176, "4950": _4950, "5313": _5313, "5011": _5011,
          "5669": _5669, "5058": _5058}


def get(track_id: str, out: Path | None = None) -> Arrangement:
    track_id = str(track_id)
    out = Path(out) if out else ROOT / "outputs" / track_id
    if track_id not in TRACKS:
        raise KeyError(f"no arrangement for {track_id}; have {sorted(TRACKS)}")
    return TRACKS[track_id](out)
