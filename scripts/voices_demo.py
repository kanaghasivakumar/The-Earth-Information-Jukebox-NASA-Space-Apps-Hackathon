"""Voice-layer demos -> demo/voices/*.mp3

    python scripts/voices_demo.py                  # both demos
    python scripts/voices_demo.py --only nightlights

1. nightlights (60 s): the 5313 story in miniature. Twelve "years" (2012 -> 2023), five
   seconds each. Anthrophony (traffic, city hum, engines) rises and drifts from the east
   across the stereo field while biophony (dawn chorus, insects, frogs) thins out. Both sit
   on a slow D-major progression through the resonator (tonal) layer.
2. 3912 sea-surface temperature: the data layer's `activity` curve (frame-to-frame change of
   the global-mean SST, 0..1) drives geophony density/loudness from the MBARI hydrophone and
   surf corpora; warmer global mean -> brighter grains; the resonators are tuned to the
   latitude-band pitches from demo/3912/timeline.json. Also mixed under demo/3912/mix.mp3.

Curves come from outputs/3912/features.npz when the pipeline has been run in this checkout;
otherwise from demo/3912/timeline.json (lead-note velocity = 50 + 50 * activity, so activity
is recovered at every note onset; tied notes hide some frames, which are interpolated).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jukebox.voices import SR, corpus_summary, render_voice  # noqa: E402

OUT = REPO / "demo" / "voices"


def write_mp3(path: Path, x: np.ndarray, sr: int = SR):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1, 1), sr, format="MP3", compression_level=0.75)
    print(f"  wrote {path.relative_to(REPO)}  {path.stat().st_size / 1e3:.0f} kB")


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)


def profile(name, x, step=5.0):
    n = int(step * SR)
    vals = [rms_db(x[i: i + n]) for i in range(0, len(x) - n + 1, n)]
    print(f"  {name:12s} RMS per {step:.0f}s (dBFS): " + " ".join(f"{v:5.1f}" for v in vals))
    return vals


def click_score(x):
    """Largest first-difference spike relative to the local (50 ms) RMS of the difference.
    Hann-windowed grains keep this small; a hard splice shows up as a spike well above ~8."""
    d = np.abs(np.diff(x.mean(1)))
    n = int(0.05 * SR)
    m = len(d) // n * n
    loc = np.sqrt((d[:m].reshape(-1, n) ** 2).mean(1)) + 1e-6
    return float((d[:m].reshape(-1, n).max(1) / loc).max())


# ------------------------------------------------------------------- night lights
def nightlights(seed: int = 0):
    print("night lights (60 s)")
    dur = 60.0
    t = np.linspace(0, dur, 241)
    years = 2012 + t / 5.0
    s = 1 / (1 + np.exp(-(t - 30) / 8))                     # smooth 2012 -> 2023 transition
    s = (s - s[0]) / (s[-1] - s[0])
    anthro = 0.06 + 0.89 * s
    bio = 0.95 - 0.87 * s
    # the year boundaries breathe a little so the arc has a pulse
    pulse = 0.06 * np.cos(2 * np.pi * t / 5.0)
    anthro = np.clip(anthro + pulse * anthro, 0, 1)
    bio = np.clip(bio - pulse * bio, 0, 1)
    pan_a = 0.7 - 0.8 * s                                   # lights spread from the east
    pan_b = -0.2 + 0.0 * t
    bright_a = 0.3 + 0.45 * s
    bright_b = 0.65 + 0.1 * np.sin(2 * np.pi * t / 12)
    # D major, slow: D  Bm  G  A  (7.5 s each, twice)
    chords = [[50, 54, 57, 62], [47, 54, 59, 62], [43, 55, 59, 62], [45, 52, 57, 61]]
    prog = [(i * 7.5, (i + 1) * 7.5, chords[i % 4]) for i in range(8)]

    t0 = time.time()
    a = render_voice("anthrophony", t, anthro, dur, brightness=bright_a, pan=pan_a, seed=seed,
                     tune_to=prog, tune_mix=0.55, width=0.45)
    b = render_voice("biophony", t, bio, dur, brightness=bright_b, pan=pan_b, seed=seed,
                     tune_to=prog, tune_mix=0.2, width=0.8)
    a_raw = render_voice("anthrophony", t, anthro, dur, brightness=bright_a, pan=pan_a, seed=seed, width=0.45)
    b_raw = render_voice("biophony", t, bio, dur, brightness=bright_b, pan=pan_b, seed=seed, width=0.8)
    print(f"  rendered in {time.time() - t0:.1f}s")
    pa, pb = profile("anthrophony", a), profile("biophony", b)
    mix = a + b
    raw = a_raw + b_raw
    print(f"  click score mix {click_score(mix):.1f}, peak {np.abs(mix).max():.2f}, RMS {rms_db(mix):.1f} dBFS")
    write_mp3(OUT / "nightlights.mp3", mix)
    write_mp3(OUT / "nightlights_untuned.mp3", raw)
    (OUT / "nightlights_curves.json").write_text(json.dumps({
        "t": t.round(3).tolist(), "year": years.round(3).tolist(),
        "anthrophony_level": anthro.round(4).tolist(), "biophony_level": bio.round(4).tolist(),
        "anthrophony_pan": pan_a.round(3).tolist(), "chords": prog,
        "rms_db_per_5s": {"anthrophony": np.round(pa, 1).tolist(), "biophony": np.round(pb, 1).tolist()},
    }))


# ------------------------------------------------------------------- 3912 SST
def load_3912():
    f = REPO / "outputs" / "3912" / "features.npz"
    tl = json.loads((REPO / "demo" / "3912" / "timeline.json").read_text())
    bands = [n for n in tl if n["voice"].startswith("band")]
    if f.exists():
        z = np.load(f)
        t, act, mean = z["timestamps"], z["activity"], z["mean"]
        src = str(f.relative_to(REPO))
    else:
        lead = sorted((n for n in tl if n["voice"] == "lead"), key=lambda n: n["start"])
        t = np.array([n["start"] for n in lead])
        act = (np.array([n["velocity"] for n in lead]) - 50) / 50.0
        mean = np.array([n["value"] for n in lead])
        src = "demo/3912/timeline.json (lead velocity)"
    return np.asarray(t, float), np.clip(np.asarray(act, float), 0, 1), np.asarray(mean, float), bands, src


def sst_3912(seed: int = 0):
    t, act, mean, bands, src = load_3912()
    dur = float(t[-1] + 0.5)
    print(f"3912 sea-surface temperature ({dur:.1f} s, activity from {src})")
    # light smoothing: activity is a frame-difference signal and is spiky
    k = np.hanning(5)
    act_s = np.convolve(np.pad(act, 2, mode="edge"), k / k.sum(), mode="valid")
    level = 0.18 + 0.82 * act_s
    ok = np.isfinite(mean)
    m = np.interp(t, t[ok], mean[ok])
    bright = (m - m.min()) / (np.ptp(m) + 1e-9) * 0.6 + 0.2
    # chord timeline from the six latitude-band voices: union of pitches sounding in each 8 s window
    seg = 8.0
    prog = []
    for a in np.arange(0, dur, seg):
        b = min(dur, a + seg)
        p = sorted({n["pitch"] for n in bands if n["start"] < b and n["end"] > a})
        if p:
            prog.append((float(a), float(b), [q + 12 if q < 45 else q for q in p]))
    t0 = time.time()
    geo = render_voice("geophony", t, level, dur, brightness=bright, seed=seed,
                       sources=["hydrophone", "ocean", "water"], tune_to=prog, tune_mix=0.45, width=0.7)
    print(f"  rendered in {time.time() - t0:.1f}s; tuned to {len(prog)} band chords")
    profile("geophony", geo)
    lv5 = [float(np.mean(level[(t >= i) & (t < i + 5)])) for i in np.arange(0, dur - 5 + 1e-9, 5)]
    print("  level  per 5s:              " + " ".join(f"{v:5.2f}" for v in lv5))
    print(f"  click score {click_score(geo):.1f}, peak {np.abs(geo).max():.2f}")
    write_mp3(OUT / "3912_geophony.mp3", geo)
    mixf = REPO / "demo" / "3912" / "mix.mp3"
    if mixf.exists():
        import soundfile as sf
        mus, msr = sf.read(str(mixf), dtype="float32", always_2d=True)
        if msr != SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(msr, SR)
            mus = resample_poly(mus, SR // g, msr // g, axis=0).astype(np.float32)
        if mus.shape[1] == 1:
            mus = np.repeat(mus, 2, 1)
        n = max(len(mus), len(geo))
        out = np.zeros((n, 2), np.float32)
        out[: len(mus)] += 0.8 * mus
        out[: len(geo)] += 0.9 * geo
        pk = np.abs(out).max()
        if pk > 0.95:
            out *= 0.95 / pk
        write_mp3(OUT / "3912_melody_plus_geophony.mp3", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["nightlights", "3912"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    for g, s in corpus_summary().items():
        print(f"{g:12s} {s['minutes']:5.1f} min  {s['files']:2d} files  {s['grains']:5d} grains  "
              f"{'SYNTHETIC STAND-IN ' if s['synthetic'] else ''}{s['minutes_by_tag']}")
    if a.only in (None, "nightlights"):
        nightlights(a.seed)
    if a.only in (None, "3912"):
        sst_3912(a.seed)


if __name__ == "__main__":
    main()
