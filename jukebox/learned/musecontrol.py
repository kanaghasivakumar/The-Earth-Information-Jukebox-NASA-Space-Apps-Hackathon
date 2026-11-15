"""MuseControlLite pilot: data curves -> time-varying controls -> Stable Audio Open -> fidelity check.

The data still fixes the melody. MuseControlLite (Tsai et al., ICML 2025; MIT) only *renders* it: its adapters
take a time-varying melody control (top-4 CQT bins per frame), and the woSDD-all set also takes dynamics
(a smoothed loudness curve) and rhythm (beat/downbeat activations). We build every control from the track's
own data and then test, with the same register-bounded pyin round trip as `jukebox.render.arrange`, whether
the data melody survives:

  mode A  generated audio alone           R² / share within ½ semitone, vs a shuffled-data control that is
                                          generated with the same prompt, seed and settings
  mode B  generated audio as accompaniment under the existing data-lead stem, with arrange's sidechain
          (B-replace: lead + bands + events + generated; B-augment: every current stem + generated)

Steps (each one is a sub-command; `run` does all three):

    python -m jukebox.learned.musecontrol prepare 5209              # copy inputs, build + plot controls (CPU)
    python -m jukebox.learned.musecontrol generate 5209 --device cuda --guidance 1.0 1.5 2.5
    python -m jukebox.learned.musecontrol evaluate 5209             # modes A and B -> mcl_report.json

Inputs  <src>/<id>/  (an arranged track: timeline.json, controls.json, track.json, arrange_report.json, stems)
Output  outputs/<id>_mcl/
  controls_mcl.npz       melody indices (8 x frames per segment), dynamics curve, rhythm activations
  controls_mcl.png       the three controls
  cond_real.wav / cond_control.wav  the legato conditioning tones (data lead / shuffled lead)
  gen_<variant>_{real,control}.wav  generated audio (variant = adapter + guidance)
  mixB_<variant>_{replace,augment}[_control].wav
  mcl_report.json        settings, timings, device, mode A and B fidelity
  mcl.png                spectrogram of the generated audio with the data notes and pyin track
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..render import arrange as A
from ..score.mapping import Note

SR = 44100
SAO_SAMPLES = 2097152                    # Stable Audio Open's fixed latent window: 1024 latents = 47.55 s
SAO_SECONDS = SAO_SAMPLES / SR
MAX_SEG = 47.0                           # generate at most this much per call
XFADE = 4.0                              # equal-power crossfade between segments
BASE_MODEL = "stabilityai/stable-audio-open-1.0"
HF_ADAPTERS = "fundwotsai2001/Text-to-Music_control_family"
ASSETS = Path(os.environ.get("MCL_ASSETS", "/projects/e32706/omb8654/assets"))
DEFAULT_SRC = Path(os.environ.get("MCL_SRC", "/projects/e32706/omb8654/jukebox-wt/integrate/outputs"))
ROOT = Path(__file__).resolve().parents[2]

# Per-track prompt, the lead round trip (same window/offset as arrangements.py) and the rhythm plan.
TRACKS = {
    "5209": dict(
        prompt="Orchestral, warming, solemn strings and piano, slow cinematic build, flute melody",
        win=0.18, offset=0.03,
        rhythm="years",            # one year per data step: beat every 2 years, downbeat every 8
    ),
    "5313": dict(
        prompt="Night city electronic ambient, warm synth pads, soft pulse, glowing, flute melody",
        win=0.45, offset=0.10,
        rhythm="year-ticks",       # downbeat at every year boundary, 4 beats per year
    ),
    "3912": dict(
        prompt="Oceanic ambient, flowing, soft pads and strings, gentle waves, flute melody",
        win=0.30, offset=0.03,
        rhythm="frames",           # beat every data frame (0.5 s), downbeat every 4
    ),
    # added for the B+ build (jukebox.learned.bplus); pilot-style prompts are the "base" comparison there
    "4950": dict(
        prompt="Forest ambient, woodwinds and warm strings, organic, gentle, flute melody",
        win=0.22, offset=0.03,
        rhythm="lead-onsets", lead_voices=("melody canopy",),   # beat at every lead note, downbeat every 4
    ),
    "5176": dict(
        prompt="Ocean ambient, soft pads and strings, calm, warm, flute melody",
        win=0.22, offset=0.03,
        rhythm="lead-onsets", lead_voices=("lead global",),
    ),
    "5011": dict(
        prompt="Stormy ambient, dark strings and synth pads, rolling thunder, flute melody",
        win=0.30, offset=0.03,
        rhythm="lead-onsets", lead_voices=("storm-belt latitude",),   # not the flashes: those are data drums
    ),
    "5058": dict(
        prompt="Somber ambient, slow strings and warm pads, fading city lights, trumpet melody",
        win=0.11, offset=0.03,
        rhythm="lead-onsets", lead_voices=("city",),
    ),
    "5669": dict(
        prompt="Airy ambient, drifting pads and soft strings, dust and haze, gentle",
        win=0.20, offset=0.03,
        rhythm="every", beat_s=1.0,                        # one beat per second (four 0.25 s sweep steps)
    ),
}
COPY_JSON = ("timeline.json", "controls.json", "track.json", "report.json", "arrange_report.json")
COPY_WAV = ("melody.wav", "control_shuffled.wav", "mix.wav", "mix_control.wav", "bands.wav", "events.wav",
            "harmony.wav", "geophony.wav", "anthrophony.wav", "biophony.wav")


# ---------------------------------------------------------------- inputs
def out_dir(tid: str) -> Path:
    return Path(os.environ.get("MCL_OUTROOT", ROOT / "outputs")) / f"{tid}_mcl"


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g, axis=0).astype(np.float32)
    if a.shape[1] == 1:
        a = np.repeat(a, 2, 1)
    return a


def fid_spec(out: Path, tid: str) -> A.FidSpec:
    """The track's first FidSpec, rebuilt from arrange_report.json (voices, PitchMap) + arrangements.py windows."""
    rep = json.loads((out / "arrange_report.json").read_text())
    label, f = next(iter(rep["fidelity"].items()))
    t = TRACKS[tid]
    return A.FidSpec(tuple(f["voices"]), f["pitch_map"], label, win=t["win"], offset=t["offset"])


def load_track(out: Path, tid: str):
    notes = A.load_notes(out / "timeline.json")
    track = json.loads((out / "track.json").read_text())
    spec = fid_spec(out, tid)
    lead = A.fid_notes(notes, spec)
    ctrl = A.shuffled_control(notes, spec.voices, 0)     # arrange's control (cfg.seed = 0)
    ctrl = [n for n in ctrl if n.voice in spec.voices and np.isfinite(A.truth_value(n, spec))]
    duration = float(track.get("duration_s") or max(n.end for n in notes))
    return notes, track, spec, lead, ctrl, duration


# ---------------------------------------------------------------- segments
def segments(total: float, max_seg: float = MAX_SEG, xf: float = XFADE) -> list[tuple[float, float]]:
    """Equal segments of at most max_seg seconds that overlap by xf (one segment if it fits)."""
    if total <= max_seg:
        return [(0.0, total)]
    n = math.ceil((total - xf) / (max_seg - xf))
    L = (total + (n - 1) * xf) / n
    return [(i * (L - xf), i * (L - xf) + L) for i in range(n)]


def stitch(parts: list[np.ndarray], segs, total: float) -> np.ndarray:
    """Equal-power crossfade of the generated segments onto one (N, 2) timeline."""
    N = int(round(total * SR))
    out = np.zeros((N, 2), np.float32)
    for k, (a, (s, e)) in enumerate(zip(parts, segs)):
        i0 = int(round(s * SR))
        a = a[: max(0, min(len(a), N - i0))].copy()
        n = len(a)
        w = np.ones(n, np.float32)
        x = int(round(XFADE * SR))
        if k > 0:
            ov = min(x, n)
            w[:ov] = np.sin(0.5 * np.pi * np.linspace(0, 1, ov))
        if k < len(parts) - 1:
            ov = min(x, n)
            w[n - ov:] = np.cos(0.5 * np.pi * np.linspace(0, 1, ov))
        out[i0:i0 + n] += a * w[:, None]
    return out


# ---------------------------------------------------------------- controls
def legato_tone(notes: list[Note], total: float, harmonics=(1.0, 0.45, 0.25, 0.12)) -> np.ndarray:
    """The data melody as a clean, gap-free tone: each lead note held until the next onset (the first note
    also covers the lead-in, the last one the tail). Silence or reverb tails would feed the melody control
    arbitrary top-4 bins, so the condition is the data notes only."""
    N = int(round(total * SR))
    y = np.zeros(N, np.float64)
    ns = sorted(notes, key=lambda n: n.start)
    if not ns:
        return np.zeros((N, 2), np.float32)
    bounds = [0.0] + [n.start for n in ns[1:]] + [total]
    phase = 0.0
    for n, a, b in zip(ns, bounds[:-1], bounds[1:]):
        i0, i1 = int(round(a * SR)), int(round(b * SR))
        if i1 <= i0:
            continue
        f = 440.0 * 2 ** ((n.pitch - 69) / 12)
        t = np.arange(i1 - i0) / SR
        seg = sum(h * np.sin(2 * np.pi * f * (k + 1) * t + phase * (k + 1)) for k, h in enumerate(harmonics))
        y[i0:i1] = seg
        phase = (phase + 2 * np.pi * f * (i1 - i0) / SR) % (2 * np.pi)   # continuous phase at note changes
    y *= 0.25 / max(1e-9, np.abs(y).max())
    return np.repeat(y[:, None], 2, 1).astype(np.float32)


def fit_window(a: np.ndarray, start: float, n: int = SAO_SAMPLES, hold_last: bool = True) -> np.ndarray:
    """The SAO window that starts at `start`: cut, then pad to 47.55 s by holding the last 0.5 s (the padding
    is trimmed after generation, but the control must cover the whole latent window)."""
    i0 = int(round(start * SR))
    w = a[i0:i0 + n]
    if len(w) < n:
        if hold_last and len(w) > SR // 2:
            loop = w[-SR // 2:]
            reps = int(np.ceil((n - len(w)) / len(loop)))
            w = np.concatenate([w, np.tile(loop, (reps, 1))[: n - len(w)]])
        else:
            w = np.pad(w, ((0, n - len(w)), (0, 0)))
    return w


def _cqt_db(x: np.ndarray) -> np.ndarray:
    """MuseControlLite's `compute_music_represent`: 261.6 Hz high-pass, 128-bin CQT from MIDI 0, hop 512, dB."""
    import librosa
    import torch
    import torchaudio
    y = torchaudio.functional.highpass_biquad(torch.from_numpy(np.ascontiguousarray(x, np.float32)), SR, 261.6)
    c = librosa.cqt(y=y.numpy(), fmin=librosa.midi_to_hz(0), sr=SR, n_bins=128, bins_per_octave=12, hop_length=512)
    return librosa.amplitude_to_db(np.abs(c), ref=np.max)


def melody_indices(stereo: np.ndarray) -> np.ndarray:
    """The HF MuseControlLite melody control (`melody_processing.compute_melody`): per frame and channel the
    top-4 CQT bins (1-based MIDI+1), interleaved [L1 R1 L2 R2 L3 R3 L4 R4] -> (8, frames) int."""
    L, R = _cqt_db(stereo[:, 0]), _cqt_db(stereo[:, 1])
    T = L.shape[1]
    c = np.zeros((8, T), np.int64)
    for j in range(T):
        iL = np.argsort(L[:, j])[-4:][::-1] + 1
        iR = np.argsort(R[:, j])[-4:][::-1] + 1
        c[0::2, j], c[1::2, j] = iL, iR
    return c


def melody_onehot_mono(stereo: np.ndarray) -> np.ndarray:
    """The woSDD-all melody control (`extract_conditions.compute_melody`): union of each channel's top-4 bins."""
    out = None
    for ch in range(2):
        d = _cqt_db(stereo[:, ch])
        if out is None:
            out = np.zeros_like(d, dtype=np.float32)
        idx = np.argsort(d, axis=0)[-4:]
        np.put_along_axis(out, idx, 1.0, axis=0)
    return out


def dynamics_curve(lead_audio: np.ndarray, controls, total: float) -> tuple[np.ndarray, dict]:
    """Dynamics control (`compute_dynamics` format: dB re max, hop 160 at 44.1 kHz, Savitzky-Golay 279).
    Half the lead's own mel-energy envelope, half the voices' controls curves (mean of the non-zero curves,
    mapped to -24..0 dB): the music swells where the data says the planet gets louder."""
    import librosa
    from scipy.signal import savgol_filter
    mono = lead_audio.mean(1)
    S = np.abs(librosa.stft(mono, n_fft=1024, hop_length=160))
    mel = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=64, fmin=0, fmax=8000)
    e = ((mel @ S) ** 2).sum(0)
    lead_db = np.clip(librosa.amplitude_to_db(e, ref=np.max), -40, 0)   # amplitude_to_db of energy, as upstream
    ft = np.arange(len(lead_db)) * 160 / SR
    ctrl_db = np.zeros_like(lead_db)
    used = []
    if controls is not None:
        t, curves, _ = controls
        cs = [v for k, v in curves.items() if k != "brightness" and np.any(v > 1e-4)]
        used = [k for k, v in curves.items() if k != "brightness" and np.any(v > 1e-4)]
        if cs:
            c = np.mean([v / max(1e-9, v.max()) for v in cs], 0)
            ctrl_db = -24 * (1 - np.interp(ft, t, c))
    d = 0.5 * lead_db + 0.5 * ctrl_db
    d = savgol_filter(d, window_length=279, polyorder=1) if len(d) > 279 else d
    return d.astype(np.float32), {"lead_weight": 0.5, "curves": used, "range_db": [float(d.min()), float(d.max())]}


def rhythm_times(tid: str, notes: list[Note], controls, total: float) -> tuple[np.ndarray, np.ndarray, str]:
    """(beats, downbeats) in seconds from the data's own time grid."""
    plan = TRACKS[tid]["rhythm"]
    if plan == "years":
        t = np.asarray(controls[0]) if controls is not None else np.arange(0, total, 0.194)
        beats, down = t[::2], t[::8]
        why = "one year per data step: beat every 2 years, downbeat every 8 years"
    elif plan == "year-ticks":
        ticks = np.array(sorted(n.start for n in notes if n.voice == "year-ticks"))
        if len(ticks) < 2:
            ticks = np.array(sorted({round(n.start, 3) for n in notes if n.voice == "years"}))
        step = np.median(np.diff(ticks))
        beats = np.concatenate([ticks[:, None] + step * np.arange(4)[None] / 4]).ravel()
        down = ticks
        why = f"downbeat at each year boundary ({len(ticks)}), 4 beats per year ({step / 4:.2f} s)"
    elif plan == "lead-onsets":
        voices = TRACKS[tid]["lead_voices"]
        on = np.array(sorted({round(n.start, 3) for n in notes if n.voice in voices}))
        beats, down = on, on[::4]
        why = f"a beat at every lead note ({len(on)}), downbeat every 4"
    elif plan == "every":
        step = TRACKS[tid]["beat_s"]
        beats = np.arange(0, total, step)
        down = beats[::4]
        why = f"a beat every {step:g} s, downbeat every 4"
    else:
        t = np.asarray(controls[0]) if controls is not None else np.arange(0, total, 0.5)
        beats, down = t, t[::4]
        why = f"beat every data frame ({np.median(np.diff(t)):.2f} s), downbeat every 4"
    beats = beats[(beats >= 0) & (beats < total)]
    down = down[(down >= 0) & (down < total)]
    return np.unique(beats), np.unique(down), why


def rhythm_activation(beats, down, total: float, fps: int = 100) -> np.ndarray:
    """madmom RNNDownBeatProcessor-style activations (frames, 2) = [beat (non-downbeat), downbeat]."""
    T = int(np.ceil(total * fps))
    t = np.arange(T) / fps
    act = np.zeros((T, 2), np.float32)
    sd = 0.02
    for col, times in ((0, np.setdiff1d(np.round(beats, 3), np.round(down, 3))), (1, down)):
        for b in times:
            act[:, col] = np.maximum(act[:, col], 0.9 * np.exp(-0.5 * ((t - b) / sd) ** 2))
    return act


def prepare(tid: str, src: Path, out: Path | None = None, plot: bool = True) -> dict:
    out = out or out_dir(tid)
    out.mkdir(parents=True, exist_ok=True)
    s = src / tid
    for f in COPY_JSON + COPY_WAV:
        if (s / f).exists() and not (out / f).exists():
            shutil.copy2(s / f, out / f)
    notes, track, spec, lead, ctrl, duration = load_track(out, tid)
    controls = A._controls(out, duration)
    mel = read_wav(out / "melody.wav")
    total = len(mel) / SR                                    # duration + arrange's tail: the stems' length
    segs = segments(total)
    cond_real = legato_tone(lead, total)
    cond_ctrl = legato_tone(ctrl, total)
    A.write_wav(out / "cond_real.wav", cond_real)
    A.write_wav(out / "cond_control.wav", cond_ctrl)
    dyn, dyn_info = dynamics_curve(mel, controls, total)
    beats, down, why = rhythm_times(tid, notes, controls, total)
    act = rhythm_activation(beats, down, total)
    arrays = {"dynamics": dyn, "rhythm": act, "beats": beats, "downbeats": down,
              "segments": np.array(segs, float)}
    t0 = time.time()
    for k, (a, b) in enumerate(segs):
        arrays[f"melody_real_{k}"] = melody_indices(fit_window(cond_real, a))
        arrays[f"melody_control_{k}"] = melody_indices(fit_window(cond_ctrl, a))
    # "full" condition: the data lead plus the arranged harmony (AMT) and band stems 6 dB under it, so the
    # top-4 bins also carry chord tones (a richer accompaniment, at some risk to the lead)
    accs = [read_wav(out / f) for f in ("harmony.wav", "bands.wav") if (out / f).exists()]
    full_info = None
    if accs:
        n = len(cond_real)
        acc = sum(np.pad(x[:n], ((0, max(0, n - len(x))), (0, 0))) for x in accs)
        gain = np.float32(A.active_rms(cond_real) * 10 ** (-6 / 20) / max(1e-9, A.active_rms(acc)))
        full_real, full_ctrl = cond_real + acc * gain, cond_ctrl + acc * gain
        A.write_wav(out / "cond_real_full.wav", full_real)
        A.write_wav(out / "cond_control_full.wav", full_ctrl)
        for k, (a, b) in enumerate(segs):
            arrays[f"melody_real_full_{k}"] = melody_indices(fit_window(full_real, a))
            arrays[f"melody_control_full_{k}"] = melody_indices(fit_window(full_ctrl, a))
        full_info = {"stems": [f for f in ("harmony.wav", "bands.wav") if (out / f).exists()], "db_vs_lead": -6}
    t_mel = time.time() - t0
    np.savez_compressed(out / "controls_mcl.npz", **arrays)
    meta = {"id": tid, "prompt": TRACKS[tid]["prompt"], "duration_s": duration, "total_s": total,
            "segments": segs, "crossfade_s": XFADE if len(segs) > 1 else 0.0,
            "melody": {"source": "legato tone of the data lead notes (fidelity voices only)",
                       "voices": list(spec.voices), "n_notes": len(lead),
                       "format": "top-4 CQT bins per channel (8 x frames), MIDI+1, hop 512",
                       "seconds": round(t_mel, 1)},
            "full_condition": full_info,
            "control_melody": "arrange's shuffled control lead (seed 0), same slots, pitches permuted",
            "dynamics": {**dyn_info, "frames": int(len(dyn)), "hop": 160},
            "rhythm": {"plan": why, "n_beats": int(len(beats)), "n_downbeats": int(len(down)), "fps": 100},
            "fidelity_spec": {"label": spec.label, "voices": list(spec.voices), "pitch_map": spec.pm,
                              "win": spec.win, "offset": spec.offset}}
    (out / "controls_mcl.json").write_text(json.dumps(A.json_safe(meta), indent=1))
    if plot:
        _plot_controls(out, arrays, lead, total, meta)
    return meta


def _plot_controls(out, arrays, lead, total, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    m = arrays["melody_real_0"]
    segs = meta["segments"]
    for k, (a, b) in enumerate(segs):
        m = arrays[f"melody_real_{k}"]
        ft = a + np.arange(m.shape[1]) * 512 / SR
        keep = ft <= b
        for r in range(8):
            ax[0].scatter(ft[keep][::4], m[r, keep][::4] - 1, s=1, c="C0" if r < 2 else "0.7")
    for n in lead:
        ax[0].plot([n.start, n.end], [n.pitch, n.pitch], c="C3", lw=2)
    ax[0].set_ylabel("MIDI (control bins)")
    ax[0].set_title(f"{meta['id']} melody control: top-4 CQT bins (blue = top-1), data lead notes (red)")
    lo = min(n.pitch for n in lead)
    ax[0].set_ylim(lo - 14, lo + 60)
    d = arrays["dynamics"]
    ax[1].plot(np.arange(len(d)) * 160 / SR, d)
    ax[1].set_ylabel("dB")
    ax[1].set_title("dynamics control: lead envelope + voices' controls curves")
    act = arrays["rhythm"]
    tt = np.arange(len(act)) / 100
    ax[2].plot(tt, act[:, 0], lw=0.6, label="beat")
    ax[2].plot(tt, act[:, 1], lw=0.8, label="downbeat")
    ax[2].legend(loc="upper right")
    ax[2].set_title("rhythm control: " + meta["rhythm"]["plan"])
    ax[2].set_xlabel("s")
    for a in ax:
        for (s, e) in segs[1:]:
            a.axvline(s, c="k", ls=":", lw=0.8)
    ax[2].set_xlim(0, total)
    fig.tight_layout()
    fig.savefig(out / "controls_mcl.png", dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- model
def _import_hf_package():
    """The HF repo's custom pipeline code (pipeline.py, stable_audio_multi_cfg_pipe.py, ...) as package `mcl_hf`."""
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(HF_ADAPTERS, allow_patterns=["*.py"]))
    pkg = ASSETS / "mcl_hf_pkg" / "mcl_hf"
    pkg.mkdir(parents=True, exist_ok=True)
    for f in snap.glob("*.py"):
        shutil.copy2(f, pkg / f.name)
    (pkg / "__init__.py").write_text("")
    if str(pkg.parent) not in sys.path:
        sys.path.insert(0, str(pkg.parent))
    import mcl_hf.pipeline as P
    return P


SAO_PARTS = ["model_index.json", "LICENSE.md", "projection_model/*", "scheduler/*", "text_encoder/*", "tokenizer/*",
             "transformer/*", "vae/*"]


def base_model_path() -> str:
    """Local snapshot of the diffusers parts only. Passing a repo id to from_pretrained would also pull the
    4.9 GB stable-audio-tools checkpoint at the repo root, which diffusers never reads."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(BASE_MODEL, allow_patterns=SAO_PARTS, local_files_only=True)
    except Exception:
        return snapshot_download(BASE_MODEL, allow_patterns=SAO_PARTS)


def load_pipeline(adapter: str, device: str, dtype_name: str):
    """adapter: 'hf-melody' (HF MuseControlLite melody adapters, 8x top-4 indices) or 'wosdd-all'
    (the GitHub release's melody_mono + dynamics + rhythm adapters, from ASSETS/MuseControlLite/checkpoints)."""
    import torch
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    P = _import_hf_package()
    extractors = {}
    if adapter == "hf-melody":
        pipe = P.MuseControlLiteStableAudioPipeline.from_pretrained(
            base_model_path(), torch_dtype=dtype, attn_processor_type="MuseControlLite", adapter_repo_id=HF_ADAPTERS,
            adapter_checkpoint="MuseControlLite/adapters.safetensors",
            melody_encoder_checkpoint="MuseControlLite/melody_encoder.safetensors")
    elif adapter == "wosdd-all":
        from safetensors.torch import load_file
        from mcl_hf.stable_audio_multi_cfg_pipe import StableAudioPipeline
        from mcl_hf.MuseControlLite_attn_processor import StableAudioAttnProcessor2_0, StableAudioAttnProcessor2_0_rotary
        ck = ASSETS / "MuseControlLite" / "checkpoints" / "woSDD-all"
        sys.path.insert(0, str(ASSETS / "MuseControlLite"))
        from utils.feature_extractor import dynamics_extractor, melody_extractor_mono, rhythm_extractor
        pipe = StableAudioPipeline.from_pretrained(base_model_path(), torch_dtype=dtype)
        sd = load_file(str(ck / "model_3.safetensors"), device="cpu")
        procs = {}
        for name in pipe.transformer.attn_processors.keys():
            if name.endswith("attn1.processor"):
                procs[name] = StableAudioAttnProcessor2_0()
            else:
                p = StableAudioAttnProcessor2_0_rotary(layer_id=name.split(".")[1], hidden_size=768, name=name,
                                                       cross_attention_dim=768, scale=1.0)
                for w in ("to_k_ip", "to_v_ip", "conv_out"):
                    getattr(p, w).weight = torch.nn.Parameter(sd[f"{name}.{w}.weight"].float())
                procs[name] = p.to(dtype=dtype)
        pipe.transformer.set_attn_processor(procs)
        for k, cls, f in (("dynamics", dynamics_extractor, "model_1"), ("melody", melody_extractor_mono, "model"),
                          ("rhythm", rhythm_extractor, "model_2")):
            m = cls()
            m.load_state_dict(load_file(str(ck / f"{f}.safetensors"), device="cpu"))
            extractors[k] = m.eval().float().to(device)
    else:
        raise ValueError(adapter)
    pipe.scheduler.config.sigma_min = 0.3
    pipe.scheduler.config.sigma_max = 500
    pipe = pipe.to(device)
    if adapter == "wosdd-all":   # the HF subclass moves its processors itself; this one needs it done here
        for proc in pipe.transformer.attn_processors.values():
            if hasattr(proc, "to"):
                proc.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe, extractors


def build_condition(pipe, extractors, adapter, seg_arrays: dict, device: str):
    """-> (3, 1024, 768): [masked, masked, condition] for the pipeline's three-way CFG."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        if adapter == "hf-melody":
            idx = torch.from_numpy(seg_arrays["melody_idx"]).long().unsqueeze(0).to(device)
            x = pipe.melody_encoder(idx)
            x = F.interpolate(x.float(), size=1024, mode="linear", align_corners=False)
        else:
            def ex(k, arr):
                y = extractors[k](torch.from_numpy(arr).float().unsqueeze(0).to(device))
                return F.interpolate(y, size=1024, mode="linear", align_corners=False)
            x = torch.cat([ex("rhythm", seg_arrays["rhythm"].T), ex("dynamics", seg_arrays["dynamics"][None]),
                           ex("melody", seg_arrays["melody_onehot"]), torch.zeros(1, 192, 1024, device=device)], 1)
        z = torch.zeros_like(x)
        dtype = next(pipe.transformer.parameters()).dtype
        return torch.cat([z, z, x], 0).transpose(1, 2).to(dtype)


def seg_inputs(adapter, arrays, which, k, cond_audio, a, b):
    """Per-segment model inputs, cut to the SAO window that starts at a."""
    d = {}
    if adapter == "hf-melody":
        d["melody_idx"] = arrays[f"melody_{which}_{k}"]   # which = real | control | real_full | control_full
    else:
        d["melody_onehot"] = melody_onehot_mono(fit_window(cond_audio, a))
        dyn = arrays["dynamics"]
        i0 = int(round(a * SR / 160))
        n = 1 + SAO_SAMPLES // 160
        w = dyn[i0:i0 + n]
        d["dynamics"] = np.pad(w, (0, n - len(w)), mode="edge") if len(w) < n else w
        act = arrays["rhythm"]
        i0, n = int(round(a * 100)), int(np.ceil(SAO_SECONDS * 100))
        w = act[i0:i0 + n]
        d["rhythm"] = np.pad(w, ((0, n - len(w)), (0, 0))) if len(w) < n else w
    return d


def generate(tid: str, adapter: str = "hf-melody", device: str = "cuda", steps: int = 50,
             guidance: list[float] = (1.5,), seed: int = 42, dtype: str = "fp16", max_seconds: float | None = None,
             which: tuple = ("real", "control"), out: Path | None = None, cond: str = "lead") -> dict:
    import torch
    out = out or out_dir(tid)
    meta = json.loads((out / "controls_mcl.json").read_text())
    arrays = dict(np.load(out / "controls_mcl.npz"))
    total = meta["total_s"] if max_seconds is None else min(meta["total_s"], max_seconds)
    segs = [tuple(s) for s in meta["segments"]]
    if max_seconds is not None:            # CPU smoke test: first segment(s) only, trimmed
        segs = [(a, min(b, total)) for a, b in segs if a < total]
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
        dtype = "fp32"
    t0 = time.time()
    pipe, extractors = load_pipeline(adapter, device, dtype)
    t_load = time.time() - t0
    dev_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else f"CPU x{torch.get_num_threads()}"
    runs = []
    rep_path = out / "gen_runs.json"
    old = json.loads(rep_path.read_text()) if rep_path.exists() else []
    for g in guidance:
        sfx = "" if cond == "lead" else f"_{cond}"
        variant = (f"{adapter}_g{g:g}" + (f"_s{steps}" if steps != 50 else "") + sfx
                   + (f"_seed{seed}" if seed != 42 else "") + ("_smoke" if max_seconds else ""))
        for w in which:
            cond_audio = read_wav(out / f"cond_{w}{sfx}.wav") if adapter != "hf-melody" else None
            parts = []
            t1 = time.time()
            for k, (a, b) in enumerate(segs):
                c = build_condition(pipe, extractors, adapter, seg_inputs(adapter, arrays, w + sfx, k, cond_audio, a, b),
                                    device)
                with torch.no_grad():
                    wav = pipe(extracted_condition=c, prompt=meta["prompt"], negative_prompt="",
                               num_inference_steps=steps, guidance_scale_text=7.0, guidance_scale_con=g,
                               num_waveforms_per_prompt=1, audio_end_in_s=float(b - a),
                               generator=torch.Generator().manual_seed(seed)).audios
                parts.append(wav[0].T.float().cpu().numpy())
            secs = time.time() - t1
            audio = stitch(parts, segs, total)
            audio *= np.float32(0.89 / max(1e-9, np.abs(audio).max()))   # -1 dBFS peak
            A.write_wav(out / f"gen_{variant}_{w}.wav", audio)
            r = {"variant": variant, "which": w, "adapter": adapter, "condition": cond, "guidance_con": g, "guidance_text": 7.0,
                 "steps": steps, "seed": seed, "dtype": dtype, "device": dev_name, "segments": segs,
                 "audio_s": total, "gen_seconds": round(secs, 1), "seconds_per_audio_minute": round(secs / total * 60, 1),
                 "load_seconds": round(t_load, 1), "prompt": meta["prompt"]}
            runs.append(r)
            print(json.dumps(r), flush=True)
            old = json.loads(rep_path.read_text()) if rep_path.exists() else []
            old = [o for o in old if not (o["variant"] == variant and o["which"] == w)] + [r]
            rep_path.write_text(json.dumps(old, indent=1))
    return {"runs": runs}


# ---------------------------------------------------------------- evaluation
def chroma_accuracy(cond: np.ndarray, gen: np.ndarray) -> float:
    """MuseControlLite's own melody metric: frame-wise argmax-chroma agreement (261 Hz high-pass)."""
    import librosa
    from scipy import signal
    def oh(x):
        y = x.mean(1)
        b, a = signal.butter(2, 261.2 / (0.5 * SR), btype="high")
        c = librosa.feature.chroma_stft(y=signal.filtfilt(b, a, y), sr=SR, n_fft=2048, hop_length=256)
        return np.argmax(c, 0)
    p, q = oh(cond), oh(gen)
    n = min(len(p), len(q))
    return float(np.mean(p[:n] == q[:n]))


def richness(audio: np.ndarray, notes: list[Note]) -> dict:
    """How much is there besides the lead? Share of spectral energy (100 Hz..8 kHz, while lead notes sound)
    that lies outside ±60 cents of the lead's first 10 harmonics, plus mean spectral flatness."""
    import librosa
    y = audio.mean(1)
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=4096)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=SR, hop_length=1024)
    band = (f >= 100) & (f <= 8000)
    off, tot = 0.0, 0.0
    for n in notes:
        cols = (t >= n.start + 0.05) & (t < n.end)
        if not cols.any():
            continue
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12)
        near = np.zeros_like(f, bool)
        for h in range(1, 11):
            near |= np.abs(1200 * np.log2(np.maximum(f, 1) / (h * f0))) < 60
        e = S[:, cols][band]
        tot += e.sum()
        off += e[~near[band]].sum()
    return {"non_lead_energy_share": float(off / tot) if tot > 0 else None,
            "spectral_flatness": float(librosa.feature.spectral_flatness(y=y).mean())}


def octave_folded(audio: np.ndarray, notes: list[Note], spec: A.FidSpec) -> dict:
    """Diagnostic: wide-range pyin (C2..C7); is each note's pitch class right even if the octave is not?"""
    tr = A.pyin_track(audio, 65.4, 2093.0)
    est = A.note_estimates(tr, notes, spec.win, spec.offset)
    ok = np.isfinite(est)
    if not ok.any():
        return {"voiced": 0.0}
    d = est[ok] - np.array([n.pitch for n in notes])[ok]
    fold = np.abs((d + 6) % 12 - 6)
    return {"voiced": float(ok.mean()), "pc_within_half_semitone": float((fold < 0.5).mean()),
            "exact_within_half_semitone": float((np.abs(d) < 0.5).mean()),
            "median_octave_offset": float(np.median(np.round(d / 12)))}


def mode_a(audio, notes, spec, lead, ctrl, ctrl_audio=None) -> dict:
    rng = A.pyin_range(notes, spec)
    r = {"real": A.fidelity(audio, notes, spec, track=A.pyin_track(audio, *rng))}
    r["real"]["octave_folded"] = octave_folded(audio, lead, spec)
    if ctrl_audio is not None:
        r["control"] = A.fidelity(ctrl_audio, ctrl, spec, truth_notes=lead, track=A.pyin_track(ctrl_audio, *rng))
    return r


def mix_b(out: Path, track: dict, gen: np.ndarray, lead_audio: np.ndarray, how: str, gen_db: float) -> tuple:
    """Rebuild a mix from the current (already ducked, normalised) stems plus the generated audio, with arrange's
    sidechain over the whole accompaniment bus and arrange's loudness/limiter rules."""
    stems = {}
    for s in track["stems"]:
        g = s["group"]
        if how == "replace" and g in ("harmony", "geophony", "anthrophony", "biophony"):
            continue
        p = out / s["file"]
        if p.exists() and g != "melody":
            stems[s["name"]] = (g, read_wav(p))
    n = len(lead_audio)
    fit = lambda a: np.pad(a[:n], ((0, max(0, n - len(a))), (0, 0)))
    ref = A.active_rms(lead_audio)
    g_gain = ref * 10 ** (gen_db / 20) / max(1e-9, A.active_rms(fit(gen)))
    stems["generated"] = ("harmony", fit(gen) * np.float32(g_gain))
    acc = [k for k, (g, _) in stems.items() if g in ("bands", "harmony", "geophony", "anthrophony", "biophony")]
    gd = A.duck_gain(lead_audio, sum(stems[k][1] for k in acc), 8.0, 9.0)
    for k in acc:
        stems[k] = (stems[k][0], stems[k][1] * gd[:, None])
    acc_sum = sum(a for _, a in stems.values())

    def finish(lead):
        pre = lead + acc_sum
        L0 = A.lufs(pre)
        k = 10 ** ((-16.0 - L0) / 20)
        k = min(k, 10 ** (4.5 / 20) * 10 ** (-1 / 20) / max(float(np.abs(pre).max()), 1e-9))
        mix, gr = A.limiter((pre * np.float32(k)).astype(np.float32))
        return mix, {"lufs": A.lufs(mix), "limiter_gr_db": gr, "norm_gain_db": A.db(k)}
    duck = {"mean_gr_db": float(-20 * np.log10(np.mean(gd))), "max_gr_db": float(-20 * np.log10(gd.min()))}
    return finish, {"stems": sorted(stems), "generated_gain_db_vs_lead": gen_db, "ducking": duck}


def evaluate(tid: str, out: Path | None = None, variants: list[str] | None = None, plot: bool = True) -> dict:
    out = out or out_dir(tid)
    notes, track, spec, lead, ctrl, duration = load_track(out, tid)
    meta = json.loads((out / "controls_mcl.json").read_text())
    runs = json.loads((out / "gen_runs.json").read_text()) if (out / "gen_runs.json").exists() else []
    lead_audio = read_wav(out / "melody.wav")
    ctrl_lead_audio = read_wav(out / "control_shuffled.wav")
    rng = A.pyin_range(notes, spec)
    rep = {"id": tid, "prompt": meta["prompt"], "fidelity_spec": meta["fidelity_spec"],
           "method": "note-level register-bounded pyin (arrange.fidelity), same notes, PitchMap and windows",
           "reference": {}, "variants": {}}
    for name, f in (("integrated_mix", "mix.wav"), ("integrated_mix_control", "mix_control.wav"),
                    ("lead_stem", "melody.wav"), ("cond_real", "cond_real.wav")):
        if (out / f).exists():
            a = read_wav(out / f)
            if name == "integrated_mix_control":
                rep["reference"][name] = A.fidelity(a, ctrl, spec, truth_notes=lead, track=A.pyin_track(a, *rng))
            else:
                rep["reference"][name] = A.fidelity(a, notes, spec, track=A.pyin_track(a, *rng))
            rep["reference"][name]["richness"] = richness(a, lead)
    vs = sorted({r["variant"] for r in runs})
    if variants:
        vs = [v for v in vs if v in variants]
    for v in vs:
        pr, pc = out / f"gen_{v}_real.wav", out / f"gen_{v}_control.wav"
        if not pr.exists():
            continue
        gen = read_wav(pr)
        genc = read_wav(pc) if pc.exists() else None
        span = len(gen) / SR
        ln = [n for n in lead if n.start + spec.win <= span]
        cn = [n for n in ctrl if n.start + spec.win <= span]
        sub = [n for n in notes if n.start + spec.win <= span]   # smoke tests cover only the first seconds
        r = {"runs": [x for x in runs if x["variant"] == v], "audio_s": span, "n_lead_notes": len(ln)}
        r["A"] = mode_a(gen, sub, spec, ln, cn, genc)
        sfx = "_full" if "_full" in v else ""
        cond_real = read_wav(out / f"cond_real{sfx}.wav")
        r["A"]["chroma_accuracy_vs_condition"] = chroma_accuracy(cond_real[: len(gen)], gen)
        r["A"]["richness"] = richness(gen, ln)
        if genc is not None:
            cc = read_wav(out / f"cond_control{sfx}.wav")
            r["A"]["chroma_accuracy_control_vs_its_condition"] = chroma_accuracy(cc[: len(genc)], genc)
        r["B"] = {}
        if span >= 0.95 * len(lead_audio) / SR:
            for how, gdb in (("replace", -10.0), ("augment", -13.0)):
                finish, info = mix_b(out, track, gen, lead_audio, how, gdb)
                mix, li = finish(lead_audio)
                mixc, _ = finish(ctrl_lead_audio)
                A.write_wav(out / f"mixB_{v}_{how}.wav", mix)
                A.write_wav(out / f"mixB_{v}_{how}_control.wav", mixc)
                r["B"][how] = {**info, **li,
                               "mix": A.fidelity(mix, notes, spec, track=A.pyin_track(mix, *rng)),
                               "control": A.fidelity(mixc, ctrl, spec, truth_notes=lead,
                                                     track=A.pyin_track(mixc, *rng))}
                if genc is not None:   # end to end: shuffled lead AND the accompaniment generated from it
                    finish_c, _ = mix_b(out, track, genc, ctrl_lead_audio, how, gdb)
                    mixe, _ = finish_c(ctrl_lead_audio)
                    A.write_wav(out / f"mixB_{v}_{how}_control_e2e.wav", mixe)
                    r["B"][how]["control_e2e"] = A.fidelity(mixe, ctrl, spec, truth_notes=lead,
                                                            track=A.pyin_track(mixe, *rng))
        rep["variants"][v] = r
        if plot:
            _plot_gen(out, v, gen, ln, spec, rng)
    (out / "mcl_report.json").write_text(json.dumps(A.json_safe(rep), indent=1))
    return rep


def _plot_gen(out, v, gen, lead, spec, rng):
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = gen.mean(1)
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=SR, hop_length=512, fmin=librosa.midi_to_hz(36),
                                                   n_bins=72, bins_per_octave=12)), ref=np.max)
    tr = A.pyin_track(gen, *rng)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(C, origin="lower", aspect="auto", cmap="magma", vmin=-60,
              extent=[0, len(y) / SR, 35.5, 36 + 72 - 0.5])
    for n in lead:
        ax.plot([n.start, n.end], [n.pitch, n.pitch], c="cyan", lw=2, alpha=0.8)
    ax.plot(tr[0], tr[1], ".", c="lime", ms=1.5)
    ax.set_ylabel("MIDI")
    ax.set_xlabel("s")
    ax.set_title(f"{out.name} {v}: generated CQT, data lead notes (cyan), register-bounded pyin (green)")
    fig.tight_layout()
    fig.savefig(out / f"mcl_{v}.png", dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("prepare", "generate", "evaluate", "run"))
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--adapter", default="hf-melody", choices=("hf-melody", "wosdd-all"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond", default="lead", choices=("lead", "full"),
                    help="melody control from the data lead alone, or lead + arranged harmony/bands (6 dB under)")
    ap.add_argument("--max-seconds", type=float, default=None, help="CPU smoke test: generate only this much")
    a = ap.parse_args(argv)
    for tid in a.ids:
        if a.cmd in ("prepare", "run"):
            print(json.dumps(A.json_safe(prepare(tid, a.src)))[:600], flush=True)
        if a.cmd in ("generate", "run"):
            generate(tid, a.adapter, a.device, a.steps, a.guidance, a.seed, a.dtype, a.max_seconds, cond=a.cond)
        if a.cmd in ("evaluate", "run"):
            rep = evaluate(tid)
            for v, r in rep["variants"].items():
                ra, rc = r["A"]["real"], r["A"].get("control", {})
                print(f"{tid} {v} A: R2={ra.get('r2')} within½={ra.get('pitch_within_half_semitone')} "
                      f"control R2={rc.get('r2')}  B: " + " ".join(
                          f"{h}={b['mix'].get('r2')}" for h, b in r["B"].items()), flush=True)


if __name__ == "__main__":
    main()
