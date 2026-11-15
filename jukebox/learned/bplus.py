"""MuseControlLite B+: a drum-free MCL accompaniment UNDER the deterministic data lead, bands and Earth voices.

B+ is the pilot's mode B "augment" without cymbals: the user rated augment best on 5313 and liked its bass, but
heard the wosdd-all cymbals/drums "break the flow" on 5209, and "replace" lost the recorded voices. Two families
ship per track so the user can choose on the Compare page:

  soft   "B+ soft rhythm"   bass + soft low drums allowed, no cymbals / hi-hats / crash / ride
  nop    "B+ no percussion" bass, no drums at all

  generation  wosdd-all adapters, control CFG 1.5, text CFG 7, 50 steps, seed 42, lead-only melody control
              (never chord tones). Percussion is removed in steps, each measured (VARIANTS):
                (a) prompt wording ("no cymbals, no hi-hats, no crash, no ride", and for nop "no drums,
                    no percussion"); the bass is asked for ("warm bass line") so it survives
                (b) a negative prompt (the pipeline's text-CFG unconditional branch)
                (c) the rhythm control weakened (x0.5 / x0.3) or off (melody + dynamics only; zero is the
                    adapters' trained "no condition")
              Fallback (d), disclosed per track: attenuate the HPSS percussive component above 5 kHz in the
              MCL stem only (never the data lead or the recorded voices).
  seam        tracks longer than one 47 s window use full windows with the largest even overlap and one 8 s
              equal-power crossfade per seam, placed where the two takes are spectrally closest
              (level-matched over the overlap). Measured as a log-mel jump at the seam vs the track's median.
  mix         every arranged stem stays at its arranged (already ducked) level: bands, events and all recorded
              voice beds. The MCL accompaniment takes the AMT harmony's slot ("replace": same target level vs
              the lead) or shares it ("blend": both 3 dB down). It is ducked by arrange's sidechain with the
              track's own margin (8/9 dB; 5058 14/14 dB) computed on the whole accompaniment bus; then
              -16 LUFS and arrange's limiter. Controls: lead swap and end to end, through the identical chain.

Rendering is split into units (variant x real/control x 47 s window) so that CPU jobs (one per unit) and a
GPU job (all units) can race; a lock file per unit keeps them from duplicating work:

    python -m jukebox.learned.bplus prepare <id>            # mcl env (torchaudio): inputs + controls
    python -m jukebox.learned.bplus units <id>              # list the units "variant which window"
    python -m jukebox.learned.bplus render <id> --unit v w k --device cpu     # mcl env: one unit
    python -m jukebox.learned.bplus generate <id>           # mcl env, GPU: every unit not done
    python -m jukebox.learned.bplus stitch <id>             # crossfade the windows -> gen_<v>_<which>.wav
    python -m jukebox.learned.bplus evaluate|mix|export <id>   # jukebox env, CPU

Outputs: outputs/<id>_mclbplus/ (git-ignored); demo/<id>_mclbplus/ (small). Driver: scripts/bplus_one.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np

from ..render import arrange as A
from . import musecontrol as M

SR = M.SR
ROOT = M.ROOT
ADAPTER, GUIDANCE, TEXT_CFG, STEPS, SEED = "wosdd-all", 1.5, 7.0, 50, 42
IDS = ("5209", "5313", "3912", "4950", "5176", "5011", "5058", "5669")
DESC = {   # per-track wording; the bass is asked for explicitly so "no percussion" does not take it away
    "5209": "Solemn orchestral strings and piano, slow cinematic build, warm bass line",
    "5313": "Night ambient, warm synth pads and soft piano, glowing, warm bass line",
    "3912": "Oceanic ambient, flowing strings and soft pads, gentle, warm bass line",
    "4950": "Forest ambient, woodwinds and warm strings, organic, warm bass line",
    "5176": "Ocean ambient, soft pads, slow and calm, warm bass line",
    "5011": "Stormy ambient, dark strings and low synth pads, slow and brooding, warm bass line",
    "5058": "Somber ambient in G major, slow strings and warm pads, fading light, warm bass line",
    "5669": "Airy ambient, drifting pads and soft strings, hazy and slow, warm bass line",
}
NO_CYM = "no cymbals, no hi-hats, no crash, no ride"
TAIL = {"soft": f"soft low drums, {NO_CYM}", "nop": f"no drums, no percussion, {NO_CYM}", "cym": NO_CYM}
NEG = {"soft": "cymbals, hi-hats, crash cymbal, ride cymbal, splash cymbal, shaker, tambourine, bright snare",
       "nop": "drums, drum kit, percussion, kick drum, snare, cymbals, hi-hats, crash cymbal, ride cymbal, shaker, claps"}


def V(prompt, neg, rhythm, melody=1.0, cond=""):
    return {"prompt": prompt, "neg": neg, "rhythm": rhythm, "melody": melody, "cond": cond}


# prompt kind ("pilot" = musecontrol.TRACKS prompt), negative-prompt kind, rhythm scale, melody scale, cond suffix
VARIANTS = {
    "base": V("pilot", None, 1.0),        # the pilot recipe (what "old augment" used), new seam
    "nop_p": V("nop", None, 1.0),         # (a) prompt wording only
    "nop_pn": V("nop", "nop", 1.0),       # (a) + (b) negative prompt
    "nop_r03": V("nop", "nop", 0.3),      # + rhythm control weakened
    "nop_r0": V("nop", "nop", 0.0),       # + rhythm control off (melody + dynamics only)
    "soft_pn": V("soft", "soft", 1.0),    # soft rhythm: cymbal-free prompt + negative prompt, rhythm on
    "soft_r05": V("soft", "soft", 0.5),   # soft rhythm, rhythm control halved
    # 5058: 0.12 s trumpet notes make a poor melody control -> coarse (0.5 s) melody, or none (dynamics + key)
    "nop_coarse": V("nop", "nop", 0.0, cond="_coarse"),
    "nop_nomel": V("nop", "nop", 0.0, melody=0.0),
    "soft_coarse": V("soft", "soft", 0.5, cond="_coarse"),
    "soft_nomel": V("soft", "soft", 0.5, melody=0.0),
    # 5669: melody control from the dominant aerosol voice per time step, or dynamics only
    "nop_dom": V("nop", "nop", 0.0),
    "soft_dom": V("soft", "soft", 0.5),
}
# The shipped single variant (deadline triage): bass kept, no cymbals / hi-hats, rhythm control halved.
VARIANTS["bplus"] = V("cym", "soft", 0.5)
VARIANTS["bplus_coarse"] = V("cym", "soft", 0.5, cond="_coarse")     # 5058: coarse 0.5 s melody control
VARIANTS["bplus_nomel"] = V("cym", "soft", 0.5, melody=0.0)          # 5058: no melody control (dynamics + key)
DEFAULT_VARIANTS = ["bplus"]
TRACK_VARIANTS = {"5058": ["bplus_coarse", "bplus_nomel"]}
# 3912, 5209, 5313 are not regenerated: the pilot's wosdd-all CFG 1.5 take (rated as mode B augment) is reused
REUSE_PILOT = {"5209", "5313", "3912"}
LEADLESS = {"5669"}                     # no data lead: the four aerosol instruments act as the lead
FAMILIES = ("soft", "nop")
SEAM_XF = 8.0                           # crossfade length
EDGE_HEAD, EDGE_TAIL = 2.0, 3.0         # keep the crossfade off a window's first 2 s and last 3 s
DUCK_GROUPS = ("bands", "harmony", "geophony", "anthrophony", "biophony")
GEN_GROUP = "mcl"
PILOT = Path(os.environ.get("MCL_PILOT", "/projects/e32706/omb8654/jukebox-wt/musecontrol/outputs"))
DEMO = ROOT / "demo"


def out_dir(tid: str) -> Path:
    return Path(os.environ.get("MCL_OUTROOT", ROOT / "outputs")) / f"{tid}_mclbplus"


def family(v: str) -> str | None:
    f = v.split("_")[0]
    return f if f in FAMILIES else None


def variants_for(tid: str) -> list[str]:
    return TRACK_VARIANTS.get(tid, DEFAULT_VARIANTS)


def which_for(tid: str) -> tuple[str, ...]:
    return ("real",) if tid in LEADLESS else ("real", "control")


def prompt_for(tid: str, variant: str) -> tuple[str, str]:
    s = VARIANTS[variant.split("_seed")[0]]
    prompt = M.TRACKS[tid]["prompt"] if s["prompt"] == "pilot" else f"{DESC[tid]}, {TAIL[s['prompt']]}"
    return prompt, (NEG[s["neg"]] if s["neg"] else "")


# ---------------------------------------------------------------- segments and seam
MIN_OVERLAP = EDGE_HEAD + SEAM_XF + EDGE_TAIL + 2.0


def plan_segments(total: float, max_seg: float = M.MAX_SEG) -> list[tuple[float, float]]:
    """One window if it fits; otherwise n full windows, evenly spaced, so each overlap is as long as possible
    (the pilot used 2 x 36 s with a fixed 4 s crossfade at the window edge)."""
    if total <= max_seg:
        return [(0.0, total)]
    n = int(np.ceil((total - max_seg) / (max_seg - MIN_OVERLAP))) + 1
    step = (total - max_seg) / (n - 1)
    return [(k * step, k * step + max_seg) for k in range(n)]


def logmel(a: np.ndarray, hop: int = 1024) -> np.ndarray:
    import librosa
    y = a.mean(1) if a.ndim == 2 else a
    S = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=SR, n_fft=4096, hop_length=hop, n_mels=64,
                                       fmax=16000)
    return librosa.power_to_db(S, ref=1.0, amin=1e-10)


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-20))


def _best_centre(o0, o1, s1, e0, xf):
    L0, L1 = logmel(o0), logmel(o1)
    floor = max(L0.max(), L1.max()) - 80
    d = np.abs(np.maximum(L0, floor) - np.maximum(L1, floor)).mean(0)        # dB per frame
    ft = s1 + np.arange(len(d)) * 1024 / SR
    cands = np.arange(s1 + EDGE_HEAD + xf / 2, e0 - EDGE_TAIL - xf / 2 + 1e-9, 0.25)
    cost = np.array([d[(ft >= c - xf / 2) & (ft < c + xf / 2)].mean() for c in cands])
    return float(cands[int(np.argmin(cost))]), float(cost.min()), float(np.median(cost))


def stitch_best(parts: list[np.ndarray], segs, total: float, xf: float = SEAM_XF) -> tuple[np.ndarray, dict]:
    """Crossfade n takes: each seam at the spectrally closest point of its overlap, each take level-matched to
    the previous one over their overlap."""
    N = int(round(total * SR))
    out = np.zeros((N, 2), np.float32)
    if len(parts) == 1:
        a = parts[0][:N]
        out[: len(a)] = a
        return out, {"windows": 1}
    parts = [p.astype(np.float32) for p in parts]
    seams = []
    for k in range(len(parts) - 1):
        (s0, e0), (s1, e1) = segs[k], segs[k + 1]
        i1, ie0 = int(round((s1 - s0) * SR)), int(round((e0 - s1) * SR))
        o0, o1 = parts[k][i1:i1 + ie0], parts[k + 1][:ie0]
        g = float(np.clip(_rms(o0) / _rms(o1), 10 ** (-6 / 20), 10 ** (6 / 20)))
        parts[k + 1] = parts[k + 1] * np.float32(g)
        c, cmin, cmed = _best_centre(o0, o1 * g, s1, e0, xf)
        seams.append({"overlap_s": [s1, e0], "centre_s": c, "cost_db": cmin, "cost_db_median": cmed,
                      "take_gain_db": A.db(g)})
    t = np.arange(N) / SR
    for k, (p, (s, e)) in enumerate(zip(parts, segs)):
        w = np.ones(N)
        if k > 0:
            c = seams[k - 1]["centre_s"]
            w *= np.sin(0.5 * np.pi * np.clip((t - (c - xf / 2)) / xf, 0, 1))
        if k < len(parts) - 1:
            c = seams[k]["centre_s"]
            w *= np.cos(0.5 * np.pi * np.clip((t - (c - xf / 2)) / xf, 0, 1))
        i0 = int(round(s * SR))
        n = min(len(p), N - i0)
        out[i0:i0 + n] += p[:n] * w[i0:i0 + n, None].astype(np.float32)
    return out, {"windows": len(parts), "segments": [list(s) for s in segs], "crossfade_s": xf, "seams": seams,
                 "centre_s": seams[0]["centre_s"]}


def disc_curve(audio: np.ndarray, scale: float, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Spectral discontinuity: RMS (over 64 mel bands) of the change in mean log-mel level between the
    `scale` seconds before and after each frame, in dB."""
    L = logmel(audio, hop)
    L = np.maximum(L, L.max() - 80)
    k = max(1, int(round(scale * SR / hop)))
    cs = np.concatenate([np.zeros((L.shape[0], 1)), np.cumsum(L, 1)], 1)
    j = np.arange(k, L.shape[1] - k)
    before = (cs[:, j] - cs[:, j - k]) / k
    after = (cs[:, j + k] - cs[:, j]) / k
    return j * hop / SR, np.sqrt(((after - before) ** 2).mean(0))


def seam_report(audio: np.ndarray, centre: float, half: float = 2.0) -> dict:
    """Largest jump within +-half s of the seam centre, against the whole track's distribution."""
    r = {"centre_s": centre}
    for scale in (0.5, 2.0):
        t, D = disc_curve(audio, scale)
        m = (t >= centre - half) & (t <= centre + half)
        seam = float(D[m].max())
        r[f"scale_{scale:g}s"] = {"seam_db": seam, "median_db": float(np.median(D)), "p95_db": float(np.percentile(D, 95)),
                                 "seam_over_median": seam / float(np.median(D)), "percentile": float((D < seam).mean() * 100)}
    return r


# ---------------------------------------------------------------- percussiveness
def percussiveness(audio: np.ndarray) -> dict:
    """Mono 44.1 kHz, normalised to -23 LUFS, STFT 4096 / hop 1024, HPSS with margin 2. Energies are
    A-weighted (raw energy is dominated by the bass, which would hide cymbals entirely).
    perc_ratio       percussive / total energy, whole band
    hf_perc_share    percussive energy in 5-16 kHz / total energy (cymbals, hi-hats, crash, ride)
    hf_share         all 5-16 kHz energy / total energy (cymbal wash is noisy, so HPSS splits it)
    hf_perc_of_hf    percussive share of the 5-16 kHz energy
    low_db           40-250 Hz level (unweighted) in dB at -23 LUFS: the bass; low_share: its unweighted share
    hat_hits_per_s   peaks of the 5-16 kHz log-mel flux envelope > 4 MAD above its local mean
    kick_flux_db     mean positive log-mel flux in 30-150 Hz (dB per 23 ms frame)"""
    import librosa
    L = A.lufs(audio)
    y = (audio.mean(1) * np.float32(10 ** ((-23 - L) / 20))).astype(np.float32)
    S = librosa.stft(y, n_fft=4096, hop_length=1024)
    _, P = librosa.decompose.hpss(S, margin=2.0)
    E, EP = np.abs(S) ** 2, np.abs(P) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=4096)
    w = (10 ** (librosa.A_weighting(np.maximum(f, 1.0)) / 10))[:, None]
    hf, lo = (f >= 5000) & (f <= 16000), (f >= 40) & (f <= 250)
    tot = max((E * w).sum(), 1e-20)
    mel = librosa.feature.melspectrogram(S=E, sr=SR, n_mels=128, fmin=20, fmax=16000)
    mf = librosa.mel_frequencies(n_mels=128, fmin=20, fmax=16000)
    mdb = librosa.power_to_db(mel, ref=1.0, amin=1e-10)
    mdb = np.maximum(mdb, mdb.max() - 70)
    flux = np.maximum(0, np.diff(mdb, axis=1))
    env = flux[mf >= 5000].mean(0)
    mad = float(np.median(np.abs(env - np.median(env)))) + 1e-9
    pk = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=20, post_avg=20, delta=4 * mad, wait=4)
    return {"perc_ratio": float((EP * w).sum() / tot), "hf_perc_share": float((EP * w)[hf].sum() / tot),
            "hf_share": float((E * w)[hf].sum() / tot),
            "hf_perc_of_hf": float(EP[hf].sum() / max(E[hf].sum(), 1e-20)),
            "low_share": float(E[lo].sum() / max(E.sum(), 1e-20)),
            "low_db": A.db(np.sqrt(E[lo].sum() / E.shape[1])),
            "hat_hits_per_s": float(len(pk) / (len(y) / SR)),
            "kick_flux_db": float(flux[(mf >= 30) & (mf <= 150)].mean())}


def decymbal(audio: np.ndarray, atten_db: float = 12.0, f_lo: float = 5000.0) -> np.ndarray:
    """Fallback (d) for cymbals the prompt could not remove. MCL stem ONLY, never the lead or the recorded voices.
    Per channel, HPSS with margin 1 (so harmonic + percussive = the whole signal); the percussive component is
    turned down by atten_db above f_lo (1 kHz linear ramp below it). The harmonic part and everything under
    4 kHz are untouched."""
    import librosa
    f = librosa.fft_frequencies(sr=SR, n_fft=4096)
    ramp = np.clip((f - (f_lo - 1000)) / 1000, 0, 1)[:, None]
    g = 1 - ramp * (1 - 10 ** (-atten_db / 20))
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        S = librosa.stft(audio[:, ch], n_fft=4096, hop_length=1024)
        H, P = librosa.decompose.hpss(S, margin=1.0)
        out[:, ch] = librosa.istft(H + P * g, hop_length=1024, length=len(audio))
    return out.astype(np.float32)


# ---------------------------------------------------------------- prepare (mcl env)
def _write_float(path: Path, a: np.ndarray):
    import soundfile as sf
    sf.write(str(path), a.astype(np.float32), SR, subtype="FLOAT")


def prepare(tid: str, src: Path = M.DEFAULT_SRC) -> dict:
    out = out_dir(tid)
    out.mkdir(parents=True, exist_ok=True)
    s = src / tid
    track = json.loads((s / "track.json").read_text())
    for f in [st["file"] for st in track["stems"]] + ["mix.wav", "mix_control.wav", "control_shuffled.wav"]:
        if (s / f).exists() and not (out / f).exists():
            shutil.copy2(s / f, out / f)
    if tid in LEADLESS and not (out / "melody.wav").exists():
        # no lead stem: the summed instrument stems stand in for it (total length, dynamics, ducking)
        _write_float(out / "melody.wav", sum(M.read_wav(out / st["file"]) for st in track["stems"] if st["group"] == "bands"))
    meta = M.prepare(tid, src, out=out, plot=False)
    notes = A.load_notes(out / "timeline.json")
    arrays = dict(np.load(out / "controls_mcl.npz"))
    total = meta["total_s"]
    extra = {}
    if tid == "5058":
        # coarse melody: 0.5 s frames, each holding the loudest city note that starts in it (legato)
        city = sorted([n for n in notes if n.voice == "city"], key=lambda n: n.start)
        def coarse(ns):
            byf = {}
            for n in ns:
                f = int(n.start // 0.5)
                if f not in byf or n.velocity > byf[f].velocity:
                    byf[f] = n
            return [dc_replace(n, start=f * 0.5) for f, n in sorted(byf.items())]
        cc = coarse(city)
        perm = np.random.default_rng(0).permutation(len(city))
        ctrl = [dc_replace(n, pitch=city[j].pitch, velocity=city[j].velocity) for n, j in zip(city, perm)]
        A.write_wav(out / "cond_real_coarse.wav", M.legato_tone(cc, total))
        A.write_wav(out / "cond_control_coarse.wav", M.legato_tone(coarse(ctrl), total))
        extra["coarse_melody"] = {"frame_s": 0.5, "notes": len(cc), "from": len(city)}
    if tid in LEADLESS:
        # melody: the note with the largest data value at each 0.25 s sweep step (the dominant aerosol voice);
        # dynamics: the summed category fractions per step, mapped to -24..0 dB, upstream's smoothing
        from scipy.signal import savgol_filter
        steps = {}
        for n in notes:
            k = round(n.start / 0.25)
            steps.setdefault(k, []).append(n)
        dom = [max(v, key=lambda n: n.value or 0) for k, v in sorted(steps.items())]
        perm = np.random.default_rng(0).permutation(len(dom))
        domc = [dc_replace(n, pitch=dom[j].pitch) for n, j in zip(dom, perm)]
        A.write_wav(out / "cond_real.wav", M.legato_tone(dom, total))
        A.write_wav(out / "cond_control.wav", M.legato_tone(domc, total))
        ts = np.array([k * 0.25 for k in sorted(steps)])
        fr = np.array([sum(n.value or 0 for n in steps[k]) for k in sorted(steps)])
        ft = np.arange(len(arrays["dynamics"])) * 160 / SR
        d = -24 * (1 - np.interp(ft, ts, fr / max(fr.max(), 1e-9)))
        arrays["dynamics"] = savgol_filter(d, 279, 1).astype(np.float32)
        np.savez_compressed(out / "controls_mcl.npz", **arrays)
        extra["dominant_melody"] = {"notes": len(dom), "voices": sorted({n.voice for n in dom})}
        extra["dynamics_override"] = "summed aerosol fractions per 0.25 s step, -24..0 dB"
    meta["segments_bplus"] = plan_segments(total)
    meta["variants_bplus"] = variants_for(tid)
    meta["prompts_bplus"] = {v: dict(zip(("prompt", "negative_prompt"), prompt_for(tid, v))) for v in variants_for(tid)}
    meta["bplus_extra"] = extra
    (out / "controls_mcl.json").write_text(json.dumps(A.json_safe(meta), indent=1))
    return meta


# ---------------------------------------------------------------- render units (mcl env)
def units(tid: str) -> list[tuple[str, str, int]]:
    meta = json.loads((out_dir(tid) / "controls_mcl.json").read_text())
    return [(v, w, k) for v in variants_for(tid) for w in which_for(tid) for k in range(len(meta["segments_bplus"]))]


def part_path(out: Path, v: str, w: str, k: int) -> Path:
    return out / f"part_{v}_{w}_{k}.npy"


def _job_alive(jid: str) -> bool:
    if not jid or jid == "local":
        return False
    try:
        r = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T"], capture_output=True, text=True, timeout=60)
        return r.stdout.strip() in ("RUNNING", "COMPLETING", "CONFIGURING")
    except Exception:
        return True       # cannot tell: assume alive


def take_lock(out: Path, v: str, w: str, k: int, device: str) -> bool:
    """True if this process may render the unit. GPU takes over CPU locks (it is ~100x faster); CPU respects
    any live lock. A lock whose job is gone is stale."""
    d = out / "locks"
    d.mkdir(exist_ok=True)
    p = d / f"{v}_{w}_{k}"
    me = f"{os.environ.get('SLURM_JOB_ID', 'local')} {'gpu' if device.startswith('cuda') else 'cpu'}"
    if p.exists():
        jid, dev = (p.read_text().split() + ["", ""])[:2]
        if jid == me.split()[0]:
            return True
        if _job_alive(jid) and not (device.startswith("cuda") and dev == "cpu"):
            return False
        p.unlink(missing_ok=True)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, me.encode())
    os.close(fd)
    return True


def _condition(pipe, extractors, seg: dict, device: str, rhythm_scale: float, melody_scale: float):
    """musecontrol.build_condition for wosdd-all, with the rhythm / melody branches scaled
    (0 = the trained 'no condition', as in the adapters' condition dropout)."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        def ex(k, arr):
            y = extractors[k](torch.from_numpy(arr).float().unsqueeze(0).to(device))
            return F.interpolate(y, size=1024, mode="linear", align_corners=False)
        x = torch.cat([ex("rhythm", seg["rhythm"].T) * rhythm_scale, ex("dynamics", seg["dynamics"][None]),
                       ex("melody", seg["melody_onehot"]) * melody_scale, torch.zeros(1, 192, 1024, device=device)], 1)
        assert x.shape[1] == 768, x.shape
        z = torch.zeros_like(x)
        dtype = next(pipe.transformer.parameters()).dtype
        return torch.cat([z, z, x], 0).transpose(1, 2).to(dtype)


class Renderer:
    def __init__(self, device: str):
        import torch
        self.device = device
        if device == "cpu":
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
        t0 = time.time()
        self.pipe, self.ex = M.load_pipeline(ADAPTER, device, "fp16" if device.startswith("cuda") else "fp32")
        self.load_s = time.time() - t0
        self.name = torch.cuda.get_device_name(0) if device.startswith("cuda") else f"CPU x{torch.get_num_threads()}"

    def unit(self, tid: str, v: str, w: str, k: int, seed: int = SEED, force: bool = False) -> bool:
        import torch
        out = out_dir(tid)
        pp = part_path(out, v, w, k)
        if pp.exists() and not force:
            return False
        if not take_lock(out, v, w, k, self.device):
            print(f"{tid} {v} {w} {k}: locked by another job", flush=True)
            return False
        meta = json.loads((out / "controls_mcl.json").read_text())
        arrays = dict(np.load(out / "controls_mcl.npz"))
        a, b = meta["segments_bplus"][k]
        spec = VARIANTS[v]
        prompt, neg = prompt_for(tid, v)
        cond = M.read_wav(out / f"cond_{w}{spec['cond']}.wav")
        t1 = time.time()
        c = _condition(self.pipe, self.ex, M.seg_inputs(ADAPTER, arrays, w, k, cond, a, b), self.device,
                       spec["rhythm"], spec["melody"])
        with torch.no_grad():
            wav = self.pipe(extracted_condition=c, prompt=prompt, negative_prompt=neg, num_inference_steps=STEPS,
                            guidance_scale_text=TEXT_CFG, guidance_scale_con=GUIDANCE, num_waveforms_per_prompt=1,
                            audio_end_in_s=float(b - a), generator=torch.Generator().manual_seed(seed)).audios
        p = wav[0].T.float().cpu().numpy().astype(np.float16)
        tmp = pp.with_suffix(f".tmp{os.getpid()}.npy")
        np.save(tmp, p)
        os.replace(tmp, pp)
        info = {"variant": v, "which": w, "window": k, "segment": [a, b], "prompt": prompt, "negative_prompt": neg,
                "rhythm_scale": spec["rhythm"], "melody_scale": spec["melody"], "condition": f"cond_{w}{spec['cond']}.wav",
                "adapter": ADAPTER, "guidance_con": GUIDANCE, "guidance_text": TEXT_CFG, "steps": STEPS, "seed": seed,
                "device": self.name, "job": os.environ.get("SLURM_JOB_ID", "local"),
                "gen_seconds": round(time.time() - t1, 1), "load_seconds": round(self.load_s, 1)}
        pp.with_suffix(".json").write_text(json.dumps(A.json_safe(info), indent=1))
        print(json.dumps(A.json_safe(info)), flush=True)
        return True


def render(tid: str, v: str, w: str, k: int, device: str = "cpu") -> None:
    out = out_dir(tid)
    if part_path(out, v, w, k).exists():
        print(f"{tid} {v} {w} {k}: already rendered")
        return
    Renderer(device).unit(tid, v, w, k)


def generate(tid: str, device: str = "cuda") -> None:
    """Every unit of the track that is not done (GPU path)."""
    r = Renderer(device)
    for v, w, k in units(tid):
        r.unit(tid, v, w, k)


def missing_units(tid: str) -> list:
    out = out_dir(tid)
    return [u for u in units(tid) if not part_path(out, *u).exists()]


def stitch(tid: str) -> None:
    """Crossfade each variant's windows into gen_<v>_<which>.wav and record the run (gen_runs.json)."""
    out = out_dir(tid)
    meta = json.loads((out / "controls_mcl.json").read_text())
    segs = [tuple(s) for s in meta["segments_bplus"]]
    runs = []
    for v in variants_for(tid):
        for w in which_for(tid):
            pp = [part_path(out, v, w, k) for k in range(len(segs))]
            if not all(p.exists() for p in pp):
                print(f"{tid} {v} {w}: missing windows, skipped", flush=True)
                continue
            parts = [np.load(p).astype(np.float32) for p in pp]
            infos = [json.loads(p.with_suffix(".json").read_text()) for p in pp]
            audio, seam = stitch_best(parts, segs, meta["total_s"])
            audio *= np.float32(0.89 / max(1e-9, np.abs(audio).max()))
            A.write_wav(out / f"gen_{v}_{w}.wav", audio)
            i0 = infos[0]
            runs.append({**{k_: i0[k_] for k_ in ("prompt", "negative_prompt", "rhythm_scale", "melody_scale", "condition",
                                                  "adapter", "guidance_con", "guidance_text", "steps", "seed")},
                         "variant": v, "which": w, "segments": segs, "seam": seam, "audio_s": meta["total_s"],
                         "device": sorted({i["device"] for i in infos}), "jobs": [i["job"] for i in infos],
                         "gen_seconds": round(sum(i["gen_seconds"] for i in infos), 1)})
    (out / "gen_runs.json").write_text(json.dumps(A.json_safe(runs), indent=1))
    print(f"{tid}: stitched {len(runs)} runs", flush=True)


# ---------------------------------------------------------------- B+ post-processing (jukebox env, CPU)
DECYMBAL_DB = 15.0


def pilot_dir(tid: str) -> Path:
    return PILOT / f"{tid}_mcl"


def chosen_variant(tid: str) -> str:
    """BPLUS_VARIANT overrides; else outputs/<id>_mclbplus/variant.txt (written when a choice was made); else the first."""
    p = out_dir(tid) / "variant.txt"
    return os.environ.get("BPLUS_VARIANT") or (p.read_text().strip() if p.exists() else variants_for(tid)[0])


def raw_gen(tid: str, which: str) -> tuple[np.ndarray | None, str]:
    """The MCL take before cleanup: the pilot's wosdd-all CFG 1.5 seed 42 render (3912, 5209, 5313) or the
    B+ render stitched here."""
    if tid in REUSE_PILOT:
        p = pilot_dir(tid) / f"gen_wosdd-all_g1.5_{which}.wav"
    else:
        p = out_dir(tid) / f"gen_{chosen_variant(tid)}_{which}.wav"
    return (M.read_wav(p) if p.exists() else None), str(p)


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    return np.pad(a[:n], ((0, max(0, n - len(a))), (0, 0)))


def arranged_stems(out: Path, track: dict, n: int, tid: str) -> dict:
    skip = ("melody", "bands") if tid in LEADLESS else ("melody",)
    return {s["name"]: (s["group"], _fit(M.read_wav(out / s["file"]), n)) for s in track["stems"]
            if s["group"] not in skip and (out / s["file"]).exists()}


def duck_params(out: Path) -> tuple[float, float]:
    d = json.loads((out / "arrange_report.json").read_text())["loudness"].get("ducking") or {}
    return float(d.get("margin_db", 8.0)), float(d.get("max_db", 9.0))


def gen_target_db(out: Path, tid: str) -> float:
    if tid in LEADLESS:
        return -15.0                     # under the four instruments (the beds sit -12 dB)
    lv = json.loads((out / "arrange_report.json").read_text())["levels"]
    return float((lv.get("harmony") or {}).get("target_db_vs_melody", A.DEFAULT_GAINS_DB["harmony"]))


def bplus_stems(out: Path, track: dict, tid: str, lead: np.ndarray, gen: np.ndarray, how: str) -> tuple[dict, dict]:
    """Arranged stems as they are + the MCL accompaniment in the harmony slot ('replace') or sharing it ('blend')."""
    n = len(lead)
    stems = arranged_stems(out, track, n, tid)
    tgt = gen_target_db(out, tid)
    has_h = any(g == "harmony" for g, _ in stems.values())
    if how == "replace":
        stems = {k: v for k, v in stems.items() if v[0] != "harmony"}
        gen_db = tgt
    else:
        stems = {k: ((g, a * np.float32(10 ** (-3 / 20))) if g == "harmony" else (g, a)) for k, (g, a) in stems.items()}
        gen_db = tgt - 3.0 if has_h else tgt
    g = _fit(gen, n)
    g = g * np.float32(A.active_rms(lead) * 10 ** (gen_db / 20) / max(1e-9, A.active_rms(g)))
    bus = sum((a for k, (grp, a) in stems.items() if grp in DUCK_GROUPS), np.zeros_like(g)) + g
    margin, mx = duck_params(out)
    gd = A.duck_gain(lead, bus, margin, mx)
    stems[GEN_GROUP] = (GEN_GROUP, g * gd[:, None])
    return stems, {"how": how, "mcl_db_vs_lead_before_duck": gen_db, "duck_margin_db": margin, "duck_max_db": mx,
                   "mcl_duck": {"mean_gr_db": float(-20 * np.log10(np.mean(gd))), "max_gr_db": float(-20 * np.log10(gd.min()))}}


def pilot_augment_stems(out: Path, track: dict, tid: str, lead: np.ndarray, gen: np.ndarray) -> dict:
    """The pilot's mode B augment (musecontrol.mix_b): every stem + generated at -13 dB, whole bus re-ducked."""
    n = len(lead)
    stems = arranged_stems(out, track, n, tid)
    g = _fit(gen, n)
    stems[GEN_GROUP] = (GEN_GROUP, g * np.float32(A.active_rms(lead) * 10 ** (-13 / 20) / max(1e-9, A.active_rms(g))))
    acc = [k for k, (grp, _) in stems.items() if grp in DUCK_GROUPS + (GEN_GROUP,)]
    gd = A.duck_gain(lead, sum(stems[k][1] for k in acc), 8.0, 9.0)
    return {k: ((grp, a * gd[:, None]) if k in acc else (grp, a)) for k, (grp, a) in stems.items()}


def finish(lead: np.ndarray, stems: dict) -> tuple[np.ndarray, float, dict]:
    """musecontrol.mix_b's loudness rule: -16 LUFS, limiter drive capped at 4.5 dB, then arrange's limiter."""
    pre = lead + sum(a for _, a in stems.values())
    k = 10 ** ((-16.0 - A.lufs(pre)) / 20)
    k = min(k, 10 ** (4.5 / 20) * 10 ** (-1 / 20) / max(float(np.abs(pre).max()), 1e-9))
    mix, gr = A.limiter((pre * np.float32(k)).astype(np.float32))
    return mix, k, {"lufs": A.lufs(mix), "limiter_gr_db": gr, "norm_gain_db": A.db(k)}


def energy_shares(lead: np.ndarray, stems: dict) -> dict:
    """K-weighted energy share of each group in the pre-limiter mix (lead + stems)."""
    def e(x):
        return float((A.k_weight(x.astype(np.float64)) ** 2).sum())
    pre = lead + sum(a for _, a in stems.values())
    tot = e(pre)
    sh = {"lead": e(lead) / tot}
    for grp in sorted({g for g, _ in stems.values()}):
        sh[grp] = e(sum(a for g, a in stems.values() if g == grp)) / tot
    beds = [a for g, a in stems.values() if g in A.VOICE_GROUPS]
    sh["voice_beds"] = e(sum(beds)) / tot if beds else 0.0
    return sh


def low_band_db(x: np.ndarray) -> float:
    """40-250 Hz level of a stem as it sits in the mix (dB re full scale)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [40, 250], btype="band", fs=SR, output="sos")
    return A.db(_rms(sosfiltfilt(sos, x.mean(1))))


def chroma_agreement(a: np.ndarray, b: np.ndarray) -> float | None:
    """Energy-weighted cosine similarity of ~0.25 s chroma frames (does the MCL bed voice the AMT chords?)."""
    import librosa
    from scipy.signal import resample_poly
    def ch(x):
        return librosa.feature.chroma_cqt(y=resample_poly(x.mean(1), 1, 2).astype(np.float32), sr=SR // 2, hop_length=4096)
    n = min(len(a), len(b))
    A_, B_ = ch(a[:n]), ch(b[:n])
    w = np.minimum(A_.sum(0), B_.sum(0))
    cos = (A_ * B_).sum(0) / (np.linalg.norm(A_, axis=0) * np.linalg.norm(B_, axis=0) + 1e-9)
    return float((cos * w).sum() / w.sum()) if w.sum() > 0 else None


def hit_salience(audio: np.ndarray, hits: np.ndarray, mcl: np.ndarray | None = None) -> dict:
    """5011: are the data drum hits still the transients? Onset strength at the hits (max within +-2 frames)
    over its median elsewhere; and the MCL stem's onsets per s further than 80 ms from any data hit."""
    import librosa
    env = librosa.onset.onset_strength(y=audio.mean(1), sr=SR, hop_length=512)
    t = librosa.frames_to_time(np.arange(len(env)), sr=SR, hop_length=512)
    near = np.zeros(len(env), bool)
    at = []
    for h in hits:
        m = np.abs(t - h) <= 2 * 512 / SR
        near |= np.abs(t - h) <= 0.08
        if m.any():
            at.append(env[m].max())
    r = {"hit_salience": float(np.mean(at) / max(np.median(env[~near]), 1e-9))}
    if mcl is not None:
        on = librosa.onset.onset_detect(y=mcl.mean(1), sr=SR, hop_length=512, units="time")
        off = [o for o in on if np.min(np.abs(hits - o)) > 0.08] if len(hits) else on
        r["mcl_offhit_onsets_per_s"] = float(len(off) / (len(audio) / SR))
    return r


def type_r(audio: np.ndarray, notes) -> dict:
    """5669: per aerosol type, Pearson r between each note's data fraction and its level in the audio
    (CQT magnitude at the note's pitch, dB, over the note)."""
    import librosa
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(audio.mean(1), sr=SR, hop_length=512, fmin=librosa.midi_to_hz(24),
                                                   n_bins=96, bins_per_octave=12)), ref=np.max)
    t = librosa.frames_to_time(np.arange(C.shape[1]), sr=SR, hop_length=512)
    out = {}
    for typ in sorted({n.voice for n in notes}):
        ns = [n for n in notes if n.voice == typ and n.value is not None and 24 <= n.pitch < 120]
        lv, vals = [], []
        for n in ns:
            m = (t >= n.start + 0.02) & (t < max(n.end, n.start + 0.1))
            if m.any():
                lv.append(C[n.pitch - 24, m].mean())
                vals.append(n.value)
        out[typ] = {"n": len(lv), "r": float(np.corrcoef(vals, lv)[0, 1]) if len(lv) > 2 and np.std(lv) > 0 else None}
    return out


def mode_a(tid: str, gen: np.ndarray, genc: np.ndarray | None) -> dict | None:
    if tid in LEADLESS:
        return None
    out = out_dir(tid)
    notes, track, spec, lead, ctrl, duration = M.load_track(out, tid)
    return M.mode_a(gen, notes, spec, lead, ctrl, genc)


def postprocess(tid: str) -> dict:
    """Clean the MCL stem (decymbal), mix it under the arranged stems, measure everything."""
    out = out_dir(tid)
    notes, track, spec, lead_notes, ctrl, duration = M.load_track(out, tid)
    leadless = tid in LEADLESS
    rng = None if leadless else A.pyin_range(notes, spec)
    fid = (lambda a: None) if leadless else (lambda a: A.fidelity(a, notes, spec, track=A.pyin_track(a, *rng)))
    fidc = (lambda a: None) if leadless else (
        lambda a: A.fidelity(a, ctrl, spec, truth_notes=lead_notes, track=A.pyin_track(a, *rng)))
    lead = M.read_wav(out / "melody.wav")
    lead_c = M.read_wav(out / "control_shuffled.wav") if (out / "control_shuffled.wav").exists() else None
    g_raw, src = raw_gen(tid, "real")
    gc_raw, _ = raw_gen(tid, "control")
    g = decymbal(g_raw, DECYMBAL_DB)
    gc = decymbal(gc_raw, DECYMBAL_DB) if gc_raw is not None else None
    A.write_wav(out / "bplus_mcl_clean.wav", g)
    rep = {"id": tid, "source": src, "reused_pilot_take": tid in REUSE_PILOT,
           "decymbal": {"atten_db": DECYMBAL_DB, "above_hz": 5000, "hpss_margin": 1.0, "applied_to": "MCL stem only"},
           "bed": {"raw": percussiveness(g_raw), "clean": percussiveness(g)}, "mixes": {}, "reference": {}}
    runs = json.loads((out / "gen_runs.json").read_text()) if (out / "gen_runs.json").exists() else []
    run = next((r for r in runs if r["which"] == "real" and r["variant"] == chosen_variant(tid)), None)
    if run is None and tid in REUSE_PILOT:
        pr = json.loads((pilot_dir(tid) / "gen_runs.json").read_text())
        run = next(r for r in pr if r["variant"] == "wosdd-all_g1.5" and r["which"] == "real")
        run = {**run, "negative_prompt": "", "rhythm_scale": 1.0, "melody_scale": 1.0,
               "seam": {"windows": len(run["segments"]),
                        **({"centre_s": run["segments"][0][1] - M.XFADE / 2, "crossfade_s": M.XFADE} if len(run["segments"]) > 1 else {})}}
    rep["run"] = run
    if run and run["seam"].get("windows", 1) > 1:
        rep["seam"] = {"bplus": seam_report(g, run["seam"]["centre_s"])}
        if tid == "3912" or tid in REUSE_PILOT:
            rep["seam"]["note"] = "pilot take reused (triage): its 4 s crossfade at 32-36 s is unchanged"
    a = mode_a(tid, g, gc)
    if a:
        rep["mode_a_clean_bed"] = {"r2": a["real"].get("r2"), "within": a["real"].get("pitch_within_half_semitone"),
                                   "control_r2": (a.get("control") or {}).get("r2")}

    det = arranged_stems(out, track, len(lead), tid)
    det_mix = M.read_wav(out / "mix.wav")
    dref = {"mix": fid(det_mix), "lead_stem": fid(lead), "energy_share": energy_shares(lead, det),
            "percussion": percussiveness(det_mix), "lufs": A.lufs(det_mix)}
    if (out / "mix_control.wav").exists() and not leadless:
        dref["control_leadswap"] = fidc(M.read_wav(out / "mix_control.wav"))
    rep["reference"]["deterministic"] = dref
    if tid in REUSE_PILOT:
        pb = json.loads((pilot_dir(tid) / "mcl_report.json").read_text())["variants"]["wosdd-all_g1.5"]["B"]["augment"]
        pmix = M.read_wav(pilot_dir(tid) / "mixB_wosdd-all_g1.5_augment.wav")
        pst = pilot_augment_stems(out, track, tid, lead, g_raw)
        rep["reference"]["old_augment"] = {
            "mix": pb["mix"], "control_e2e": pb["control_e2e"], "control_leadswap": pb["control"],
            "percussion": percussiveness(pmix), "energy_share": energy_shares(lead, pst),
            "mcl_low_db_in_mix": low_band_db(pst[GEN_GROUP][1])}
    for how in ("replace", "blend"):
        stems, info = bplus_stems(out, track, tid, lead, g, how)
        m, k, li = finish(lead, stems)
        A.write_wav(out / f"bplus_{how}_mix.wav", m)
        r = {**info, **li, "mix": fid(m), "energy_share": energy_shares(lead, stems), "percussion": percussiveness(m),
             "mcl_low_db_in_mix": low_band_db(stems[GEN_GROUP][1] * np.float32(k))}
        raw_st, _ = bplus_stems(out, track, tid, lead, g_raw, how)        # same mix, cymbals not removed
        r["percussion_without_decymbal"] = percussiveness(finish(lead, raw_st)[0])
        if lead_c is not None:
            if gc is not None:
                stc, _ = bplus_stems(out, track, tid, lead_c, gc, how)       # end to end
                mc, _, _ = finish(lead_c, stc)
                A.write_wav(out / f"bplus_{how}_control.wav", mc)
                r["control_e2e"] = fidc(mc)
            ml, _, _ = finish(lead_c, stems)                                 # lead swap
            A.write_wav(out / f"bplus_{how}_control_leadswap.wav", ml)
            r["control_leadswap"] = fidc(ml)
        if how == "blend":
            h = next((x for gg, x in det.items() if x[0] == "harmony"), None)
            r["chroma_agreement_amt_vs_mcl"] = chroma_agreement(h[1], g) if h is not None else None
        if tid == "5011":
            hits = np.array(sorted(n.start for n in notes if n.voice == "flashes"))
            r["data_hits"] = hit_salience(m, hits, stems[GEN_GROUP][1])
            rep["reference"]["deterministic"]["data_hits"] = hit_salience(det_mix, hits)
        if leadless:
            r["type_r"] = type_r(m, notes)
            rep["reference"]["deterministic"]["type_r"] = type_r(det_mix, notes)
        if how == "replace":
            A.write_wav(out / "bplus_accompaniment.wav", stems[GEN_GROUP][1] * np.float32(k))
        rep["mixes"][how] = r
        mr = (r["mix"] or {}).get("r2")
        print(f"{tid} {how}: R2={mr} ctrl={(r.get('control_e2e') or {}).get('r2')} beds={r['energy_share']['voice_beds']:.3f} "
              f"(det {dref['energy_share']['voice_beds']:.3f}) hf_perc={r['percussion']['hf_perc_share']:.5f} "
              f"(no decymbal {r['percussion_without_decymbal']['hf_perc_share']:.5f})", flush=True)
    rep["choice"] = choose_mix(rep)
    rep["variant"] = None if tid in REUSE_PILOT else chosen_variant(tid)
    (out / "bplus_report.json").write_text(json.dumps(A.json_safe(rep), indent=1))
    if rep["variant"]:
        (out / f"bplus_report_{rep['variant']}.json").write_text(json.dumps(A.json_safe(rep), indent=1))
    return rep


def choose_mix(rep: dict) -> dict:
    """Blend (AMT harmony and MCL both 3 dB down) or replace (MCL in the AMT slot)? The AMT harmony stem carries
    the deterministic bass line, so replace thins the low end. Blend wins when it keeps R² (within 0.002) and the
    voice-bed share (within 0.005), and its 40-250 Hz level is closer to the old augment's (the version whose bass
    the user liked; the deterministic mix where there is no old augment)."""
    ref = rep["reference"].get("old_augment") or rep["reference"]["deterministic"]
    low = ref["percussion"]["low_db"]
    d = rep["reference"]["deterministic"]["energy_share"]["voice_beds"]
    rp, bl = rep["mixes"]["replace"], rep["mixes"]["blend"]
    r2 = lambda r: (r["mix"] or {}).get("r2") or 0.0
    blend_ok = (r2(bl) >= r2(rp) - 0.002
                and abs(bl["energy_share"]["voice_beds"] - d) <= abs(rp["energy_share"]["voice_beds"] - d) + 0.005
                and abs(bl["percussion"]["low_db"] - low) < abs(rp["percussion"]["low_db"] - low))
    return {"how": "blend" if blend_ok else "replace", "chroma_agreement": bl.get("chroma_agreement_amt_vs_mcl"),
            "low_db": {"reference": low, "replace": rp["percussion"]["low_db"], "blend": bl["percussion"]["low_db"]},
            "rule": "blend if it keeps R² (±0.002) and the voice-bed share (±0.005) and its 40-250 Hz level is closer "
                    "to the old augment's (deterministic mix where none)"}


# ---------------------------------------------------------------- export (demo/<id>_mclbplus)
def ffmpeg() -> str:
    return shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")


def mp3(src: Path, dst: Path, kbps: int = 128):
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(src), "-codec:a", "libmp3lame", "-b:a", f"{kbps}k", str(dst)],
                   check=True)


def _plot(tid: str, out: Path, dest: Path, rep: dict):
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    how = rep["choice"]["how"]
    g_raw, _ = raw_gen(tid, "real")
    panels = [("MCL take before cleanup", g_raw), ("MCL stem after cleanup (HPSS percussive > 5 kHz -15 dB)",
                                                   M.read_wav(out / "bplus_mcl_clean.wav")),
              (f"B+ mix ({how}): data lead, bands, Earth voices, MCL accompaniment", M.read_wav(out / f"bplus_{how}_mix.wav"))]
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for a, (title, y) in zip(ax, panels):
        S = librosa.power_to_db(librosa.feature.melspectrogram(y=y.mean(1), sr=SR, n_fft=2048, hop_length=512, n_mels=128,
                                                               fmax=16000), ref=np.max)
        a.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-70, extent=[0, len(y) / SR, 0, 128])
        p = percussiveness(y)
        a.set_title(f"{title}   [HF percussive share {p['hf_perc_share']:.4f}, 40-250 Hz {p['low_db']:.1f} dB]", fontsize=9)
        a.set_ylabel("mel bin (0-16 kHz)")
        sm = (rep.get("run") or {}).get("seam") or {}
        for s in sm.get("seams", []) or ([{"centre_s": sm["centre_s"]}] if "centre_s" in sm else []):
            a.axvspan(s["centre_s"] - sm.get("crossfade_s", SEAM_XF) / 2, s["centre_s"] + sm.get("crossfade_s", SEAM_XF) / 2,
                      color="cyan", alpha=0.08)
    ax[-1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(dest / "spectrogram.png", dpi=60)
    plt.close(fig)


def _f(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def _dbs(x):
    """an energy share as dB re the total"""
    return "–" if x is None else f"{10 * np.log10(max(x, 1e-12)):.1f} dB"


def export(tid: str) -> None:
    out = out_dir(tid)
    rep = json.loads((out / "bplus_report.json").read_text())
    rep["choice"] = choose_mix(rep)
    (out / "bplus_report.json").write_text(json.dumps(A.json_safe(rep), indent=1))
    dest = DEMO / f"{tid}_mclbplus"
    dest.mkdir(parents=True, exist_ok=True)
    how = rep["choice"]["how"]
    mp3(out / f"bplus_{how}_mix.wav", dest / "mix.mp3")
    if (out / f"bplus_{how}_control.wav").exists():
        mp3(out / f"bplus_{how}_control.wav", dest / "control.mp3")
    mp3(out / "bplus_accompaniment.wav", dest / "accompaniment.mp3", 96)
    _plot(tid, out, dest, rep)
    small = {k: rep[k] for k in rep if k != "mixes"}
    small["mix"] = rep["mixes"][how]
    small["other_mix"] = rep["mixes"]["blend" if how == "replace" else "replace"]
    (dest / "bplus_report.json").write_text(json.dumps(A.json_safe(small), indent=1))
    m, det = rep["mixes"][how], rep["reference"]["deterministic"]
    old = rep["reference"].get("old_augment")
    run = rep.get("run") or {}
    br, bc = rep["bed"]["raw"], rep["bed"]["clean"]
    mm, dm = m["mix"] or {}, det["mix"] or {}
    L = [f"# {tid} with MuseControlLite B+ (no cymbals, bass kept, with Earth voices)", "",
         "The data lead, the bands and every recorded Earth-voice bed stay exactly at their arranged levels; a",
         ("MuseControlLite accompaniment (with its bass) shares the AMT harmony's slot (blend: both 3 dB down, so the "
          "AMT bass line stays). **Nothing here has" if how == "blend" else
          "MuseControlLite accompaniment (with its bass) takes the AMT harmony's place (replace). **Nothing here has"),
         "been listened to**: judged by pyin, HPSS and spectrograms only. Method and table: `demo/learned/README.md`.", ""]
    if rep["reused_pilot_take"]:
        L += ["- **MCL take:** the pilot's existing render (wosdd-all, control CFG 1.5, seed 42; the mode-B augment",
              f"  bed the user rated), not regenerated. Prompt: \"{run.get('prompt', '')}\"."]
    else:
        L += [f"- **MCL take:** rendered for B+ on CPU (fp32): wosdd-all adapters on Stable Audio Open 1.0, {run.get('steps')} steps,",
              f"  text CFG {run.get('guidance_text')}, control CFG {run.get('guidance_con')}, seed {run.get('seed')}, "
              f"rhythm control x{run.get('rhythm_scale')}, melody control x{run.get('melody_scale')}"
              + (" (coarse 0.5 s melody)" if "coarse" in str(run.get("condition")) else "") + ".",
              f"- **Prompt:** \"{run.get('prompt')}\"", f"- **Negative prompt:** \"{run.get('negative_prompt')}\""]
    L += [f"- **Cymbal removal (disclosed post-processing):** HPSS (margin 1) on the MCL stem only; its percussive",
          f"  component above 5 kHz is turned down {DECYMBAL_DB:g} dB (1 kHz ramp). The data lead, bands and recorded voices",
          "  are untouched, and so is everything under 4 kHz (the bass).", ""]
    L += ["| measure | deterministic mix | old augment | B+ |", "|---|---|---|---|"]
    if mm:
        L += [f"| full-mix R² vs data | {_f(dm.get('r2'))} | {_f((old or {}).get('mix', {}).get('r2'))} | {_f(mm.get('r2'))} |",
              f"| notes within ½ semitone | {_f(dm.get('pitch_within_half_semitone'), 2)} | "
              f"{_f((old or {}).get('mix', {}).get('pitch_within_half_semitone'), 2)} | {_f(mm.get('pitch_within_half_semitone'), 2)} |",
              f"| shuffled control R² (B+: end to end) | {_f((det.get('control_leadswap') or {}).get('r2'))} (lead swap) | "
              f"{_f((old or {}).get('control_e2e', {}).get('r2'))} | {_f((m.get('control_e2e') or {}).get('r2'))} |"]
    L += [f"| voice-bed share of K-weighted energy | {_f(det['energy_share']['voice_beds'])} | "
          f"{_f((old or {}).get('energy_share', {}).get('voice_beds'))} | {_f(m['energy_share']['voice_beds'])} |",
          f"| HF (5–16 kHz) percussive energy, full mix (re total, A-weighted) | {_dbs(det['percussion']['hf_perc_share'])} | "
          f"{_dbs((old or {}).get('percussion', {}).get('hf_perc_share'))} | {_dbs(m['percussion']['hf_perc_share'])} |",
          f"| whole-mix 40–250 Hz level at -23 LUFS (bass) | {_f(det['percussion']['low_db'], 1)} dB | "
          f"{_f((old or {}).get('percussion', {}).get('low_db'), 1)} dB | {_f(m['percussion']['low_db'], 1)} dB |",
          f"| HPSS percussive share, full mix | {_f(det['percussion']['perc_ratio'])} | "
          f"{_f((old or {}).get('percussion', {}).get('perc_ratio'))} | {_f(m['percussion']['perc_ratio'])} |",
          f"| MCL stem 40–250 Hz level in the mix (dB) | – | {_f((old or {}).get('mcl_low_db_in_mix'), 1)} | {_f(m['mcl_low_db_in_mix'], 1)} |",
          "", f"MCL stem alone, before / after cleanup: HF (5–16 kHz) percussive energy {_dbs(br['hf_perc_share'])} / {_dbs(bc['hf_perc_share'])}, "
          f"all HF energy {_dbs(br['hf_share'])} / {_dbs(bc['hf_share'])}, "
          f"HPSS percussive share {_f(br['perc_ratio'])} / {_f(bc['perc_ratio'])}, 40–250 Hz {_f(br['low_db'], 1)} / {_f(bc['low_db'], 1)} dB.", ""]
    if "data_hits" in m:
        L += [f"Data drum hits (5011): onset salience at the hits {_f(m['data_hits']['hit_salience'], 2)} vs "
              f"{_f(det['data_hits']['hit_salience'], 2)} in the deterministic mix; MCL onsets away from hits "
              f"{_f(m['data_hits'].get('mcl_offhit_onsets_per_s'), 2)} per s.", ""]
    if "type_r" in m:
        L += ["Per aerosol type, r between note data fraction and its level in the mix (deterministic → B+): "
              + ", ".join(f"{t} {_f(det['type_r'][t]['r'], 2)} → {_f(m['type_r'][t]['r'], 2)}" for t in m["type_r"]) + ".", ""]
    if rep.get("seam"):
        s = rep["seam"]["bplus"]["scale_0.5s"]
        L += [f"Seam: largest 1 s log-mel jump within ±2 s of the crossfade {s['seam_db']:.2f} dB "
              f"({s['seam_over_median']:.2f} x the track's median, percentile {s['percentile']:.0f})."
              + (" " + rep["seam"]["note"] + "." if rep["seam"].get("note") else ""), ""]
    L += ["Files: `mix.mp3` (B+), `control.mp3` (shuffled lead + accompaniment generated from it, identical chain)"
          if (dest / "control.mp3").exists() else "Files: `mix.mp3` (B+)",
          "`accompaniment.mp3` (the cleaned MCL stem at its level in the mix), `spectrogram.png`, `bplus_report.json`.", "",
          "Credits: Stable Audio Open 1.0 (Stability AI Community License, \"Powered by Stability AI\"), MuseControlLite "
          "adapters and code (MIT, Tsai et al. 2025), recorded voices as in `jukebox/voices/ATTRIBUTION.md`."]
    (dest / "README.md").write_text("\n".join(L) + "\n")
    print(f"{tid}: exported {dest}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("prepare", "units", "render", "generate", "stitch", "evaluate", "mix", "export"))
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--unit", nargs=3, default=None, metavar=("VARIANT", "WHICH", "WINDOW"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    for tid in a.ids:
        if a.cmd == "prepare":
            m = prepare(tid)
            print(tid, m["segments_bplus"], m["rhythm"], m.get("bplus_extra"), flush=True)
        elif a.cmd == "units":
            for v, w, k in units(tid):
                print(tid, v, w, k)
        elif a.cmd == "render":
            render(tid, a.unit[0], a.unit[1], int(a.unit[2]), a.device)
        elif a.cmd == "generate":
            generate(tid, a.device)
        elif a.cmd == "stitch":
            stitch(tid)
        elif a.cmd in ("evaluate", "mix"):
            postprocess(tid)
        elif a.cmd == "export":
            export(tid)


if __name__ == "__main__":
    main()
