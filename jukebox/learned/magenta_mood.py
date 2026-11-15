"""Mood texture from Magenta RealTime 2, steered by the frame -> style weights, mixed UNDER the data.

    python -m jukebox.learned.mood 5313                       # weights first (SigLIP)
    python -m jukebox.learned.magenta_mood 5313 --size mrt2_small
    python -m jukebox.learned.magenta_mood 5313 --mix-only --tag base   # re-mix an existing mood_base.wav

Steps
1. Style: every anchor's music prompt -> MusicCoCa text embedding (with the text->audio mapper, as
   `mrt jax generate` does). Each 0.4 s chunk gets sum_k w_k(t) e_k, renormalised, from weights.json
   (the weights at the chunk's centre time; movie time == track time).
2. Key: Magenta RT 2 takes a 128-pitch notes conditioning per 40 ms frame (-1 masked, 0 off, 3 free).
   `--notes key` turns every pitch outside the track's diatonic scale OFF, every pitch from one semitone
   under the data lead's lowest note upward OFF (the lead's register is left to the data), and the
   harmony stem's chord tones (AMT strings + bass, from the arranged timeline) below that FREE.
   `--notes none` masks everything (style only). The result is checked with chroma correlation against
   the harmony stem (shift-0 vs the 11 transpositions) and the share of chroma on the scale.
3. Generation is continuous: one streaming state over the whole track (2 s pre-roll discarded), so
   the texture is in sync with the timeline.
4. Mix: the integrate stems (lead, bands, events, harmony, voices) are read unchanged; the mood stem
   sits `--mood-db` under the melody's active RMS, goes through the same sidechain (duck_gain against
   the lead, computed on the whole accompaniment bus incl. the mood, applied to the mood), then the
   whole sum is renormalised to -16 LUFS through the same limiter as arrange.py.
5. Fidelity: the same note-level pyin round trip (arrange.fidelity) on the old mix, the new mix, and
   the shuffled-control mix with mood.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..render import arrange as A
from ..render.synth import SR
from .mood import INTEGRATE, ROOT, corr

MAGENTA_HOME = "/projects/e32706/omb8654/magenta_home"
FRAME_RATE = 25            # Magenta RT 2 frames per second (40 ms)
MRT_SR = 48000
SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "major_pent": (0, 2, 4, 5, 7, 9, 11),   # pentatonic lead: allow its
          "minor": (0, 2, 3, 5, 7, 8, 10), "minor_pent": (0, 2, 3, 5, 7, 8, 10)}  # parent diatonic scale


# ------------------------------------------------------------------ inputs
def track_notes(src: Path):
    return A.load_notes(src / "timeline.json")


def lead_spec(tid: str, src: Path):
    from ..render.arrangements import get
    cfg = get(tid, src)
    return cfg, cfg.fidelity[0]


def scale_pcs(pm: dict) -> set[int]:
    root = int(pm.get("root", 0))
    return {(root + i) % 12 for i in SCALES.get(pm.get("scale", "major"), SCALES["major"])}


def notes_roll(t0: float, t1: float, chords, lead_low: int, pcs: set[int], mode: str) -> list[int]:
    roll = [-1] * 128
    if mode == "none":
        return roll
    tones = set()
    for a, b, ps in chords or []:
        if b > t0 and a < t1:
            tones |= {p % 12 for p in ps}
    for p in range(128):
        if p >= lead_low - 1 or (p % 12) not in pcs:
            roll[p] = 0
        elif p % 12 in tones and 36 <= p:
            roll[p] = 3
    return roll


# ------------------------------------------------------------------ generation
def generate(tid: str, size: str = "mrt2_small", *, chunk_frames: int = 10, notes: str = "key",
             cfg_style: float = 3.0, cfg_notes: float = 1.0, temperature: float = 1.2, top_k: int = 40,
             preroll_s: float = 2.0, out: Path | None = None, src: Path | None = None, tag: str | None = None) -> dict:
    os.environ.setdefault("MAGENTA_HOME", MAGENTA_HOME)
    import jax
    from magenta_rt import MagentaRT2Jax
    from magenta_rt.config import MUSICCOCA, PIANOROLL_WITH_ONSETS

    tid = str(tid)
    tag = tag or size.replace("mrt2_", "")
    out = Path(out or ROOT / "outputs" / f"{tid}_mood")
    src = Path(src or INTEGRATE / tid)
    W = json.loads((out / "weights.json").read_text())
    wt, w = np.asarray(W["t"]), np.asarray(W["weights"])
    prompts = [a["music"] for a in W["anchors"]]
    duration = float(W["duration_s"])
    cfg, spec = lead_spec(tid, src)
    nts = track_notes(src)
    lead = A.fid_notes(nts, spec)
    lead_low = min(n.pitch for n in lead)
    harm = [n for n in nts if n.voice in A.MODEL_VOICES]
    chords = A.chord_timeline(harm or [n for n in nts if A.group_of(n.voice, cfg) == "bands"], duration)
    pcs = scale_pcs(spec.pm)

    t_load = time.time()
    mrt = MagentaRT2Jax(size=size, temperature=temperature, top_k=top_k,
                        cfg_scales={"musiccoca": cfg_style, "notes": cfg_notes, "drums": 1.0})
    t_load = time.time() - t_load
    E = np.stack([np.asarray(mrt.embed_style(p, use_mapper=True), np.float32) for p in prompts])

    def style_at(t):
        wk = np.array([np.interp(t, wt, w[:, k]) for k in range(w.shape[1])])
        e = (wk / wk.sum()) @ E
        return (e / np.linalg.norm(e)).astype(np.float32), wk

    n_chunks = int(np.ceil((duration + cfg.tail) * FRAME_RATE / chunk_frames))
    t_gen = time.time()
    state = None
    pre = int(round(preroll_s * FRAME_RATE))
    if pre:
        e0, _ = style_at(0.0)
        r0 = notes_roll(0.0, chunk_frames / FRAME_RATE, chords, lead_low, pcs, notes)
        _, state = mrt.generate(conditioning={MUSICCOCA.key: e0, PIANOROLL_WITH_ONSETS.key: r0}, frames=pre, state=state)
    pieces, log = [], []
    for i in range(n_chunks):
        a, b = i * chunk_frames / FRAME_RATE, (i + 1) * chunk_frames / FRAME_RATE
        e, wk = style_at(0.5 * (a + b))
        roll = notes_roll(a, b, chords, lead_low, pcs, notes)
        wav, state = mrt.generate(conditioning={MUSICCOCA.key: e, PIANOROLL_WITH_ONSETS.key: roll},
                                  frames=chunk_frames, state=state)
        pieces.append(np.asarray(wav.samples, np.float32))
        if i % 25 == 0:
            print(f"[mood] {tid} chunk {i}/{n_chunks} t={a:.1f}s elapsed {time.time() - t_gen:.0f}s", flush=True)
            log.append({"t": round(a, 2), "top": prompts[int(np.argmax(wk))], "w": round(float(wk.max()), 3),
                        "free_pitches": int(sum(x == 3 for x in roll))})
    t_gen = time.time() - t_gen
    y = np.concatenate(pieces)
    if y.ndim == 1:
        y = np.stack([y, y], 1)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(SR, MRT_SR)
    y = resample_poly(y, SR // g, MRT_SR // g, axis=0).astype(np.float32)
    A.write_wav(out / f"mood_{tag}.wav", y / max(1.0, float(np.abs(y).max())))
    audio_s = len(y) / SR
    info = {"id": tid, "size": size, "tag": tag, "device": str(jax.devices()[0].platform),
            "devices": [str(d) for d in jax.devices()], "chunk_frames": chunk_frames, "chunk_s": chunk_frames / FRAME_RATE,
            "notes_mode": notes, "cfg_style": cfg_style, "cfg_notes": cfg_notes, "temperature": temperature,
            "top_k": top_k, "preroll_s": preroll_s, "prompts": prompts, "scale_pcs": sorted(pcs),
            "lead_low": lead_low, "load_s": round(t_load, 1), "gen_s": round(t_gen, 1), "audio_s": round(audio_s, 2),
            "gen_s_per_audio_min": round(t_gen / audio_s * 60, 1), "realtime_factor": round(audio_s / t_gen, 3),
            "log": log}
    (out / f"generate_{tag}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("id", "size", "device", "gen_s", "audio_s", "gen_s_per_audio_min")}))
    return info


# ------------------------------------------------------------------ key check
def chroma(x: np.ndarray, sr: int = SR, hop: int = 2048) -> np.ndarray:
    import librosa
    m = x.mean(1) if x.ndim == 2 else x
    from scipy.signal import resample_poly
    m = resample_poly(m, 1, 2) if sr == 44100 else m
    return librosa.feature.chroma_cqt(y=m.astype(np.float32), sr=22050, hop_length=hop // 2 if sr == 44100 else hop)


def key_check(mood: np.ndarray, harmony: np.ndarray | None, pcs: set[int], seg_s: float = 2.0) -> dict:
    cm = chroma(mood)
    fps = 22050 / 1024
    out = {"scale_share": float(cm[sorted(pcs)].sum() / (cm.sum() + 1e-9)), "scale_share_chance": len(pcs) / 12}
    prof = cm.mean(1)
    out["mood_chroma_profile"] = np.round(prof / prof.sum(), 4).tolist()
    if harmony is None or not np.any(harmony):
        return out
    ch = chroma(harmony)
    n = min(cm.shape[1], ch.shape[1])
    k = int(seg_s * fps)
    segs = [(cm[:, i:i + k].mean(1), ch[:, i:i + k].mean(1)) for i in range(0, n - k + 1, k)]
    segs = [(a, b) for a, b in segs if b.sum() > 1e-3 and a.sum() > 1e-3]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    sims = np.array([[cos(np.roll(a, s), b) for s in range(12)] for a, b in segs])
    if not len(sims):
        return out
    mean_by_shift = sims.mean(0)
    out.update({"harmony_chroma_cos_shift0": float(mean_by_shift[0]),
                "harmony_chroma_cos_other_shifts_mean": float(mean_by_shift[1:].mean()),
                "best_shift_semitones": int(np.argmax(mean_by_shift)),
                "share_segments_shift0_best": float(np.mean(sims.argmax(1) == 0)),
                "segments": int(len(sims)),
                "clash": bool(mean_by_shift[0] < mean_by_shift[1:].mean())})
    return out


# ------------------------------------------------------------------ does the audio follow the weights?
def style_follow(tid: str, tag: str, out: Path, hop_s: float = 2.0) -> dict:
    """MusicCoCa audio embeddings of the raw generated mood (10 s clips every hop_s) against the anchor
    prompts' text embeddings (mapped to audio space). For each anchor: r over clips between the clip's
    cosine to that prompt and the mean weight the model was given over that clip. Also the share of clips
    whose closest prompt is the prompt with the largest weight (chance = 1/K)."""
    os.environ.setdefault("MAGENTA_HOME", MAGENTA_HOME)
    import soundfile as sf
    from magenta_rt.audio import Waveform
    from magenta_rt.musiccoca import MusicCoCa
    W = json.loads((out / "weights.json").read_text())
    wt, w = np.asarray(W["t"]), np.asarray(W["weights"])
    names = [a["name"] for a in W["anchors"]]
    m = MusicCoCa()
    T = np.stack([np.asarray(m.embed(a["music"], use_mapper=True), np.float32) for a in W["anchors"]])
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    y, sr = sf.read(str(out / f"mood_{tag}.wav"), dtype="float32", always_2d=True)
    X = m.embed_batch_audio([Waveform(y, sr)], hop_length=hop_s, pool_across_time=False, pad_end=False)[0]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos = X @ T.T
    clip = m.config.clip_length
    starts = np.arange(len(X)) * hop_s
    wm = np.array([[w[(wt >= a) & (wt < a + clip), k].mean() if np.any((wt >= a) & (wt < a + clip)) else np.nan
                    for k in range(len(names))] for a in starts])
    ok = np.all(np.isfinite(wm), 1)
    per = {n: corr(cos[ok, k], wm[ok, k]) for k, n in enumerate(names)}
    agree = float(np.mean(cos[ok].argmax(1) == wm[ok].argmax(1))) if ok.any() else None
    # control: the same audio against weights reversed in time (any agreement from a static texture)
    wr = wm[ok][::-1]
    per_rev = {n: corr(cos[ok, k], wr[:, k]) for k, n in enumerate(names)}
    vals = [v for v in per.values() if v is not None]
    vals_r = [v for v in per_rev.values() if v is not None]
    return {"clips": int(ok.sum()), "clip_s": clip, "hop_s": hop_s, "r_per_anchor": per,
            "mean_r": float(np.mean(vals)) if vals else None, "top_prompt_agreement": agree,
            "chance_agreement": 1 / len(names), "mean_r_time_reversed_weights": float(np.mean(vals_r)) if vals_r else None,
            "closest_prompt_share": {n: float(np.mean(cos[ok].argmax(1) == k)) for k, n in enumerate(names)}}


# ------------------------------------------------------------------ mix + fidelity
def load_stems(src: Path) -> dict:
    import soundfile as sf
    tr = json.loads((src / "track.json").read_text())
    stems = {}
    for s in tr["stems"]:
        a, sr = sf.read(str(src / s["file"]), dtype="float32", always_2d=True)
        assert sr == SR
        stems[s["name"]] = (s["group"], a)
    return stems


def mix_mood(tid: str, tag: str = "small", *, mood_db: float = -14.0, highpass_hz: float = 80.0,
             out: Path | None = None, src: Path | None = None, label: str | None = None, duck: bool = True) -> dict:
    """label: output name suffix (default = tag); e.g. a louder stress-test mix of the same mood stem."""
    import soundfile as sf
    from scipy.signal import butter, sosfilt
    tid = str(tid)
    label = label or tag
    out = Path(out or ROOT / "outputs" / f"{tid}_mood")
    src = Path(src or INTEGRATE / tid)
    cfg, spec = lead_spec(tid, src)
    notes = track_notes(src)
    stems = load_stems(src)
    n = len(stems["melody"][1])
    mood, sr = sf.read(str(out / f"mood_{tag}.wav"), dtype="float32", always_2d=True)
    assert sr == SR
    mood = np.pad(mood, ((0, max(0, n - len(mood))), (0, 0)))[:n]
    if mood.shape[1] == 1:
        mood = np.repeat(mood, 2, 1)
    if highpass_hz:
        mood = sosfilt(butter(2, highpass_hz, "highpass", fs=SR, output="sos"), mood, axis=0).astype(np.float32)
    fade = int(0.5 * SR)
    mood[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:, None]
    mood[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    mel = stems["melody"][1]
    ref = A.active_rms(mel)
    g = ref * 10 ** (mood_db / 20) / max(A.active_rms(mood), 1e-9)
    mood = mood * np.float32(g)
    acc = sum(a for gname, a in stems.values() if gname in cfg.duck_groups)
    duck_info = None
    if cfg.duck_db is not None and duck:
        gd = A.duck_gain(mel, acc + mood, cfg.duck_db, cfg.duck_max_db)
        mood = mood * gd[:, None]
        duck_info = {"margin_db": cfg.duck_db, "max_db": cfg.duck_max_db, "mean_gr_db": float(-20 * np.log10(np.mean(gd))),
                "share_of_time_ducked_over_1db": float(np.mean(gd < 10 ** (-1 / 20)))}
    pre_old = sum(a for _, a in stems.values())
    pre = pre_old + mood
    L0 = A.lufs(pre)
    k = 10 ** ((cfg.lufs - L0) / 20)
    k = min(k, 10 ** (4.5 / 20) * 10 ** (-1 / 20) / max(float(np.abs(pre).max()), 1e-9))
    mix, gr = A.limiter(pre * np.float32(k))
    ctrl, _ = sf.read(str(src / "control_shuffled.wav"), dtype="float32", always_2d=True)
    mix_ctrl, _ = A.limiter((pre - mel + ctrl[:n]) * np.float32(k))
    old_mix, _ = sf.read(str(src / "mix.wav"), dtype="float32", always_2d=True)
    old_ctrl_mix, _ = sf.read(str(src / "mix_control.wav"), dtype="float32", always_2d=True)
    A.write_wav(out / f"mood_stem_{label}.wav", mood * np.float32(k))
    A.write_wav(out / f"mix_mood_{label}.wav", mix)
    A.write_wav(out / f"mix_control_mood_{label}.wav", mix_ctrl)

    rng = A.pyin_range(notes, spec)
    shuf = A.shuffled_control(notes, spec.voices, cfg.seed)
    truth = A.fid_notes(notes, spec)

    def fid(audio, ctrl_notes=False):
        tr = A.pyin_track(audio, *rng)
        if ctrl_notes:
            return A.fidelity(audio, shuf, spec, truth_notes=truth, track=tr)
        return A.fidelity(audio, notes, spec, track=tr)
    t0 = time.time()
    f = {"mix_without_mood": fid(old_mix), "mix_with_mood": fid(mix),
         "control_mix_without_mood": fid(old_ctrl_mix, True), "control_mix_with_mood": fid(mix_ctrl, True),
         "control_stem": fid(ctrl[:n], True)}
    t_fid = time.time() - t0
    harmony = stems.get("harmony", (None, None))[1]
    kc = key_check(mood, harmony, scale_pcs(spec.pm))
    lv = {name: A.db(A.active_rms(a * np.float32(k))) for name, (gname, a) in stems.items()}
    lv["mood"] = A.db(A.active_rms(mood * np.float32(k)))
    try:
        follow = style_follow(tid, tag, out)
    except Exception as e:  # pragma: no cover
        follow = {"error": f"{type(e).__name__}: {e}"[:300]}
    rep = {"id": tid, "tag": tag, "label": label, "mood_db_vs_melody": mood_db, "highpass_hz": highpass_hz, "ducking": duck_info,
           "lufs_pre": L0, "lufs_mix": A.lufs(mix), "limiter_gr_db": gr, "norm_gain_db": A.db(k),
           "active_rms_db": lv, "fidelity": f, "series": spec.label,
           "r2_drop": (f["mix_without_mood"]["r2"] - f["mix_with_mood"]["r2"]) if f["mix_with_mood"]["r2"] is not None else None,
           "key_check": kc, "style_follow": follow, "fidelity_s": round(t_fid, 1)}
    (out / f"mood_report_{label}.json").write_text(json.dumps(A.json_safe(rep), indent=1))
    try:
        _plot(out, tid, label, mood * np.float32(k), mix, notes, cfg)
    except Exception as e:  # pragma: no cover
        print(f"[mood] plot failed: {e}")
    print(json.dumps(A.json_safe({"id": tid, "tag": label, "r2_without": f["mix_without_mood"]["r2"],
                                  "r2_with": f["mix_with_mood"]["r2"], "ctrl_with": f["control_mix_with_mood"]["r2"],
                                  "pitch_ok_with": f["mix_with_mood"]["pitch_within_half_semitone"],
                                  "key": {k_: kc.get(k_) for k_ in ("scale_share", "harmony_chroma_cos_shift0",
                                                                     "harmony_chroma_cos_other_shifts_mean", "best_shift_semitones")}})))
    return rep


def _plot(out, tid, label, mood, mix, notes, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    W = json.loads((out / "weights.json").read_text())
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t = np.asarray(W["t"])
    w = np.asarray(W["weights"])
    ax[0].stackplot(t, w.T, labels=[a["music"] for a in W["anchors"]], alpha=0.85)
    ax[0].legend(fontsize=7, ncol=3, loc="upper left"); ax[0].set_title(f"{tid}: style weights sent to Magenta RT ({label})")
    for a_, lab, ttl in ((mood, "mood", 1), (mix, "mix", 2)):
        f, tt, S = spectrogram(a_.mean(1), SR, nperseg=2048, noverlap=1024)
        sel = f < 4000
        ax[ttl].pcolormesh(tt, f[sel], 10 * np.log10(S[sel] + 1e-12), shading="auto", cmap="magma", vmin=-110, vmax=-30)
        ax[ttl].set_ylabel("Hz"); ax[ttl].set_title(f"{lab} spectrogram (0-4 kHz)")
    for n_ in notes:
        if A.group_of(n_.voice, cfg) == "melody":
            hz = 440 * 2 ** ((n_.pitch - 69) / 12)
            ax[2].plot([n_.start, n_.end], [hz, hz], color="cyan", lw=1)
    ax[2].set_xlabel("s")
    fig.tight_layout(); fig.savefig(out / f"mood_{label}.png", dpi=80); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--size", default="mrt2_small")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--notes", default="key", choices=("key", "none"))
    ap.add_argument("--chunk-frames", type=int, default=10)
    ap.add_argument("--cfg-style", type=float, default=3.0)
    ap.add_argument("--cfg-notes", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--mood-db", type=float, default=-14.0)
    ap.add_argument("--mix-only", action="store_true")
    ap.add_argument("--no-duck", action="store_true", help="stress test: no sidechain on the mood")
    ap.add_argument("--label", default=None, help="output suffix for the mix (default: the tag)")
    a = ap.parse_args(argv)
    tag = a.tag or a.size.replace("mrt2_", "") + ("" if a.notes == "key" else "-nonotes")
    for tid in a.ids:
        if not a.mix_only:
            generate(tid, a.size, chunk_frames=a.chunk_frames, notes=a.notes, cfg_style=a.cfg_style,
                     cfg_notes=a.cfg_notes, temperature=a.temperature, tag=tag)
        mix_mood(tid, tag, mood_db=a.mood_db, label=a.label, duck=not a.no_duck)


if __name__ == "__main__":
    main()
