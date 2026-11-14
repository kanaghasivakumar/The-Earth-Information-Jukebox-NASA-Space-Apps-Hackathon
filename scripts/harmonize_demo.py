"""Keep the data melody, add model-written harmony + bass (AMT), render with FluidSynth.

    PYTHONPATH=. python scripts/harmonize_demo.py                       # 3912 demo, medium model
    PYTHONPATH=. python scripts/harmonize_demo.py --model stanford-crfm/music-small-800k --seed 3

Reads  demo/3912/timeline.json (+ report.json for duration / pitch map)
Writes outputs/3912_harmony/{mix.wav, score.mid, timeline.json, report.json, stems/*.wav}
       demo/3912_harmony/{mix.mp3, lead.mp3, harmony.mp3, bass.mp3, score.mid, timeline.json, report.json,
                          pianoroll.png}
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jukebox.render import fluid  # noqa: E402
from jukebox.render.synth import SR, mix  # noqa: E402
from jukebox.score.infill import DEFAULT_MODEL, clash_report, infill  # noqa: E402
import jukebox.score.infill as infill_mod  # noqa: E402
from jukebox.score.mapping import Note, timeline  # noqa: E402

GM_NAMES = {0: "piano", 11: "vibraphone", 32: "acoustic bass", 33: "finger bass", 48: "strings", 89: "warm pad"}


def load_notes(path: Path) -> list[Note]:
    return [Note(**d) for d in json.loads(path.read_text())]


def json_safe(obj):
    """NaN -> null so browsers' JSON.parse accepts the timeline (value is NaN for model notes)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_wav(path: Path, audio: np.ndarray):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SR, subtype="PCM_16")


def to_mp3(wav: Path, mp3: Path, kbps: int = 160):
    exe = Path(sys.executable).parent / "ffmpeg"
    exe = str(exe) if exe.exists() else (shutil.which("ffmpeg") or "ffmpeg")
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", f"{kbps}k",
                    str(mp3)], check=True)


def pitch_accuracy(audio: np.ndarray, lead: list[Note], step: float = 0.5) -> dict:
    """pyin on the rendered audio vs the lead pitch sounding at each step (does the accompaniment
    mask the data melody?). Returns fraction within 0.5 semitone (octave errors count as misses)."""
    from jukebox.proof.fidelity import recover_curve
    from jukebox.score.mapping import PitchMap
    end = max(n.end for n in lead)
    grid = np.arange(0, end, step)
    truth = np.array([next((n.pitch for n in lead if n.start <= t < n.end), np.nan) for t in grid], float)
    ident = PitchMap(0, 127, low=0, span=127)  # recover_curve then returns MIDI pitch directly
    est = recover_curve(audio, SR, ident, grid, fmin=150.0, fmax=2000.0)
    m = np.isfinite(truth) & np.isfinite(est)
    err = np.abs(est[m] - truth[m])
    return {"frames": int(len(grid)), "voiced": int(m.sum()), "within_half_semitone": float((err < 0.5).mean()) if m.any() else float("nan"),
            "median_abs_err_semitones": float(np.median(err)) if m.any() else float("nan")}


def piano_roll(path: Path, notes: list[Note], duration: float, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"lead": "#d9480f", "harmony": "#1c7ed6", "bass": "#2b8a3e"}
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for n in notes:
        c = colors.get(n.voice, "#adb5bd")
        ax.add_patch(plt.Rectangle((n.start, n.pitch - 0.4), n.end - n.start, 0.8, color=c,
                                   alpha=0.95 if n.voice == "lead" else 0.6, lw=0))
    ps = [n.pitch for n in notes]
    ax.set_xlim(0, duration); ax.set_ylim(min(ps) - 2, max(ps) + 2)
    ax.set_xlabel("time (s)"); ax.set_ylabel("MIDI pitch"); ax.set_title(title)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
              labels=["lead (data, fixed)", "harmony (AMT)", "bass (AMT)"], loc="upper left",
              bbox_to_anchor=(1.005, 1.0), fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", type=Path, default=Path("demo/3912/timeline.json"))
    ap.add_argument("--report", type=Path, default=None, help="M1 report.json (for duration); default next to timeline")
    ap.add_argument("--out", type=Path, default=Path("outputs/3912_harmony"))
    ap.add_argument("--demo-dir", type=Path, default=Path("demo/3912_harmony"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--lead-program", type=int, default=73, help="GM program the model is told the lead is")
    ap.add_argument("--render-lead-program", type=int, default=73,
                    help="GM program the lead is rendered with (73 flute: sustained, stays on top of the mix)")
    ap.add_argument("--keep-bands", action="store_true", help="also keep the six latitude-band pad voices (quiet)")
    ap.add_argument("--no-reverb", action="store_true")
    ap.add_argument("--no-fidelity", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="re-render the harmony/bass already in --out/timeline.json")
    ap.add_argument("--volumes", default="lead=120,harmony=45,bass=85,band=40",
                    help="MIDI CC7 per voice group")
    a = ap.parse_args(argv)

    src = load_notes(a.timeline)
    report_in = a.report or a.timeline.with_name("report.json")
    meta = json.loads(report_in.read_text()) if report_in.exists() else {}
    duration = float(meta.get("duration_s") or max(n.end for n in src))
    lead = [n for n in src if n.voice == "lead"]
    bands = [n for n in src if n.voice.startswith("band")] if a.keep_bands else []
    lead_before = [asdict(n) for n in lead]

    # 1. harmony + bass around the fixed lead
    t0 = time.time()
    prev = a.out / "timeline.json"
    if a.reuse and prev.exists():
        new = [Note(**{**d, "value": float("nan")}) for d in json.loads(prev.read_text())
               if d["voice"] in ("harmony", "bass")]
        old = json.loads((a.out / "report.json").read_text()) if (a.out / "report.json").exists() else {}
        t_infill = old.get("timing_s", {}).get("infill", float("nan"))
        stats = infill_mod.InfillStats(tokens=old.get("timing_s", {}).get("tokens_fed", 0),
                                       slides=old.get("timing_s", {}).get("window_slides", 0))
    else:
        new = infill(src, duration, model=a.model, seed=a.seed, top_p=a.top_p, lead_program=a.lead_program,
                     verbose=True)
        t_infill = time.time() - t0
        stats = infill_mod.LAST_STATS
    assert [asdict(n) for n in lead] == lead_before, "lead notes were modified"
    notes = lead + bands + new

    # 2. score + timeline (model notes keep voice 'harmony'/'bass'; program is in source)
    a.out.mkdir(parents=True, exist_ok=True)
    rnotes = [replace(n, voice=f"{n.voice}:{n.source['program']}") if n.voice in ("harmony", "bass") else n
              for n in notes]
    programs = {"lead": a.render_lead_program, **{n.voice: 89 for n in bands}}
    programs.update({n.voice: n.source["program"] for n in rnotes if ":" in n.voice})
    vol = {k: int(v) for k, v in (kv.split("=") for kv in a.volumes.split(","))}
    volumes = {v: next(x for k, x in vol.items() if v.startswith(k)) for v in programs}
    fluid.notes_to_midi(rnotes, a.out / "score.mid", programs, volumes=volumes)
    tl = json_safe(timeline(notes))
    (a.out / "timeline.json").write_text(json.dumps(tl))

    # 3. render: per-voice stems via FluidSynth (fallback: additive synth), then mix
    t1 = time.time()
    kw = dict(reverb=not a.no_reverb, volumes=volumes, tail=2.0)
    engine = "fluidsynth" if fluid.available(download=True) else "synth-fallback"
    groups = {"lead": [n for n in rnotes if n.voice == "lead"],
              "harmony": [n for n in rnotes if n.voice.startswith("harmony")],
              "bass": [n for n in rnotes if n.voice.startswith("bass")]}
    if bands:
        groups["bands"] = [n for n in rnotes if n.voice.startswith("band")]
    stems = {k: fluid.render_notes(v, duration, programs, **kw) for k, v in groups.items() if v}
    full = fluid.render_notes(rnotes, duration, programs, **kw)  # one synth pass = true mix
    t_render = time.time() - t1
    mixed = mix(full)
    write_wav(a.out / "mix.wav", mixed)
    peak = max(float(np.abs(full).max()), 1e-9)
    for k, s in stems.items():
        write_wav(a.out / "stems" / f"{k}.wav", (s / peak * 0.9).astype(np.float32))  # same gain as the mix

    # 4. checks
    rep = {
        "source_timeline": str(a.timeline), "duration_s": duration, "model": a.model, "seed": a.seed,
        "top_p": a.top_p, "lead_program_for_model": a.lead_program, "render_programs": programs,
        "render_engine": engine, "lead_unchanged": True,
        "notes": {"lead": len(lead), "bands": len(bands), "harmony": sum(n.voice == "harmony" for n in new),
                  "bass": sum(n.voice == "bass" for n in new)},
        "programs_used": sorted({(n.voice, n.source["program"]) for n in new}),
        "clash_report": clash_report(lead, new),
        "timing_s": {"infill": t_infill, "infill_per_60s_audio": t_infill / duration * 60,
                     "render": t_render, "tokens_fed": stats.tokens, "window_slides": stats.slides},
    }
    if not a.no_fidelity:
        rep["lead_pitch_accuracy"] = {"lead_stem": pitch_accuracy(stems["lead"], lead),
                                      "full_mix": pitch_accuracy(full, lead)}
    (a.out / "report.json").write_text(json.dumps(json_safe(rep), indent=1))
    piano_roll(a.out / "pianoroll.png", notes, duration,
               f"{a.timeline.parent.name}: data lead + AMT harmony/bass ({Path(a.model).name}, seed {a.seed})")

    # 5. demo copies
    a.demo_dir.mkdir(parents=True, exist_ok=True)
    to_mp3(a.out / "mix.wav", a.demo_dir / "mix.mp3")
    for k in stems:
        to_mp3(a.out / "stems" / f"{k}.wav", a.demo_dir / f"{k}.mp3", kbps=128)
    for f in ("score.mid", "timeline.json", "report.json", "pianoroll.png"):
        shutil.copy(a.out / f, a.demo_dir / f)
    print(json.dumps(json_safe(rep), indent=1))


if __name__ == "__main__":
    main()
