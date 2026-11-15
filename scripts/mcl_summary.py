"""Summarise the MuseControlLite pilot: one markdown table row per track x variant x mode, and (with --export)
small MP3s + a plot into demo/<id>_mcl/.

    python scripts/mcl_summary.py 5209 5313 3912 [--export best-variant-per-track ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f(x, nd=3):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def rows(tid: str):
    out = ROOT / "outputs" / f"{tid}_mcl"
    rep = json.loads((out / "mcl_report.json").read_text())
    ref = rep["reference"]
    im, imc = ref.get("integrated_mix", {}), ref.get("integrated_mix_control", {})
    yield (tid, "integrated mix (current, deterministic)", "reference", im.get("r2"),
           im.get("pitch_within_half_semitone"), imc.get("r2"), (im.get("richness") or {}).get("non_lead_energy_share"),
           None, "CPU (FluidSynth + AMT)")
    for v, r in rep["variants"].items():
        if "_smoke" in v:
            continue
        run = next((x for x in r["runs"] if x["which"] == "real"), {})
        spm, dev = run.get("seconds_per_audio_minute"), run.get("device", "")
        a, ac = r["A"]["real"], r["A"].get("control", {})
        yield (tid, v, "A (generated alone)", a.get("r2"), a.get("pitch_within_half_semitone"), ac.get("r2"),
               (r["A"].get("richness") or {}).get("non_lead_energy_share"), spm, dev)
        for how, b in r["B"].items():
            ce = (b.get("control_e2e") or {}).get("r2")
            yield (tid, v, f"B-{how}", b["mix"].get("r2"), b["mix"].get("pitch_within_half_semitone"),
                   f(b["control"].get("r2")) if ce is None else f"{f(ce)} (lead-swap {f(b['control'].get('r2'))})",
                   None, spm, dev)


def table(ids):
    hdr = ("| track | variant | mode | R² | within ½ st | control R² | non-lead energy | s per audio min | device |\n"
           "|---|---|---|---|---|---|---|---|---|")
    lines = [hdr]
    for tid in ids:
        for r in rows(tid):
            lines.append("| " + " | ".join([r[0], r[1], r[2], f(r[3]), f(r[4], 2), f(r[5]), f(r[6], 2),
                                              f(r[7], 1), r[8]]) + " |")
    return "\n".join(lines)


def mp3(src: Path, dst: Path, kbps=96):
    ff = shutil.which("ffmpeg") or "/projects/e32706/omb8654/conda/envs/mcl/bin/ffmpeg"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-b:a", f"{kbps}k", str(dst)], check=True)


def export(tid: str, variant: str):
    out = ROOT / "outputs" / f"{tid}_mcl"
    d = ROOT / "demo" / f"{tid}_mcl"
    d.mkdir(parents=True, exist_ok=True)
    pairs = [(f"gen_{variant}_real.wav", "A_generated.mp3"), (f"gen_{variant}_control.wav", "A_generated_shuffled_control.mp3"),
             (f"mixB_{variant}_replace.wav", "B_replace_mix.mp3"), (f"mixB_{variant}_augment.wav", "B_augment_mix.mp3")]
    for s, t in pairs:
        if (out / s).exists():
            mp3(out / s, d / t)
    for s, t in ((f"mcl_{variant}.png", "spectrogram.png"), ("controls_mcl.png", "controls.png")):
        if (out / s).exists():
            shutil.copy2(out / s, d / t)
    rep = json.loads((out / "mcl_report.json").read_text())
    keep = {"id": tid, "variant": variant, "prompt": rep["prompt"], "fidelity_spec": rep["fidelity_spec"],
            "reference": rep["reference"], "result": rep["variants"][variant],
            "controls": json.loads((out / "controls_mcl.json").read_text())}
    (d / "mcl_report.json").write_text(json.dumps(keep, indent=1))
    r = rep["variants"][variant]
    a, ac = r["A"]["real"], r["A"].get("control", {})
    ref = rep["reference"]
    run = next((x for x in r["runs"] if x["which"] == "real"), {})
    seeds = [(k, v["A"]["real"].get("r2"), (v["A"].get("control") or {}).get("r2"))
             for k, v in rep["variants"].items() if k.startswith(variant) and "_seed" in k]
    ctl = keep["controls"]
    lines = [
        f"# {tid} with MuseControlLite (pilot, `{variant}`)",
        "",
        "Learned layer on top of the deterministic track. **Nothing here has been listened to**: it was judged by",
        "the pyin round trip and spectrograms only. See `demo/learned/README.md` for the method and the comparison.",
        "",
        f"- **Prompt:** \"{rep['prompt']}\"",
        f"- **Melody control:** {ctl['melody']['source']}, {ctl['melody']['n_notes']} notes; {ctl['melody']['format']}.",
        f"- **Dynamics control:** half the lead's own loudness envelope, half the voices' `controls.json` curves "
        f"({', '.join(ctl['dynamics']['curves']) or 'none non-zero'}).",
        f"- **Rhythm control:** {ctl['rhythm']['plan']}.",
        f"- **Segments:** {len(ctl['segments'])} x <= 47 s" + (f", {ctl['crossfade_s']:.0f} s equal-power crossfade" if len(ctl['segments']) > 1 else "") + ".",
        f"- **Model:** {run.get('adapter')} adapters on Stable Audio Open 1.0, {run.get('steps')} steps, text CFG {run.get('guidance_text')}, "
        f"control CFG {run.get('guidance_con')}, seed {run.get('seed')}, {run.get('dtype')}, {run.get('device')}: "
        f"{run.get('gen_seconds')} s for {run.get('audio_s', 0):.1f} s of audio.",
        "",
        "| file | what | R² vs data | within ½ semitone | shuffled control R² |",
        "|---|---|---|---|---|",
        f"| `A_generated.mp3` | mode A: generated audio alone | {f(a.get('r2'))} | {f(a.get('pitch_within_half_semitone'), 2)} | {f(ac.get('r2'))} |",
        "| `A_generated_shuffled_control.mp3` | same prompt, seed and settings, melody control from the shuffled lead | | | |",
    ]
    for how in ("replace", "augment"):
        b = r["B"].get(how)
        if b:
            ce = (b.get("control_e2e") or {}).get("r2")
            what = ("mode B: data-lead stem + bands + events + generated (replaces AMT harmony and recorded voices)"
                    if how == "replace" else "mode B: every current stem + generated, 13 dB under the lead")
            lines.append(f"| `B_{how}_mix.mp3` | {what} | {f(b['mix'].get('r2'))} | {f(b['mix'].get('pitch_within_half_semitone'), 2)} | "
                         f"{f(ce)} end to end, {f(b['control'].get('r2'))} lead swap |")
    im = ref.get("integrated_mix", {})
    lines += [f"| (`demo/{tid}/mix.mp3`) | current integrated mix, same measurement | {f(im.get('r2'))} | "
              f"{f(im.get('pitch_within_half_semitone'), 2)} | {f(ref.get('integrated_mix_control', {}).get('r2'))} |", ""]
    if seeds:
        lines.append("Other seeds, mode A (R², control R²): " + "; ".join(f"`{k}` {f(x)}, {f(y)}" for k, x, y in seeds) + ".")
        lines.append("")
    rich = r["A"].get("richness", {})
    lines += [f"Share of spectral energy outside the lead's harmonics while lead notes sound: generated "
              f"{f(rich.get('non_lead_energy_share'), 2)}, current integrated mix "
              f"{f((im.get('richness') or {}).get('non_lead_energy_share'), 2)}.",
              "", "`spectrogram.png`: generated CQT with the data notes and the pyin track. `controls.png`: the three controls.",
              "`mcl_report.json`: every number above plus the controls summary.", "",
              "Credits: Stable Audio Open 1.0 (Stability AI Community License, \"Powered by Stability AI\"), "
              "MuseControlLite adapters and code (MIT, Tsai et al. 2025). See `demo/learned/README.md`."]
    (d / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--export", nargs="*", default=None, help="one variant per id, in the same order")
    a = ap.parse_args()
    print(table(a.ids))
    if a.export:
        for tid, v in zip(a.ids, a.export):
            export(tid, v)
