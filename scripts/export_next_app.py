"""Copy the eight data tracks and the learned-layer pilots into the Next.js app.

    python scripts/export_next_app.py [--integrate ../integrate/outputs]

Writes, under eic-vae/eic-sonification/frontend/:

  public/jukebox/tracks/index.json        the web/tracks index, unchanged
  public/jukebox/tracks/<id>/...          every file of web/tracks/<id>/ (track contract: web/README.md)
  public/jukebox/tracks/<id>/mix_control.mp3
                                          the full mix with the lead shuffled in time (arrange's
                                          mix_control.wav, 128 kbps), for the Compare view
  public/jukebox/tracks/<id>/versions/    pilot mixes for 5209, 5313, 3912 (MuseControlLite, mood layer), and
                                          mcl_bplus.mp3 wherever demo/<id>_mclbplus/ exists (MuseControlLite B+)
  public/jukebox/tracks/<id>/versions.json
                                          the versions the Compare view offers, with their fidelity numbers
  lib/data/voice-credits.json             the recordings in jukebox/voices/ATTRIBUTION.md, for /credits
  lib/data/catalogue.json                 per track: title, voice, length, fidelity and version count (chooser)

The plain-JS web/ app is left as it is. MP3 only; video stays linked from NASA SVS.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "tracks"
APP = ROOT / "eic-vae" / "eic-sonification" / "frontend"
OUT = APP / "public" / "jukebox" / "tracks"
PILOTS = ("5209", "5313", "3912")


def ffmpeg() -> str:
    return shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")


def r(x, d=4):
    return None if x is None else round(float(x), d)


def fid(r2=None, within=None, control_r2=None, control_kind=None, note=None):
    return {"r2": r(r2), "within_half_semitone": r(within, 3), "control_r2": r(control_r2), "control_kind": control_kind,
            "note": note}


def versions_for(tid: str, track: dict, has_control_mix: bool) -> list[dict]:
    f = track.get("fidelity") or {}
    demo = ROOT / "demo"
    mcl = demo / f"{tid}_mcl" / "mcl_report.json"
    mood = demo / f"{tid}_mood" / "summary.json"
    rep = json.loads(mcl.read_text()) if mcl.exists() else None
    mixctl = rep["reference"]["integrated_mix_control"]["r2"] if rep else None
    out = [{
        "id": "mix",
        "label": "Deterministic mix",
        "kind": "deterministic",
        "file": "mix.mp3",
        "line": "The data melody and bands (FluidSynth), model-written harmony and the three recorded Earth voices. The reference.",
        "fidelity": fid(f.get("mix_r2"), f.get("mix_pitch_within_half_semitone"), mixctl if mixctl is not None else f.get("control_r2"),
                        "full mix, lead shuffled" if mixctl is not None else "lead stem, shuffled",
                        "Measured on the full mix."),
    }]
    if rep:
        A, B = rep["result"]["A"], rep["result"]["B"]
        out.append({
            "id": "mcl_a", "label": "MuseControlLite, mode A", "kind": "learned", "file": "versions/mcl_A.mp3",
            "line": "Generated audio alone: Stable Audio Open with MuseControlLite adapters, steered by the data melody, dynamics and rhythm.",
            "fidelity": fid(A["real"]["r2"], A["real"]["pitch_within_half_semitone"], A["control"]["r2"],
                            "generated from the shuffled lead", "Measured on the generated audio alone."),
        })
        for key, label, line in (
            ("augment", "MuseControlLite, mode B (augment)",
             "Every current stem plus the generated audio as a bed, 13 dB under the data melody."),
            ("replace", "MuseControlLite, mode B (replace)",
             "Data melody, bands and events plus the generated bed, which replaces the model harmony and recorded voices."),
        ):
            b = B[key]
            out.append({
                "id": f"mcl_b_{key}", "label": label, "kind": "learned", "file": f"versions/mcl_B_{key}.mp3", "line": line,
                "fidelity": fid(b["mix"]["r2"], b["mix"]["pitch_within_half_semitone"], b["control_e2e"]["r2"],
                                "shuffled lead through the whole pipeline", "Measured on the full mix."),
            })
    if mood.exists():
        m = json.loads(mood.read_text())["mix"]["fidelity"]
        out.append({
            "id": "mood", "label": "Mood layer (Magenta RT 2)", "kind": "learned", "file": "versions/mood_mix.mp3",
            "line": "The deterministic mix plus a texture from Magenta RealTime 2, steered by SigLIP frame weights and the data curves, 14 dB under the melody.",
            "fidelity": fid(m["mix_with_mood"]["r2"], m["mix_with_mood"]["pitch_within_half_semitone"],
                            m["control_mix_with_mood"]["r2"], "full mix + mood, lead shuffled", "Measured on the full mix."),
        })
    bplus = demo / f"{tid}_mclbplus" / "bplus_report.json"
    if bplus.exists():
        b = json.loads(bplus.read_text())
        m, c = b["mix"].get("mix") or {}, b["mix"].get("control_e2e") or {}
        out.append({
            "id": "mcl_bplus", "label": "MuseControlLite B+ (no cymbals, with Earth voices)", "kind": "learned",
            "file": "versions/mcl_bplus.mp3",
            "line": "The data melody, bands and every recorded Earth voice at their arranged levels, with a MuseControlLite "
                    "accompaniment (bass kept) in place of the model harmony. Cymbals removed from the generated part only.",
            "fidelity": fid(m.get("r2"), m.get("pitch_within_half_semitone"), c.get("r2"),
                            "shuffled lead through the whole pipeline" if c else None,
                            "Measured on the full mix." if m else
                            "No single data melody to recover; see demo/" + tid + "_mclbplus/README.md for per-type checks."),
        })
    if has_control_mix:
        out.append({
            "id": "control", "label": "Shuffled control", "kind": "control", "file": "mix_control.mp3",
            "line": "The same arrangement with the melody's notes shuffled in time. It should sound like music but no longer follow the data.",
            "fidelity": fid(mixctl if mixctl is not None else f.get("control_r2"), None, None, None,
                            "R² of the shuffled melody against the real data." +
                            ("" if mixctl is not None else " Measured on the shuffled lead stem.")),
        })
    return out


def data_credit(ds) -> str:
    """The underlying dataset, in one line, for the credits page."""
    if isinstance(ds, str):
        return ds.split(". ")[0]
    if isinstance(ds, dict):
        return ds.get("underlying") or ds.get("music") or ""
    return ""


def voice_credits() -> list[dict]:
    text = (ROOT / "jukebox" / "voices" / "ATTRIBUTION.md").read_text()
    link = re.compile(r"^\[([^\]]+)\]\((.*)\)$")  # one link per cell; URLs may contain ()
    out, group = [], None
    for line in text.splitlines():
        if line.startswith("## "):
            group = line[3:].split(" ")[0]
            continue
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        lic, src = link.match(cells[5]), link.match(cells[6])
        out.append({
            "group": group, "title": cells[3], "author": cells[4],
            "license": lic.group(1) if lic else cells[5], "license_url": lic.group(2) if lic else None,
            "source": src.group(1) if src else cells[6], "url": src.group(2) if src else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--integrate", type=Path, default=ROOT.parent / "integrate" / "outputs",
                    help="arrange outputs with mix_control.wav per track")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEB / "index.json", OUT / "index.json")
    index = json.loads((WEB / "index.json").read_text())["tracks"]
    catalogue = []
    for entry in index:
        tid = str(entry["id"])
        src, dst = WEB / tid, OUT / tid
        dst.mkdir(exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        ctl = dst / "mix_control.mp3"
        wav = args.integrate / tid / "mix_control.wav"
        if not ctl.exists() and wav.exists():
            subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "128k", str(ctl)],
                           check=True)
        if tid in PILOTS:
            (dst / "versions").mkdir(exist_ok=True)
            for a, b in ((f"{tid}_mcl/A_generated.mp3", "mcl_A.mp3"), (f"{tid}_mcl/B_augment_mix.mp3", "mcl_B_augment.mp3"),
                         (f"{tid}_mcl/B_replace_mix.mp3", "mcl_B_replace.mp3"), (f"{tid}_mood/mix_mood.mp3", "mood_mix.mp3")):
                shutil.copy2(ROOT / "demo" / a, dst / "versions" / b)
        bp = ROOT / "demo" / f"{tid}_mclbplus" / "mix.mp3"
        if bp.exists():
            (dst / "versions").mkdir(exist_ok=True)
            shutil.copy2(bp, dst / "versions" / "mcl_bplus.mp3")
        track = json.loads((dst / "track.json").read_text())
        vs = versions_for(tid, track, ctl.exists())
        (dst / "versions.json").write_text(json.dumps({"id": tid, "duration_s": track["duration_s"], "versions": vs}, indent=1))
        f = track.get("fidelity") or {}
        catalogue.append({
            "id": tid, "title": track["title"], "voice": track.get("voice"), "summary": entry.get("summary"),
            "duration_s": track["duration_s"], "svs_url": track.get("svs_url"),
            "r2": r(f.get("mix_r2") if f.get("mix_r2") is not None else f.get("r2")), "control_r2": r(f.get("control_r2")),
            "versions": len(vs), "learned": any(v["kind"] == "learned" for v in vs),
            "data": data_credit(track.get("data_source")),
        })
        print(f"{tid}: {len(vs)} versions")
    (APP / "lib" / "data" / "catalogue.json").write_text(json.dumps(catalogue, indent=1, ensure_ascii=False))
    credits = voice_credits()
    (APP / "lib" / "data" / "voice-credits.json").write_text(json.dumps(credits, indent=1, ensure_ascii=False))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{len(credits)} recordings credited; tracks folder {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
