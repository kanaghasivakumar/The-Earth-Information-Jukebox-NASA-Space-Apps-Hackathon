"""Convert a pipeline output dir (outputs/<id>/ or demo/<id>/) into a web jukebox track.

    python scripts/export_track.py demo/3912 --meta scripts/track_meta/3912.json

Writes web/tracks/<id>/{track.json, timeline.json, controls.json, *.mp3} and upserts
web/tracks/index.json. See web/README.md for the track contract.

Stems found in the source dir (``.mp3`` copied, ``.wav`` encoded with ffmpeg):
  lead.*              -> group "melody"
  control_shuffled.*  -> the A/B control (not a stem; ``track.json["control"]``)
  mix.*               -> full mix fallback (``track.json["mix"]``), not played with the stems
  <group>*.*          -> any file whose name starts with a group name
                         (bands, events, harmony, geophony, anthrophony, biophony)
If there is no bands stem but the timeline has ``band ...`` voices, a bands-only stem is
re-rendered from the timeline with jukebox.render.synth at the level it has in mix.wav
(lead 1.0, bands 0.7), so the jukebox can mute lead and bands separately.

Videos are never copied: ``video_url`` points at the SVS server.

Arranged tracks (``jukebox.render.arrange`` output: ``track.json`` with a ``stems`` list) are
exported from that list instead: each stem keeps its name, group and label; ``mix`` and
``control`` are encoded too, and every other track.json key (segments, calendar, dates, sweep,
image_url, video_inversion, voice_groups, lead_voices, arrangement, ...) is passed through.
A ``video_url`` that is a still image (5669) becomes ``image_url`` (the player draws a sweep line).
``value: null`` (model-written notes) and NaN values are written as JSON null.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["melody", "bands", "events", "harmony", "geophony", "anthrophony", "biophony"]
VOICE_KEYWORDS = {  # SVS keyword -> default voice (THREE_VOICES.md §3b)
    "human dimensions": "anthrophony", "atmosphere": "geophony", "oceans": "geophony",
    "hazards": "geophony", "ecosystems": "biophony", "agriculture": "biophony", "biodiversity": "biophony",
}


def ffmpeg() -> str:
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg", path=str(Path(sys.executable).parent))
    if not found:
        raise SystemExit("ffmpeg not found (use the jukebox conda env)")
    return found


def to_mp3(src: Path, dest: Path, quality: int = 5, kbps: int | None = None) -> None:
    """WAV -> MP3 (VBR quality `quality`, or CBR `kbps` when given); MP3 sources are copied."""
    if src.suffix.lower() == ".mp3":
        shutil.copyfile(src, dest)
        return
    rate = ["-b:a", f"{kbps}k"] if kbps else ["-q:a", str(quality)]
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(src), "-codec:a", "libmp3lame", *rate, str(dest)],
                   check=True)


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff")
PASS_DROP = ("report", "overview", "timeline", "controls", "stems", "mix", "control")


def clean_value(v):
    """NaN / inf -> None so browsers' JSON.parse accepts the file."""
    return None if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))) else v


def calendar_from_sources(timeline: list[dict], voices) -> dict | None:
    """Spoken-date anchors from the data notes' own `source.date` / `source.year` (lead voices first)."""
    for pick in (lambda n: n["voice"] in voices, lambda n: True):
        pts = {}
        for n in timeline:
            src = n.get("source") or {}
            if not pick(n) or not isinstance(src, dict):
                continue
            d = src.get("date") or (f"{int(src['year'])}" if isinstance(src.get("year"), (int, float)) else None)
            if d:
                pts.setdefault(round(float(n["start"]), 3), str(d))
        if len(pts) >= 2:
            ts = sorted(pts)
            return {"dates_t": ts, "dates": [pts[t] for t in ts]}
    return None


def export_arranged(a, src: Path, track_in: dict, timeline: list[dict], meta: dict) -> None:
    """Export a jukebox.render.arrange output dir (stems listed in track.json)."""
    tid = str(a.id or meta.get("id") or track_in["id"])
    dest = a.out / tid
    if dest.exists():
        for f in dest.iterdir():
            if f.suffix in (".mp3", ".json"):
                f.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    loud = ("melody",)
    stems = []
    for s in track_in["stems"]:
        f = src / s["file"]
        name = Path(s["file"]).stem
        to_mp3(f, dest / f"{name}.mp3", kbps=a.kbps if s["group"] in loud else a.bed_kbps)
        stems.append({**s, "file": f"{name}.mp3"})
    track = {k: v for k, v in track_in.items() if k not in PASS_DROP}
    track["id"] = tid
    track["stems"] = stems
    if track_in.get("mix") and (src / track_in["mix"]).exists():
        to_mp3(src / track_in["mix"], dest / "mix.mp3", kbps=a.kbps)
        track["mix"] = "mix.mp3"
    ctrl = track_in.get("control")
    if ctrl and (src / ctrl["file"]).exists():
        to_mp3(src / ctrl["file"], dest / "control_shuffled.mp3", kbps=a.kbps)
        track["control"] = {k: v for k, v in ctrl.items() if k != "mix"} | {"file": "control_shuffled.mp3"}
    url = track.get("video_url") or ""
    if url.lower().endswith(IMAGE_EXT) or track.get("media_type") == "image":
        track["image_url"] = track.get("image_url") or url
        track["video_url"] = None
        track["media_type"] = "image"
    if not any(k in track for k in ("calendar", "dates", "time_axis")):
        cal = calendar_from_sources(timeline, track.get("lead_voices") or [track.get("lead_voice", "lead")])
        if cal:
            track.update(cal)
    keep = ("start", "end", "pitch", "velocity", "voice", "value", "pan")
    (dest / "timeline.json").write_text(json.dumps([{k: clean_value(n.get(k, 0.0)) for k in keep} for n in timeline],
                                                   allow_nan=False))
    track["timeline"] = "timeline.json"
    if (src / "controls.json").exists():
        shutil.copyfile(src / "controls.json", dest / "controls.json")
        track["controls"] = "controls.json"
    track.update({k: v for k, v in meta.items() if k not in ("order",)})
    track["id"] = tid
    (dest / "track.json").write_text(json.dumps(track, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    upsert_index(a.out, {"id": tid, "title": track["title"], "voice": track.get("voice"),
                         "summary": track.get("summary") or "", "duration_s": track.get("duration_s"),
                         **({"order": meta["order"]} if "order" in meta else {})})
    print(f"wrote {dest}: {[s['file'] for s in stems]}, video={track.get('video_url')}, image={track.get('image_url')}")


def find(src: Path, stem: str) -> Path | None:
    for ext in (".mp3", ".wav"):
        if (src / f"{stem}{ext}").exists():
            return src / f"{stem}{ext}"
    return None


def load_catalog(page_id: str, path: Path | None) -> dict:
    for p in [path, ROOT / "data" / "catalog" / f"{page_id}.json"]:
        if p and p.exists():
            return json.loads(p.read_text())
    try:
        req = urllib.request.Request(f"https://svs.gsfc.nasa.gov/api/{page_id}", headers={"User-Agent": "earth-jukebox"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:  # offline is fine: meta/override can fill the gaps
        print(f"warning: no catalog for {page_id}: {e}", file=sys.stderr)
        return {}


def pick_video(page: dict) -> str | None:
    """Smallest browser-friendly movie >= 600 px wide: mp4/m4v (H.264 plays everywhere), then webm."""
    sys.path.insert(0, str(ROOT))
    try:
        from jukebox.data.catalog import media
    except Exception:
        return None
    movies = [m for m in media(page) if m["type"] == "Movie"]
    for exts in ((".mp4", ".m4v"), (".webm",)):
        c = [m for m in movies if (m["filename"] or "").lower().endswith(exts) and (m["w"] or 0) >= 600]
        if c:
            return min(c, key=lambda m: (m["w"] or 0) * (m["h"] or 0))["url"]
    return None


def guess_voice(page: dict) -> str:
    for k in page.get("keywords") or []:
        v = VOICE_KEYWORDS.get(k.lower())
        if v:
            return v
    return "geophony"


def render_bands(timeline: list[dict], lead_voice: str, dest_wav: Path, duration: float) -> bool:
    """Re-render the band voices alone at their mix.wav level relative to lead.wav."""
    sys.path.insert(0, str(ROOT))
    try:
        import numpy as np
        import soundfile as sf
        from jukebox.render.synth import SR, render_notes
    except Exception as e:
        print(f"warning: cannot render bands stem ({e})", file=sys.stderr)
        return False
    notes = [SimpleNamespace(**{k: n[k] for k in ("start", "end", "pitch", "velocity")}, pan=n.get("pan", 0.0))
             for n in timeline]
    lead = [n for n, d in zip(notes, timeline) if d["voice"] == lead_voice]
    bands = [n for n, d in zip(notes, timeline) if d["voice"].startswith("band")]
    if not bands:
        return False
    lead_audio = render_notes(lead, duration, "pluck")
    band_audio = render_notes(bands, duration, "pad", gain=.12)
    scale = 0.9 / max(float(np.abs(lead_audio).max()), 1e-9)  # lead.wav normalisation (synth.mix)
    out = band_audio * 0.7 * scale                              # mix.wav gains: lead 1.0, bands 0.7
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out *= 0.99 / peak
        print(f"warning: bands stem peaked at {peak:.2f}; scaled down", file=sys.stderr)
    sf.write(str(dest_wav), out, SR, subtype="PCM_16")
    return True


def synthetic_controls(timeline: list[dict], lead_voice: str, duration: float, voice: str, step: float = 0.5) -> dict:
    """Placeholder voice-balance curves: the lead's normalised value drives the track's main voice."""
    lead = sorted((n for n in timeline if n["voice"] == lead_voice), key=lambda n: n["start"])
    vals = [n["value"] for n in lead] or [0.0]
    lo, hi = min(vals), max(vals)
    t, curve = [], []
    i, x = 0, 0.0
    while x <= duration + 1e-9:
        while i + 1 < len(lead) and lead[i + 1]["start"] <= x:
            i += 1
        v = lead[i]["value"] if lead else 0.0
        t.append(round(x, 3))
        curve.append(round((v - lo) / (hi - lo) if hi > lo else 0.5, 4))
        x += step
    zero = [0.0] * len(t)
    curves = {g: (curve if g == voice else zero) for g in ("geophony", "anthrophony", "biophony")}
    curves["brightness"] = curve
    return {"t": t, "curves": curves, "synthetic": True,
            "note": f"Placeholder: {voice} level and brightness follow the normalised lead value."}


def upsert_index(tracks_dir: Path, entry: dict) -> None:
    idx = tracks_dir / "index.json"
    data = json.loads(idx.read_text()) if idx.exists() else {"tracks": []}
    if isinstance(data, list):
        data = {"tracks": data}
    data["tracks"] = [t for t in data["tracks"] if str(t.get("id") if isinstance(t, dict) else t) != entry["id"]]
    data["tracks"].append(entry)
    data["tracks"].sort(key=lambda t: (t.get("order", 999), str(t["id"])))
    idx.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="pipeline output dir with report.json + timeline.json")
    ap.add_argument("--meta", type=Path, help="JSON merged over the generated track.json (title, description, legend, ...)")
    ap.add_argument("--catalog", type=Path, help="SVS API JSON (default data/catalog/<id>.json, else fetched)")
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "tracks")
    ap.add_argument("--id", help="track id (default report.json id)")
    ap.add_argument("--lead-voice", default="lead")
    ap.add_argument("--no-render-bands", action="store_true")
    ap.add_argument("--kbps", type=int, default=128, help="MP3 bitrate for melody, mix and control (arranged tracks)")
    ap.add_argument("--bed-kbps", type=int, default=96, help="MP3 bitrate for the other stems (arranged tracks)")
    a = ap.parse_args(argv)

    timeline = json.loads((a.src / "timeline.json").read_text())
    meta = json.loads(a.meta.read_text()) if a.meta else {}
    if (a.src / "track.json").exists():
        track_in = json.loads((a.src / "track.json").read_text())
        if track_in.get("stems"):
            return export_arranged(a, a.src, track_in, timeline, meta)
    report = json.loads((a.src / "report.json").read_text())
    tid = str(a.id or meta.get("id") or report["id"])
    page = load_catalog(tid, a.catalog)
    dest = a.out / tid
    dest.mkdir(parents=True, exist_ok=True)

    duration = float(report.get("duration_s") or max(n["end"] for n in timeline))
    voice = meta.get("voice") or guess_voice(page)
    units = report.get("units", "")

    # stems
    stems = []
    lead = find(a.src, "lead")
    if lead:
        to_mp3(lead, dest / "lead.mp3")
        stems.append({"name": "lead", "file": "lead.mp3", "group": "melody",
                      "label": "Lead melody (data notes)"})
    for f in sorted(a.src.iterdir()):
        g = next((g for g in GROUPS[1:] if f.stem.startswith(g)), None)
        if g and f.suffix.lower() in (".mp3", ".wav") and not any(s["name"] == f.stem for s in stems):
            to_mp3(f, dest / f"{f.stem}.mp3")
            stems.append({"name": f.stem, "file": f"{f.stem}.mp3", "group": g, "label": f.stem.replace("_", " ")})
    if not any(s["group"] == "bands" for s in stems) and not a.no_render_bands:
        tmp = dest / "_bands.wav"
        if render_bands(timeline, a.lead_voice, tmp, duration):
            to_mp3(tmp, dest / "bands.mp3")
            tmp.unlink()
            stems.append({"name": "bands", "file": "bands.mp3", "group": "bands",
                          "label": "Latitude bands (pads)", "derived": "re-rendered from timeline.json"})
    mix = find(a.src, "mix")
    if mix:
        to_mp3(mix, dest / "mix.mp3")
    ctrl = find(a.src, "control_shuffled")
    if ctrl:
        to_mp3(ctrl, dest / "control_shuffled.mp3")

    # timeline / controls
    keep = ("start", "end", "pitch", "velocity", "voice", "value", "pan")
    (dest / "timeline.json").write_text(json.dumps([{k: clean_value(n.get(k, 0.0)) for k in keep} for n in timeline],
                                                   allow_nan=False))
    if (a.src / "controls.json").exists():
        shutil.copyfile(a.src / "controls.json", dest / "controls.json")
    else:
        (dest / "controls.json").write_text(json.dumps(synthetic_controls(timeline, a.lead_voice, duration, voice)))

    real = report.get("fidelity_real") or {}
    ctl = report.get("fidelity_shuffled_control") or {}
    track = {
        "id": tid,
        "title": report.get("title") or page.get("title") or tid,
        "svs_url": f"https://svs.gsfc.nasa.gov/{tid}/",
        "video_url": pick_video(page),
        "duration_s": duration,
        "units": units,
        "description": (page.get("description") or "").split("||")[0].strip(),
        "voice": voice,
        "legend": [{"voice": "lead", "meaning": f"Lead melody: the data value in {units}. Higher pitch means a higher value."}],
        "stems": stems,
        "mix": "mix.mp3" if mix else None,
        "control": {"file": "control_shuffled.mp3", "replaces": "melody",
                    "label": "Same notes, data shuffled in time"} if ctrl else None,
        "timeline": "timeline.json",
        "controls": "controls.json",
        "lead_voice": a.lead_voice,
        "pitch_map": report.get("lead_pitchmap"),
        "fidelity": {"r2": real.get("r2"), "pearson_r": real.get("pearson_r"), "mae": real.get("mae"),
                     "control_r2": ctl.get("r2"), "control_pearson_r": ctl.get("pearson_r"),
                     "control_mae": ctl.get("mae"), "n": real.get("n")},
        "data_source": f"SVS {tid}, values recovered from the movie by colorbar inversion",
    }
    track.update(meta)
    track["id"] = tid
    (dest / "track.json").write_text(json.dumps(track, indent=1, ensure_ascii=False) + "\n")
    upsert_index(a.out, {"id": tid, "title": track["title"], "voice": track.get("voice"),
                         "summary": track.get("summary") or "", "duration_s": duration})
    print(f"wrote {dest}: {[s['file'] for s in stems]}, video={track['video_url']}")


if __name__ == "__main__":
    main()
