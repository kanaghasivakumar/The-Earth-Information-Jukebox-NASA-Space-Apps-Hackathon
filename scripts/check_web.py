"""Check web/tracks against the track contract (web/README.md).

    python scripts/check_web.py            # exit 1 on any problem

For every track in index.json: strict JSON (no NaN), every referenced file exists, stem groups are
valid, stems have the same length (ffprobe), timeline values are numbers or null, lead/voice-group
keys refer to voices that exist, and reports the folder size. Also runs `node --check` on web/js.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
GROUPS = {"melody", "bands", "events", "harmony", "geophony", "anthrophony", "biophony"}


def strict(path: Path):
    def bad(c):
        raise ValueError(f"{path}: non-standard JSON constant {c}")
    return json.loads(path.read_text(), parse_constant=bad)


def mp3_seconds(p: Path) -> float | None:
    exe = shutil.which("ffprobe") or str(Path(sys.executable).parent / "ffprobe")
    try:
        out = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                             capture_output=True, text=True, check=True).stdout
        return float(out.strip())
    except Exception:
        return None


def check_track(tid: str, problems: list) -> dict:
    d = WEB / "tracks" / tid
    t = strict(d / "track.json")
    for k in ("id", "title", "svs_url", "duration_s", "units", "description", "legend", "stems", "timeline",
              "controls", "fidelity", "data_source"):
        if k not in t:
            problems.append(f"{tid}: track.json missing {k}")
    if not (t.get("video_url") or t.get("image_url")):
        problems.append(f"{tid}: neither video_url nor image_url")
    files = [s["file"] for s in t["stems"]] + [t[k] for k in ("timeline", "controls", "mix") if t.get(k)]
    if t.get("control"):
        files.append(t["control"]["file"])
    for f in files:
        if not (d / f).exists():
            problems.append(f"{tid}: missing {f}")
    for s in t["stems"]:
        if s.get("group") not in GROUPS:
            problems.append(f"{tid}: stem {s['name']} has bad group {s.get('group')}")
    lens = {f: mp3_seconds(d / f) for f in files if f.endswith(".mp3") and (d / f).exists()}
    known = [v for v in lens.values() if v]
    if known and max(known) - min(known) > 0.1:
        problems.append(f"{tid}: stem lengths differ: {lens}")
    tl = strict(d / t["timeline"])
    voices = {n["voice"] for n in tl}
    for n in tl:
        v = n.get("value")
        if v is not None and not (isinstance(v, (int, float)) and math.isfinite(v)):
            problems.append(f"{tid}: bad value {v!r}")
            break
    for v in t.get("lead_voices") or []:
        if v not in voices:
            problems.append(f"{tid}: lead voice {v} not in timeline")
    vg = t.get("voice_groups") or {}
    if vg and set(vg) != voices:
        problems.append(f"{tid}: voice_groups keys != timeline voices ({sorted(voices ^ set(vg))})")
    c = strict(d / t["controls"])
    for k in ("geophony", "anthrophony", "biophony", "brightness"):
        if len(c["curves"].get(k, [])) != len(c["t"]):
            problems.append(f"{tid}: controls curve {k} length")
    size = sum(f.stat().st_size for f in d.iterdir())
    groups = sorted({s["group"] for s in t["stems"]})
    f = t.get("fidelity") or {}
    return {"id": tid, "files": len(files), "mb": round(size / 1e6, 2), "groups": groups,
            "seconds": round(max(known), 2) if known else None, "r2": f.get("r2"), "mix_r2": f.get("mix_r2"),
            "control_r2": f.get("control_r2"), "media": "image" if t.get("image_url") and not t.get("video_url") else "video"}


def main() -> int:
    problems: list[str] = []
    idx = strict(WEB / "tracks" / "index.json")
    tracks = idx["tracks"] if isinstance(idx, dict) else idx
    rows = [check_track(str(x["id"] if isinstance(x, dict) else x), problems) for x in tracks]
    for r in rows:
        print(json.dumps(r))
    total = sum(f.stat().st_size for f in (WEB / "tracks").rglob("*") if f.is_file())
    print(f"web/tracks total {total / 1e6:.1f} MB, {len(rows)} tracks, order {[r['id'] for r in rows]}")
    node = shutil.which("node")
    for js in sorted((WEB / "js").glob("*.js")):
        if node:
            r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
            if r.returncode:
                problems.append(f"node --check {js.name}: {r.stderr.strip()}")
        else:
            print("node not found: skipped JS syntax check")
            break
    print("problems:" if problems else "OK: no problems", *problems, sep="\n  ")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
