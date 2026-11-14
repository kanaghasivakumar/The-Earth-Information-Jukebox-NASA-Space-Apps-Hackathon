"""Download an SVS movie and decode it to a small frame stack.

    python -m jukebox.data.ingest 3912 --fps 2 --width 256 --out data/raw
Writes data/raw/<id>/frames.npy (T, H, W, 3) uint8, timestamps.npy, meta.json.
Requires ffmpeg on PATH (module load ffmpeg on Quest, or the conda env).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .catalog import UA, fetch, smallest_movie


def tool(name: str) -> str:
    """ffmpeg/ffprobe from PATH, else from the active Python env's bin dir."""
    found = shutil.which(name) or shutil.which(name, path=str(Path(sys.executable).parent))
    if not found:
        raise SystemExit(f"{name} not found: load an ffmpeg module or install it in the env")
    return found


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout.strip()
    w, h = map(int, out.split(",")[:2])
    return w, h


def decode(path: Path, fps: float, width: int) -> np.ndarray:
    w0, h0 = probe_size(path)
    height = int(round(width * h0 / w0 / 2)) * 2
    cmd = [tool("ffmpeg"), "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:{height}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def ingest(page_id: int, out: Path, fps: float = 2.0, width: int = 256, url: str | None = None) -> Path:
    page = fetch(page_id, out.parent / "catalog")
    movie = {"url": url, "filename": Path(url).name} if url else smallest_movie(page)
    if not movie:
        raise SystemExit(f"{page_id}: no movie found")
    d = out / str(page_id)
    src = download(movie["url"], d / movie["filename"])
    frames = decode(src, fps, width)
    np.save(d / "frames.npy", frames)
    np.save(d / "timestamps.npy", np.arange(len(frames)) / fps)
    (d / "meta.json").write_text(json.dumps({"id": page_id, "title": page.get("title"), "movie": movie,
                                             "fps": fps, "shape": list(frames.shape)}, indent=1))
    print(f"{page_id}: {page.get('title')} -> {frames.shape} from {movie['filename']}")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", type=int, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--url", default=None, help="override movie URL (single id)")
    a = ap.parse_args()
    for pid in a.ids:
        ingest(pid, a.out, a.fps, a.width, a.url)
