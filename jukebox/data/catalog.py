"""SVS API helpers: fetch a page's metadata and pick usable media.

    python -m jukebox.data.catalog 3912 --out data/catalog
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

API = "https://svs.gsfc.nasa.gov/api/{}"
UA = {"User-Agent": "earth-jukebox"}


def fetch(page_id: int, cache: Path | None = None) -> dict:
    if cache:
        f = cache / f"{page_id}.json"
        if f.exists():
            return json.loads(f.read_text())
    with urllib.request.urlopen(urllib.request.Request(API.format(page_id), headers=UA), timeout=60) as r:
        data = json.loads(r.read())
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{page_id}.json").write_text(json.dumps(data))
    return data


def media(page: dict) -> list[dict]:
    out = []
    for g in page.get("media_groups") or []:
        for it in g.get("items") or []:
            inst = it.get("instance") or {}
            if not inst.get("url"):
                continue
            size = re.search(r"\[([\d.]+)\s*(KB|MB|GB)\]", inst.get("description") or "")
            mb = None
            if size:
                mb = float(size.group(1)) * {"KB": 1e-3, "MB": 1, "GB": 1e3}[size.group(2)]
            out.append({
                "type": inst.get("media_type"), "filename": inst.get("filename"),
                "url": inst["url"].replace("http://", "https://"),
                "w": inst.get("width"), "h": inst.get("height"), "mb": mb,
                "group": g.get("title"),
            })
    return out


def smallest_movie(page: dict, min_width: int = 600) -> dict | None:
    movies = [m for m in media(page) if m["type"] == "Movie" and (m["filename"] or "").lower().endswith((".mp4", ".m4v", ".webm", ".mov"))]
    sized = [m for m in movies if (m["w"] or 0) >= min_width] or movies
    return min(sized, key=lambda m: (m["mb"] if m["mb"] is not None else 1e9)) if sized else None


def colorbars(page: dict) -> list[dict]:
    return [m for m in media(page) if m["type"] == "Image" and re.search(r"c(olor)?bar", (m["filename"] or "").lower())]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", type=int, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("data/catalog"))
    a = ap.parse_args()
    for pid in a.ids:
        p = fetch(pid, a.out)
        print(pid, p.get("title"))
        print("  movie:", smallest_movie(p))
        for c in colorbars(p):
            print("  colorbar:", c["filename"], c["url"])
