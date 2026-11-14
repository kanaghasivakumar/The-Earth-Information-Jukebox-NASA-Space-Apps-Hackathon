"""Locate (and if needed download) the General MIDI SoundFont used by `render.fluid`.

The .sf2 lives outside git, under /projects/e32706/omb8654/assets/ by default
(override with $JUKEBOX_SF2 or $JUKEBOX_ASSETS). See render/SOUNDFONT.md for the license.

    python -m jukebox.render.soundfont          # download FluidR3_GM.sf2 if missing, print its path
"""
from __future__ import annotations

import hashlib
import io
import lzma
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

ASSETS = Path(os.environ.get("JUKEBOX_ASSETS", "/projects/e32706/omb8654/assets"))
SF2_NAME = "FluidR3_GM.sf2"

# Debian's fluid-soundfont-gm package: FluidR3_GM by Frank Wen et al., MIT license
# (the package's copyright file is extracted next to the .sf2 as FluidR3_GM.COPYING).
DEB_URL = "http://deb.debian.org/debian/pool/main/f/fluid-soundfont/fluid-soundfont-gm_3.1-6_all.deb"
SF2_SHA256 = "74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0"


def find_sf2() -> Path | None:
    """Return a usable .sf2 path, or None (callers then fall back to render.synth)."""
    env = os.environ.get("JUKEBOX_SF2")
    cands = [Path(env)] if env else []
    cands += [ASSETS / SF2_NAME, Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
              Path("/usr/share/soundfonts/FluidR3_GM.sf2")]
    for p in cands:
        if p.is_file() and p.stat().st_size > 1_000_000:
            return p
    return None


def _ar_members(blob: bytes):
    """Minimal reader for the `ar` archive format used by .deb files."""
    assert blob[:8] == b"!<arch>\n", "not an ar archive"
    i = 8
    while i + 60 <= len(blob):
        hdr = blob[i:i + 60]
        name = hdr[:16].decode().strip().rstrip("/")
        size = int(hdr[48:58].decode().strip())
        yield name, blob[i + 60:i + 60 + size]
        i += 60 + size + (size % 2)


def download(dest_dir: Path = ASSETS, url: str = DEB_URL, verify: bool = True) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / SF2_NAME
    if dest.exists():
        return dest
    print(f"downloading {url} ...")
    with urllib.request.urlopen(url, timeout=300) as r:
        blob = r.read()
    data = next(b for n, b in _ar_members(blob) if n.startswith("data.tar"))
    if n_is_xz := data[:6] == b"\xfd7zXZ\x00":
        data = lzma.decompress(data)
    mode = "r:" if n_is_xz else "r:*"
    with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as tf:
        for m in tf.getmembers():
            base = Path(m.name).name
            if base == SF2_NAME and m.isfile():
                tmp = dest.with_suffix(".part")
                with tf.extractfile(m) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out)
                if verify:
                    h = hashlib.sha256(tmp.read_bytes()).hexdigest()
                    if h != SF2_SHA256:
                        tmp.unlink()
                        raise RuntimeError(f"sha256 mismatch for {SF2_NAME}: {h}")
                tmp.rename(dest)
            elif base == "copyright" and m.isfile():
                (dest_dir / "FluidR3_GM.COPYING").write_bytes(tf.extractfile(m).read())
    if not dest.exists():
        raise RuntimeError(f"{SF2_NAME} not found in {url}")
    return dest


def ensure_sf2() -> Path | None:
    """find_sf2(), downloading once if nothing is installed. Never raises."""
    p = find_sf2()
    if p is None:
        try:
            p = download()
        except Exception as e:  # offline node etc.
            print(f"[soundfont] download failed ({e}); falling back to render.synth")
            return None
    return p


if __name__ == "__main__":
    print(download() if find_sf2() is None else find_sf2())
