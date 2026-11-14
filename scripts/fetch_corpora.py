"""Fetch the three-voice sound corpora (CC0 / CC-BY / public domain only).

    python scripts/fetch_corpora.py                       # everything
    python scripts/fetch_corpora.py --groups biophony     # one voice
    python scripts/fetch_corpora.py --attribution-only    # rewrite ATTRIBUTION.md from manifest

Output (outside git, default /projects/e32706/omb8654/assets/corpora, or $JUKEBOX_CORPORA):

    <root>/<group>/<slug>.flac     44.1 kHz mono, 16-bit, DC-blocked, loudness-normalised
    <root>/manifest.json           one entry per file: url, author, license, tag, seconds
    <root>/_raw/                   original downloads (cache; safe to delete)
    jukebox/voices/ATTRIBUTION.md  regenerated from the manifest

Sources
  * MBARI Pacific Sound 16 kHz hydrophone (CC-BY 4.0): public S3 bucket, HTTP range
    requests pull a few one-minute windows out of the 4 GB day files.
  * Wikimedia Commons: the licence is read from the Commons API at fetch time and the
    file is skipped unless it is CC0, public domain or CC BY (no SA, no NC, no FAL).
  * U.S. National Park Service Sound Gallery: "The files are in the public domain".
  * One synthetic file (mains hum), generated here and released CC0.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

SR = 44100
UA = {"User-Agent": "earth-jukebox-hackathon/0.1 (corpus fetch script)"}
DEFAULT_ROOT = Path("/projects/e32706/omb8654/assets/corpora")
REPO = Path(__file__).resolve().parents[1]
ATTRIBUTION = REPO / "jukebox" / "voices" / "ATTRIBUTION.md"
ALLOWED = re.compile(r"^(CC0|Public domain|PD|CC BY \d(\.\d)?)", re.I)
NPS_LICENSE = "Public domain (U.S. National Park Service Sound Gallery: 'The files are in the public domain')"
MBARI_BUCKET = "https://pacific-sound-16khz.s3.amazonaws.com"
NPS_AV = "https://www.nps.gov/nps-audiovideo/legacy/mp3/nri/avElement/"
YELL_AV = "https://www.nps.gov/nps-audiovideo/legacy/mp3/imr/avElement/"
YELL_PAGE = "https://www.nps.gov/yell/learn/photosmultimedia/"
YELL_LICENSE = ("Public domain (Yellowstone NP Sound Library: 'The files available here were recorded in "
                "the park and are in the public domain')")

# (group, tag, kind, ref, max_seconds, offset_seconds)
#   kind = commons -> ref is the File: title
#   kind = nps     -> ref is (mp3 name, gallery page slug)
#   kind = yell    -> ref is (mp3 name, page slug) in the Yellowstone Sound Library
#   kind = mbari   -> ref is (YYYYMMDD, hour UTC)
SOURCES = [
    # ---------------------------------------------------------------- geophony
    ("geophony", "hydrophone", "mbari", ("20181115", 4), 60, 0),
    ("geophony", "hydrophone", "mbari", ("20181115", 13), 60, 0),
    ("geophony", "hydrophone", "mbari", ("20190915", 6), 60, 0),
    ("geophony", "hydrophone", "mbari", ("20170120", 12), 60, 0),
    ("geophony", "hydrophone", "mbari", ("20200401", 8), 60, 0),
    ("geophony", "hydrophone", "mbari", ("20161210", 2), 60, 0),
    ("geophony", "thunder", "commons", "File:Thunderstorm after hot summer day 17 minutes 01 of 04.ogg", 150, 0),
    ("geophony", "thunder", "commons", "File:Thunderstorm after hot summer day 17 minutes 03 of 04.ogg", 150, 0),
    ("geophony", "thunder", "commons", "File:Storm thunderbolts.ogg", 150, 0),
    ("geophony", "thunder", "commons", "File:Rhumphries - rbh-thunder-storm.ogg", 150, 0),
    ("geophony", "rain", "commons", "File:Medium shower accompanied by thunders.ogg", 150, 60),
    ("geophony", "rain", "commons", "File:Rain against the window.ogg", 90, 0),
    ("geophony", "rain", "commons", "File:Light Rain Distant Thunder July 5th 2016.wav", 110, 0),
    ("geophony", "wind", "commons", "File:Leaves in the wind (Gravity Sound).wav", 135, 0),
    ("geophony", "wind", "commons", "File:Wind in forest (Gravity Sound).wav", 60, 0),
    ("geophony", "ocean", "nps", ("nri-OceanOlympicNP.mp3", "sounds-ocean"), 120, 0),
    ("geophony", "rain", "nps", ("nri-RainDEPO.mp3", "sounds-rain"), 120, 0),
    ("geophony", "thunder", "nps", ("nri-Thunder.mp3", "sounds-thunder"), 120, 0),
    ("geophony", "water", "nps", ("nri-StreamMUWO.mp3", "sounds-stream"), 120, 0),
    # ------------------------------------------------------------- anthrophony
    ("anthrophony", "traffic", "commons", "File:Two Lane Road in Kelowna at Midnight.flac", 150, 120),
    ("anthrophony", "traffic", "commons", "File:Urban Street on a Rainy Afternoon.flac", 150, 300),
    ("anthrophony", "traffic", "commons", "File:Bundesautobahn A6.flac", 150, 0),
    ("anthrophony", "city", "commons", "File:184809 qubodup first-day-of-urban-spring-in-berlin-2013-04-14.flac", 111, 0),
    ("anthrophony", "city", "commons", "File:Sunday in the center of Mexico City (close to Alameda Park), Schoeps MS Setup.flac", 150, 0),
    ("anthrophony", "engine", "commons", "File:WWS City Bus MAN SG 220 – engine.oga", 94, 0),
    ("anthrophony", "engine", "commons", "File:Train passing by in the Wirikuta Desert (Mexico, SLP).wav", 154, 0),
    ("anthrophony", "hum", "commons", "File:Air conditioner hum 2 (Gravity Sound).wav", 18, 0),
    ("anthrophony", "hum", "commons", "File:Air vent (Gravity Sound).wav", 72, 0),
    ("anthrophony", "hum", "commons", "File:Fan (Gravity Sound).wav", 34, 0),
    ("anthrophony", "hum", "synthetic", "mains_hum_60hz", 90, 0),
    ("anthrophony", "aircraft", "nps", ("nri-JetYOSE.mp3", "sounds-jet"), 120, 0),
    ("anthrophony", "aircraft", "nps", ("nri-PropellerGOGA.mp3", "sounds-propeller"), 120, 0),
    ("anthrophony", "aircraft", "nps", ("nri-HelicopterYOSE.mp3", "sounds-helicopter"), 120, 0),
    ("anthrophony", "engine", "nps", ("nri-BoatNOCA.mp3", "sounds-motorboat"), 120, 0),
    ("anthrophony", "engine", "nps", ("nri-MotorcycleMORA.mp3", "sounds-motorcycle"), 120, 0),
    ("anthrophony", "engine", "nps", ("nri-Snowmobile.mp3", "sounds-snowmobile"), 120, 0),
    ("anthrophony", "engine", "nps", ("nri-RotarySnowPlow.mp3", "sounds-snowplow"), 120, 0),
    # ---------------------------------------------------------------- biophony
    ("biophony", "birds", "commons", "File:Dawnchorus-uk.ogg", 180, 600),
    ("biophony", "birds", "commons", "File:Réveil des oiseaux.ogg", 175, 0),
    ("biophony", "birds", "commons", "File:Dawn chorus at Glencairnie big bungalow Craigmore DM.ogg", 180, 60),
    ("biophony", "birds", "commons", "File:Royal Natal National Park dawn chorus (W1CDR0001146 BD3).ogg", 136, 0),
    ("biophony", "birds", "commons", "File:Reggelirigók.ogg", 150, 0),
    ("biophony", "frogs", "commons", "File:Pseudacris triseriata chorus.wav", 31, 0),
    ("biophony", "insects", "commons", "File:Cicada 20200619.wav", 62, 0),
    ("biophony", "insects", "commons", "File:HyalessaFuscata KR.flac", 145, 0),
    ("biophony", "insects", "commons", "File:Stridulation of Chorthippus albomarginatus.wav", 90, 0),
    ("biophony", "insects", "commons", "File:BioAcoustica 898 1 Pholidoptera griseoaptera 73r1.flac", 90, 0),
    ("biophony", "insects", "nps", ("nri-Cricket.mp3", "sounds-crickets"), 120, 0),
    ("biophony", "insects", "nps", ("nri-GrasshopperYOSE.mp3", "sounds-grasshopper"), 120, 0),
    ("biophony", "insects", "nps", ("nri-BeeYOSE.mp3", "sounds-bee"), 120, 0),
    ("biophony", "frogs", "nps", ("nri-GreenTreeFrogEvergladesNP.mp3", "sounds-green-treefrog"), 120, 0),
    ("biophony", "frogs", "nps", ("nri-PigFrogEvergladesNP.mp3", "sounds-pigfrog"), 120, 0),
    ("biophony", "frogs", "nps", ("nri-CouchsSpadefootToadSaguaroNP.mp3", "cspadefoottoad"), 120, 0),
    ("biophony", "birds", "nps", ("nri-DawnchorusinAmphitheater.mp3", "sounds-dawnchorus_bryce"), 120, 0),
    ("biophony", "birds", "nps", ("nri-HermitThrushYOSE.mp3", "sounds-hermit-thrush"), 120, 0),
    ("biophony", "birds", "nps", ("nri-AmericanRobinYELL.mp3", "sounds-american-robin"), 120, 0),
    ("biophony", "birds", "nps", ("nri-SparrowGRTE.mp3", "sounds-white-crowned-sparrow"), 120, 0),
    ("biophony", "birds", "nps", ("nri-YellowRumpedWarblerGRTE.mp3", "sounds-yellow-rumped-warbler"), 120, 0),
    ("biophony", "birds", "nps", ("nri-ALDERFLYCATCHERSYukonRiver.mp3", "sounds-alder-flycatcher"), 120, 0),
    ("biophony", "birds", "nps", ("nri-Olivesidedflycatcher1.mp3", "sounds-olivesidedflycatcher"), 120, 0),
    ("biophony", "birds", "nps", ("nri-RedbreastedNuthatchYOSE.mp3", "sounds-nuthatch"), 120, 0),
    ("biophony", "birds", "nps", ("nri-CommonPoorwillGRBA.mp3", "sounds-common-poorwill"), 120, 0),
    ("biophony", "birds", "yell", ("yell-YELLDawnChorus.mp3", "sounds-dawnchorus"), 150, 0),
    ("biophony", "birds", "yell", ("yell-YELLSORAVIRAWISN20150617T42.mp3", "sounds-birdchorus"), 150, 0),
    ("biophony", "birds", "yell", ("yell-YELLAMCO20160914T10ms.mp3", "sounds-americancoots"), 150, 120),
    ("biophony", "birds", "yell", ("yell-YELLAMRO20160506SM3.mp3", "sounds-americanrobins"), 150, 0),
    ("biophony", "birds", "yell", ("yell-YELLCOYE20160712T08.mp3", "sounds-commonyellowthroat"), 150, 0),
    ("biophony", "birds", "yell", ("yell-YELLredwingedblackbird20150427.mp3", "sounds-redwingedblackbird"), 150, 0),
    ("biophony", "birds", "yell", ("yell-YELLSavannahSparrow20150617T22.mp3", "sounds-savannahsparrow"), 150, 0),
    ("biophony", "birds", "yell", ("yell-YELLMOBL20150617T26.mp3", "sounds-mountainbluebird"), 150, 0),
    ("biophony", "birds", "yell", ("yell-YELLLAZB20160625SM303143.mp3", "lazulibunting"), 150, 0),
    ("biophony", "birds", "yell", ("yell-YELLCOLO20160913T18ms.mp3", "sounds-commonloon"), 150, 0),
    ("biophony", "frogs", "yell", ("yell-20150427T13BorealfrogSACRWISN.mp3", "sounds-chorusfrogs"), 150, 0),
]


# ------------------------------------------------------------------ helpers
def get(url: str, headers: dict | None = None, timeout: int = 120, tries: int = 6) -> bytes:
    h = dict(UA, **(headers or {}))
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
            if e.code == 429:  # upload.wikimedia.org throttles bursts of originals
                ra = e.headers.get("Retry-After", "")
                time.sleep(int(ra) if ra.isdigit() else 30 * (k + 1))
                continue
        except (urllib.error.URLError, TimeoutError):
            if k == tries - 1:
                raise
        time.sleep(5 * (k + 1))
    raise RuntimeError(url)


def slugify(s: str) -> str:
    s = re.sub(r"^File:", "", s)
    s = re.sub(r"\.[A-Za-z0-9]+$", "", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:60]


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """bytes -> mono float32. libsndfile first; ffmpeg for what it cannot open
    (e.g. Ogg files with a Skeleton stream, which many Commons uploads have)."""
    import soundfile as sf
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return x.mean(1), sr
    except Exception:
        import shutil
        import subprocess
        ff = shutil.which("ffmpeg") or str(Path(sys.executable).with_name("ffmpeg"))
        r = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-map", "0:a:0",
                            "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                           input=data, capture_output=True, check=True)
        return np.frombuffer(r.stdout, np.float32).copy(), SR


def condition(x: np.ndarray, sr: int, max_s: float, offset_s: float) -> np.ndarray:
    """crop -> resample to 44.1k -> DC/rumble high-pass -> loudness normalise -> fade."""
    from math import gcd
    from scipy.signal import butter, resample_poly, sosfiltfilt

    a = int(offset_s * sr)
    if a >= len(x) - sr * 5:  # offset past the end: take from the start
        a = 0
    x = x[a: a + int(max_s * sr)].astype(np.float64)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    x = sosfiltfilt(butter(2, 25, "highpass", fs=SR, output="sos"), x)
    # 90th percentile of 50 ms RMS -> -18 dBFS, then keep the peak under -1 dBFS
    n = int(0.05 * SR)
    fr = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    ref = np.percentile(rms, 90)
    x *= 10 ** (-18 / 20) / max(ref, 1e-9)
    pk = np.abs(x).max()
    if pk > 0.89:
        x *= 0.89 / pk
    f = int(0.02 * SR)
    ramp = np.linspace(0, 1, f)
    x[:f] *= ramp
    x[-f:] *= ramp[::-1]
    return x.astype(np.float32)


# ------------------------------------------------------------------ sources
def commons_info(titles: list[str]) -> dict:
    out = {}
    for i in range(0, len(titles), 40):
        chunk = titles[i: i + 40]
        q = dict(action="query", format="json", titles="|".join(chunk), prop="imageinfo|videoinfo",
                 iiprop="url|size|extmetadata", iiextmetadatafilter="Artist|LicenseShortName|LicenseUrl|Credit",
                 viprop="derivatives")
        r = json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q)))
        norm = {n["to"]: n["from"] for n in r["query"].get("normalized", [])}
        for p in r["query"]["pages"].values():
            if "imageinfo" not in p:
                continue
            ii = p["imageinfo"][0]
            m = ii.get("extmetadata", {})
            ders = {d.get("transcodekey"): d["src"] for d in (p.get("videoinfo") or [{}])[0].get("derivatives", [])}
            out[norm.get(p["title"], p["title"])] = {
                "transcode": ders.get("mp3") or ders.get("ogg"),
                "url": ii["url"], "page": ii["descriptionurl"], "bytes": ii["size"],
                "author": strip_html(m.get("Artist", {}).get("value", "")) or "unknown",
                "license": strip_html(m.get("LicenseShortName", {}).get("value", "")),
                "license_url": m.get("LicenseUrl", {}).get("value", ""),
            }
        time.sleep(1)
    return out


def mbari_window(day: str, hour: int, seconds: float) -> tuple[np.ndarray, int, str]:
    key = f"{day[:4]}/{day[4:6]}/MARS-{day}T000000Z-16kHz.wav"
    url = f"{MBARI_BUCKET}/{key}"
    head = get(url, {"Range": "bytes=0-8191"})
    assert head[:4] == b"RIFF" and head[8:12] == b"WAVE", "not a WAV"
    pos, fmt, data_off = 12, None, None
    while pos + 8 <= len(head):
        cid, size = head[pos: pos + 4], struct.unpack("<I", head[pos + 4: pos + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", head[pos + 8: pos + 24])
        if cid == b"data":
            data_off = pos + 8
            break
        pos += 8 + size + (size & 1)
    _, ch, sr, _, align, bits = fmt
    assert ch == 1 and bits == 24, fmt
    start = data_off + int(hour * 3600 * sr) * align
    n = int(seconds * sr) * align
    raw = np.frombuffer(get(url, {"Range": f"bytes={start}-{start + n - 1}"}), np.uint8)
    raw = raw[: len(raw) // 3 * 3].reshape(-1, 3).astype(np.int32)
    v = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    v = np.where(v >= 1 << 23, v - (1 << 24), v)
    return (v / float(1 << 23)).astype(np.float32), sr, url


def synth_mains_hum(seconds: float, f0: float = 60.0, seed: int = 7) -> np.ndarray:
    """Transformer / power-line hum: 60 Hz family with 120 Hz magnetostriction dominant,
    slow amplitude wander and a little rectifier buzz. Generated, not recorded."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    drift = np.cumsum(rng.standard_normal(n)) / np.sqrt(SR) * 0.002  # ~mHz frequency wander
    ph = 2 * np.pi * (f0 * t + np.cumsum(drift) / SR)
    x = np.zeros(n)
    amps = {1: .35, 2: 1.0, 3: .3, 4: .45, 5: .12, 6: .2, 8: .1, 10: .05, 12: .03}
    for k, a in amps.items():
        wob = 1 + 0.15 * np.sin(2 * np.pi * rng.uniform(.03, .2) * t + rng.uniform(0, 6.3))
        x += a * wob * np.sin(k * ph + rng.uniform(0, 6.3))
    buzz = np.sign(np.sin(2 * ph)) * 0.04
    from scipy.signal import butter, sosfilt
    x += sosfilt(butter(2, [400, 3000], "bandpass", fs=SR, output="sos"), buzz)
    x += 0.01 * rng.standard_normal(n)
    return (x / np.abs(x).max() * 0.5).astype(np.float32)


# ------------------------------------------------------------------ main
def cached_get(raw: Path, url: str, **kw) -> bytes:
    """Download once into <root>/_raw so processing can be re-run (or run on another node)."""
    if raw.exists() and raw.stat().st_size > 0:
        return raw.read_bytes()
    data = get(url, **kw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(data)
    return data


def fetch(root: Path, groups: set[str], force: bool = False, download_only: bool = False,
          pace: float = 8.0, cached_only: bool = False) -> list[dict]:
    """download_only: just fill <root>/_raw (network only, light enough for a login node);
    pace: seconds between Commons downloads (Wikimedia rate-limits bursts with HTTP 429)."""
    import soundfile as sf

    man_path = root / "manifest.json"
    manifest = {m["file"]: m for m in json.loads(man_path.read_text())} if man_path.exists() else {}
    todo = [s for s in SOURCES if s[0] in groups]
    info = commons_info([s[3] for s in todo if s[2] == "commons"])

    for group, tag, kind, ref, max_s, off in todo:
        if kind == "commons":
            slug = slugify(ref)
        elif kind == "nps":
            slug = "nps_" + slugify(ref[0].replace("nri-", ""))
        elif kind == "yell":
            slug = "yell_" + slugify(ref[1].replace("sounds-", ""))
        elif kind == "mbari":
            slug = f"mbari_{ref[0]}_{ref[1]:02d}h"
        else:
            slug = ref
        rel = f"{group}/{slug}.flac"
        dest = root / rel
        if dest.exists() and rel in manifest and not force:
            print(f"[skip] {rel}")
            continue
        if download_only and kind not in ("commons", "nps", "yell"):
            continue
        try:
            if kind == "commons":
                meta = info.get(ref)
                if meta is None:
                    print(f"[miss] {ref}: not on Commons")
                    continue
                if not ALLOWED.match(meta["license"]) or re.search(r"\b(SA|NC|ND)\b|-SA|-NC|-ND", meta["license"]):
                    print(f"[deny] {ref}: licence {meta['license']!r}")
                    continue
                # big originals (> 15 MB WAV/FLAC): take Commons' MP3/Ogg transcode instead; it is
                # a fraction of the size and Wikimedia throttles bulk original downloads (HTTP 429)
                url = meta["url"]
                if meta["bytes"] > 15e6 and meta.get("transcode"):
                    url = meta["transcode"]
                have = sorted((root / "_raw" / group).glob(slug + ".*"))
                raw = have[0] if have else root / "_raw" / group / (slug + Path(urllib.parse.urlparse(url).path).suffix)
                if have and raw.suffix != Path(urllib.parse.urlparse(url).path).suffix:
                    url = meta["url"]  # cached original from an earlier run
                fresh = not raw.exists()
                if fresh and cached_only:
                    print(f"[wait] {rel}: not in _raw yet (run --download-only)")
                    continue
                data = cached_get(raw, url, timeout=90, tries=4)
                if fresh:
                    time.sleep(pace)
                if download_only:
                    print(f"[raw]  {raw.name}  {len(data) / 1e6:.1f} MB")
                    continue
                x, sr = decode(data)
                entry = dict(source="Wikimedia Commons", title=ref[5:], url=meta["page"], download=url,
                             author=meta["author"], license=meta["license"], license_url=meta["license_url"])
            elif kind == "nps":
                url = NPS_AV + ref[0]
                data = cached_get(root / "_raw" / group / (slug + ".mp3"), url)
                if download_only:
                    print(f"[raw]  {slug}.mp3")
                    continue
                x, sr = decode(data)
                page = f"https://www.nps.gov/subjects/sound/{ref[1]}.htm"
                entry = dict(source="NPS Sound Gallery", title=ref[1].replace("sounds-", ""), url=page, download=url,
                             author="U.S. National Park Service", license=NPS_LICENSE, license_url="")
            elif kind == "yell":
                url = YELL_AV + ref[0]
                data = cached_get(root / "_raw" / group / (slug + ".mp3"), url)
                if download_only:
                    print(f"[raw]  {slug}.mp3")
                    continue
                x, sr = decode(data)
                entry = dict(source="NPS Yellowstone Sound Library", title=ref[1].replace("sounds-", ""),
                             url=YELL_PAGE + ref[1] + ".htm", download=url,
                             author="U.S. National Park Service (Yellowstone NP; many with MSU Acoustic Atlas)",
                             license=YELL_LICENSE, license_url="")
            elif kind == "mbari":
                x, sr, url = mbari_window(ref[0], ref[1], max_s)
                entry = dict(source="MBARI Pacific Sound (MARS hydrophone, 891 m, Monterey Bay)",
                             title=f"{ref[0]} {ref[1]:02d}:00 UTC, {max_s:.0f} s window", url=url, download=url,
                             author="Monterey Bay Aquarium Research Institute (MBARI)", license="CC BY 4.0",
                             license_url="https://creativecommons.org/licenses/by/4.0/",
                             cite="Ryan et al. 2016, doi:10.1109/OCEANS.2016.7761363; registry.opendata.aws/pacific-sound")
                off = 0
            else:
                x, sr = synth_mains_hum(max_s), SR
                entry = dict(source="generated by scripts/fetch_corpora.py", title="synthetic 60 Hz mains/transformer hum",
                             url="", download="", author="Earth Jukebox team", license="CC0",
                             license_url="https://creativecommons.org/publicdomain/zero/1.0/")
            y = condition(x, sr, max_s, off)
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dest), y, SR, format="FLAC", subtype="PCM_16")
            entry.update(file=rel, group=group, tag=tag, seconds=round(len(y) / SR, 2), offset_s=off)
            manifest[rel] = entry
            man_path.write_text(json.dumps(sorted(manifest.values(), key=lambda m: m["file"]), indent=1))
            print(f"[ok]   {rel:70s} {entry['seconds']:6.1f}s  {entry['license']}")
        except Exception as e:  # keep going; the manifest records what made it
            print(f"[fail] {rel}: {type(e).__name__}: {e}")
        time.sleep(0.5)
    out = sorted(manifest.values(), key=lambda m: m["file"])
    if not download_only:  # a download-only pass must not clobber a concurrent processing pass
        man_path.write_text(json.dumps(out, indent=1))
    return out


def write_attribution(manifest: list[dict], path: Path = ATTRIBUTION):
    lines = [
        "# Voice corpora: attribution",
        "",
        "Generated by `scripts/fetch_corpora.py` from `<corpora>/manifest.json`. Only CC0, CC BY and",
        "public-domain material is used (no SA / NC / FAL). Files are cropped to the window shown,",
        "resampled to 44.1 kHz mono, high-passed at 25 Hz and loudness-normalised; the grain engine",
        "then slices them into 100–400 ms grains (these are modifications in the CC BY sense).",
        "",
    ]
    for group in ("geophony", "anthrophony", "biophony"):
        rows = [m for m in manifest if m["group"] == group]
        mins = sum(m["seconds"] for m in rows) / 60
        lines += [f"## {group} ({len(rows)} files, {mins:.1f} min)", "",
                  "| file | tag | seconds | title | author | license | source |", "|---|---|---|---|---|---|---|"]
        for m in rows:
            title = m["title"].replace("|", "/")
            src = f"[{m['source']}]({m['url']})" if m["url"] else m["source"]
            lic = f"[{m['license']}]({m['license_url']})" if m.get("license_url") else m["license"]
            lines.append(f"| `{Path(m['file']).name}` | {m['tag']} | {m['seconds']:.0f} "
                         f"(from {m.get('offset_s', 0)} s) | {title} | {m['author'].replace('|', '/')} | {lic} | {src} |")
        lines.append("")
    lines += [
        "MBARI data: please cite Ryan, J. P., et al. (2016), *Ocean Sound in Monterey Bay*, OCEANS 2016,",
        "doi:10.1109/OCEANS.2016.7761363, and the AWS Open Data registry entry `pacific-sound`.",
        "",
        "NPS Sound Gallery (https://www.nps.gov/subjects/sound/gallery.htm): \"The files are in the public",
        "domain and may be downloaded. Enjoy, and please give the appropriate credit.\"",
        "",
        "Yellowstone Sound Library (https://www.nps.gov/yell/learn/photosmultimedia/soundlibrary.htm): \"The",
        "files available here were recorded in the park and are in the public domain. They may be downloaded",
        "and used without limitation; however, please credit the National Park Service where appropriate.\"",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def main(argv=None):
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("JUKEBOX_CORPORA", DEFAULT_ROOT)))
    ap.add_argument("--groups", nargs="*", default=["geophony", "anthrophony", "biophony"])
    ap.add_argument("--force", action="store_true", help="re-download files already present")
    ap.add_argument("--attribution-only", action="store_true")
    ap.add_argument("--download-only", action="store_true",
                    help="only fill <root>/_raw (run this on a login node if compute nodes get HTTP 429)")
    ap.add_argument("--pace", type=float, default=8.0, help="seconds between Commons downloads")
    ap.add_argument("--cached-only", action="store_true",
                    help="process only Commons files already in <root>/_raw (no Commons downloads)")
    a = ap.parse_args(argv)
    a.root.mkdir(parents=True, exist_ok=True)
    if a.attribution_only:
        manifest = json.loads((a.root / "manifest.json").read_text())
    else:
        manifest = fetch(a.root, set(a.groups), a.force, a.download_only, a.pace, a.cached_only)
        if a.download_only:
            return 0
    write_attribution(manifest)
    for g in ("geophony", "anthrophony", "biophony"):
        s = sum(m["seconds"] for m in manifest if m["group"] == g)
        print(f"{g:12s} {s / 60:5.1f} min")


if __name__ == "__main__":
    sys.exit(main())
