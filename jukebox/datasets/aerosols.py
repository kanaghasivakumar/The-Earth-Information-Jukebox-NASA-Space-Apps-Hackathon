"""SVS 5669 GEOS Aerosols (2025-08-19 20:00 UTC) -> a west->east sweep with one instrument per aerosol type.

    python -m jukebox.datasets.aerosols [--duration 45]

The page is a single still from the GEOS model, so time = longitude: a playhead sweeps the map
from 180W to 180E in `--duration` seconds. The colour key (Aerosols_cbar.png) is categorical:
black carbon (smoke, orange-red), dust (lilac), sea salt (grey-teal), sulfate (green).

Classification: each pixel's colour is brightness-normalised (the renderer fades plumes toward a
dark background, so chromaticity, not lightness, carries the type) and classified with
`colorbar.Palette` against the four swatches read from the colorbar image plus a neutral class
(grey land, ice, white cloud). Pixels darker than `--min-v` count as clean air.

Music per 2 deg longitude step (0.25 s at 45 s):
- each aerosol type is an instrument; its area fraction in the current column sets the loudness;
  the mean latitude of that type in each hemisphere sets the pitch (north = higher), so a type
  present in both hemispheres plays two notes.
- controls: geophony = sea salt + dust, anthrophony = sulfate + black carbon (combustion and
  industry; GEOS sulfate also includes volcanic and marine sources), biophony = clean air,
  brightness = plume brightness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import catalog
from ..data.colorbar import Palette
from ..proof.fidelity import score
from ..render.synth import SR, midi_hz, render_notes
from ..score.mapping import Note, PitchMap, to_midi
from .events_common import (Stem, export_demo, midi_to_value, norm01, note_pitches, render_curve_voice, smooth,
                            write_track)

PAGE = 5669
IMAGE = "https://svs.gsfc.nasa.gov/vis/a000000/a005600/a005669/Aerosols_2025-08-19T20_00_00_1920x1080.png"
CBAR = "https://svs.gsfc.nasa.gov/vis/a000000/a005600/a005669/Aerosols_cbar.png"
# swatch x-ranges in Aerosols_cbar.png (1500x300), rows 165..210
SWATCHES = {"black carbon": (223, 380), "dust": (518, 676), "sea salt": (819, 977), "sulfate": (1113, 1271)}
INSTR = {  # name: (timbre, low MIDI, gain, GM program)
    "sea salt": ("pad", 38, 0.16, 89),      # low pad (ocean)
    "dust": ("pad", 50, 0.14, 52),          # mid pad (choir aahs in GM)
    "sulfate": ("pad", 62, 0.10, 49),       # high pad (strings in GM)
    "black carbon": ("pluck", 69, 0.20, 12),  # plucked / struck (marimba in GM): sparse smoke plumes
}
LANDMARKS = {"Cape Agulhas": (20.0, -34.8), "Cape Horn": (-67.3, -56.0), "Florida tip": (-80.9, 25.1),
             "Gibraltar": (-5.6, 36.0), "Kamchatka S": (156.7, 51.0), "Tasmania S": (146.8, -43.6)}


def lonlat_grid(h, w):
    """The still is plain equirectangular (checked against coastline landmarks, see overview.png)."""
    lon = -180 + (np.arange(w) + 0.5) / w * 360
    lat = 90 - (np.arange(h) + 0.5) / h * 180
    return lon, lat


def palette_from_cbar(cb_rgb):
    pal = {k: tuple(int(x) for x in np.median(cb_rgb[165:210, a:b].reshape(-1, 3), 0)) for k, (a, b) in SWATCHES.items()}
    return pal


def classify(img, pal, min_v=40, max_de=30, min_chroma=6.0):
    f = img.astype(np.float32)
    pk = f.max(2, keepdims=True)
    norm = np.where(pk > 0, f / np.maximum(pk, 1) * 255, 0).astype(np.uint8)
    nrm = lambda c: tuple(int(x) for x in np.array(c, float) / max(c) * 255)
    classes = {k: nrm(v) for k, v in pal.items()}
    classes["neutral"] = (255, 255, 255)
    P = Palette.from_rgb(classes)
    cls = P.classify(norm, max_delta_e=max_de)
    V = pk[..., 0]
    cls[V < min_v] = -1
    # grey land / ice / cloud: too little colour to be any aerosol (sea salt is the palest swatch, chroma ~12)
    from ..data.colorbar import to_lab
    lab = to_lab(img)
    cls[(np.hypot(lab[..., 1], lab[..., 2]) < min_chroma) & (cls >= 0)] = P.names.index("neutral")
    return cls, P.names, V


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--step-deg", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=960, help="analysis width (image is 1920x1080)")
    ap.add_argument("--min-v", type=int, default=40)
    ap.add_argument("--min-frac", type=float, default=0.03, help="hemisphere fraction below which a type rests")
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--demo", type=Path, default=Path("demo"))
    a = ap.parse_args(argv)
    out = a.out / str(PAGE)
    out.mkdir(parents=True, exist_ok=True)
    d = a.data / str(PAGE)
    from ..data.ingest import download
    img_p = download(IMAGE, d / Path(IMAGE).name)
    cb_p = download(CBAR, d / Path(CBAR).name)
    page = catalog.fetch(PAGE, a.data.parent / "catalog")
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)[..., ::-1]
    cb = cv2.imread(str(cb_p), cv2.IMREAD_COLOR)[..., ::-1]
    H0, W0 = img.shape[:2]
    small = cv2.resize(img, (a.width, int(round(a.width * H0 / W0))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    pal = palette_from_cbar(cb)
    cls, names, V = classify(small, pal, a.min_v)
    types = list(SWATCHES)
    lon, lat = lonlat_grid(h, w)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, w))

    # sweep steps
    edges = np.arange(-180, 180 + 1e-9, a.step_deg)
    nstep = len(edges) - 1
    dt_step = a.duration / nstep
    col_of = np.clip(((lon + 180) / 360 * nstep).astype(int), 0, nstep - 1)
    frac = np.zeros((len(types), nstep))
    hemi = np.zeros((len(types), 2, nstep))      # fraction per hemisphere (N, S)
    hlat = np.full((len(types), 2, nstep), np.nan)  # mean latitude per hemisphere
    bright = np.zeros(nstep)
    north = (lat > 0)[:, None] * np.ones((1, w), bool)
    for s in range(nstep):
        cols = col_of == s
        A = area[:, cols]
        tot = A.sum()
        bright[s] = float((V[:, cols] * A).sum() / tot)
        for k, nme in enumerate(types):
            m = cls[:, cols] == names.index(nme)
            frac[k, s] = (A * m).sum() / tot
            for hh, hm in enumerate((north[:, cols], ~north[:, cols])):
                mm = m & hm
                hemi[k, hh, s] = (A * mm).sum() / (A * hm).sum()
                if mm.any():
                    L = (lat[:, None] * np.ones((1, cols.sum())))[mm]
                    hlat[k, hh, s] = float(np.average(L, weights=A[mm]))

    # notes
    notes: list[Note] = []
    pms = {}
    for k, nme in enumerate(types):
        timbre, low, gain, prog = INSTR[nme]
        pm = PitchMap(-70, 80, low=low, span=24, root=7, scale="major_pent")  # G major pentatonic, 2 octaves
        pms[nme] = pm
        fmax = max(1e-9, hemi[k].max())
        for hh, hname in enumerate(("north", "south")):
            prev = None
            for s in range(nstep):
                f_ = hemi[k, hh, s]
                if f_ < a.min_frac or not np.isfinite(hlat[k, hh, s]):
                    prev = None
                    continue
                p = int(pm(hlat[k, hh, s])[0])
                vel = int(np.clip(35 + 90 * np.sqrt(f_ / fmax), 30, 127))
                t0 = s * dt_step
                lon_c = float((edges[s] + edges[s + 1]) / 2)
                # pads tie repeated pitches; plucked / struck types re-strike every step so loudness tracks the data
                if timbre == "pad" and prev is not None and prev.pitch == p and abs(prev.end - t0) < 1e-6 and abs(prev.velocity - vel) < 12:
                    prev.end = t0 + dt_step
                    continue
                prev = Note(t0, t0 + dt_step, p, vel, nme, float(f_), float(np.clip(lon_c / 180, -1, 1)),
                            {"lon": lon_c, "lat": round(float(hlat[k, hh, s]), 2), "hemisphere": hname,
                             "metric": f"{nme} area fraction of the {hname}ern half of this 2-deg column"})
                notes.append(prev)
    # pan: the playhead moves west->east, so every note is panned to its own longitude, which sweeps
    # the stereo field left->right across the piece.

    # render one stem per type
    stems, audio = [], {}
    for nme in types:
        timbre, low, gain, prog = INSTR[nme]
        audio[nme] = render_notes([n for n in notes if n.voice == nme], a.duration, timbre, gain=gain)
    t_ctrl = (np.arange(nstep) + 0.5) * dt_step
    ti = lambda nm: types.index(nm)
    geo = norm01(smooth(frac[ti("sea salt")] + frac[ti("dust")], 5), 1, 99)
    anth = norm01(smooth(frac[ti("sulfate")] + frac[ti("black carbon")], 5), 1, 99)
    load = frac.sum(0)
    bio = np.clip(1 - norm01(smooth(load, 5), 1, 99), 0, 1)
    brt = norm01(smooth(bright, 3), 1, 99)
    curves = {"geophony": geo, "anthrophony": anth, "biophony": bio, "brightness": brt}
    v_geo = render_curve_voice("geophony", t_ctrl, geo, a.duration, seed=21)
    v_an = render_curve_voice("anthrophony", t_ctrl, anth, a.duration, seed=22)
    v_bio = render_curve_voice("biophony", t_ctrl, bio, a.duration, seed=23)
    for nme in types:
        stems.append(Stem(nme.replace(" ", "-"), "bands", audio[nme], 1.0))
    stems += [Stem("wind-surf", "geophony", v_geo, 0.25), Stem("city", "anthrophony", v_an, 0.2),
              Stem("chorus", "biophony", v_bio, 0.2)]

    # proof: (1) loudness of each instrument stem follows its column fraction;
    # (2) pitch at note onsets maps back to latitude (single-voice stretches only)
    import librosa
    fid = {}
    for k, nme in enumerate(types):
        mono = audio[nme].mean(1)
        rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
        tr = librosa.times_like(rms, sr=SR, hop_length=512)
        env = np.array([rms[(tr >= s * dt_step) & (tr < (s + 1) * dt_step)].mean() for s in range(nstep)])
        r_loud = float(np.corrcoef(env, hemi[k].max(0))[0, 1]) if env.std() > 0 else float("nan")
        vn = [n for n in notes if n.voice == nme]
        solo = [n for n in vn if not any(o is not n and o.start < n.end and n.start < o.end for o in vn)]
        if len(solo) >= 5:
            pm_ = pms[nme]
            mp = note_pitches(audio[nme], solo, fmin=float(midi_hz(pm_.low - 1)), fmax=float(midi_hz(pm_.low + pm_.span + 1)), win=0.2)
            est = midi_to_value(mp, pms[nme])
            sc = score(np.array([n.source["lat"] for n in solo]), est)
        else:
            sc = {"n": len(solo)}
        fid[nme] = {"loudness_vs_fraction_r": r_loud, "pitch_vs_latitude": sc, "notes": len(vn)}

    # report
    glob_frac = {nme: float((area * (cls == names.index(nme))).sum() / area.sum()) for nme in types}
    peak_lon = {nme: float((edges[np.argmax(frac[k])] + edges[np.argmax(frac[k]) + 1]) / 2) for k, nme in enumerate(types)}
    title = page.get("title", "GEOS Aerosols")
    desc = ("A single snapshot of the GEOS model's aerosols on 19 August 2025, played as a sweep from west to east. "
            "Each aerosol type is its own instrument: sea salt is a low pad, dust a middle pad, sulfate a high pad and "
            "smoke (black carbon) a plucked note. An instrument gets louder where its type covers more of the map at that longitude, "
            "and higher where it sits further north; a type in both hemispheres plays two notes. "
            f"Listen for the Saharan dust over the Atlantic and the African smoke plume, loudest near "
            f"{abs(peak_lon['dust']):.0f} deg {'W' if peak_lon['dust'] < 0 else 'E'} and {abs(peak_lon['black carbon']):.0f} deg "
            f"{'W' if peak_lon['black carbon'] < 0 else 'E'}.")
    legend = [
        {"voice": "low pad (sea salt)", "meaning": "sea salt (grey-teal); loudness = share of the column, pitch = latitude"},
        {"voice": "mid pad (dust)", "meaning": "mineral dust (lilac); loudness = share of the column, pitch = latitude"},
        {"voice": "high pad (sulfate)", "meaning": "sulfate (green); loudness = share, pitch = latitude"},
        {"voice": "pluck (black carbon)", "meaning": "smoke / black carbon (orange-red); loudness = share, pitch = latitude"},
        {"voice": "wind and surf (geophony)", "meaning": "sea salt + dust share"},
        {"voice": "city hum (anthrophony)", "meaning": "sulfate + black carbon share"},
        {"voice": "birds and insects (biophony)", "meaning": "clean air (little aerosol in the column)"},
    ]
    report = {
        "id": PAGE, "title": title, "image": IMAGE, "colorbar": CBAR, "palette_rgb": pal,
        "classifier": {"method": "brightness-normalised colour -> nearest swatch (CIELAB, Palette.classify)",
                       "max_delta_e": 30, "min_v": a.min_v, "min_chroma": 6.0, "extra_class": "neutral (grey land / ice / cloud)",
                       "analysis_size": [w, h]},
        "class_fractions_global": glob_frac,
        "unclassified_or_clean_fraction": float((area * (cls < 0)).sum() / area.sum()),
        "neutral_fraction": float((area * (cls == names.index("neutral"))).sum() / area.sum()),
        "peak_longitude": peak_lon,
        "sweep": {"lon_start": -180, "lon_end": 180, "step_deg": a.step_deg, "step_s": dt_step, "duration_s": a.duration},
        "projection": "equirectangular, lon -180..180 left->right, lat 90..-90 top->bottom (landmark check in overview.png)",
        "pitchmap": {k: v.describe() for k, v in pms.items()},
        "notes": {nme: sum(1 for n in notes if n.voice == nme) for nme in types},
        "fidelity": fid,
    }
    to_midi(notes, out / "score.mid", programs={k: v[3] for k, v in INSTR.items()})
    write_track(out, page_id=PAGE, title=title, video_url=IMAGE, duration_s=a.duration,
                units="aerosol type share of each 2-degree longitude column (0..1); latitude in degrees",
                description=desc, legend=legend, stems=stems, notes=notes, t_ctrl=t_ctrl, curves=curves,
                fidelity=fid, report=report,
                data_source={"kind": "pixels", "image": IMAGE, "colorbar": CBAR,
                             "underlying": "NASA GEOS model aerosols (GEOS-FP), 2025-08-19T20:00Z",
                             "scale": "categorical; brightness used only as a clean-air threshold"},
                extra={"media_type": "image", "image_url": IMAGE,
                       "sweep": {"lon_start": -180, "lon_end": 180, "duration_s": a.duration,
                                 "x_of_lon": "x = (lon + 180) / 360 * image_width"}})
    plot(out, small, cls, names, types, pal, frac, edges, t_ctrl, curves, lon, lat)
    export_demo(out, a.demo / str(PAGE), ["mix"])
    print(json.dumps({k: report[k] for k in ("palette_rgb", "class_fractions_global", "unclassified_or_clean_fraction",
                                             "peak_longitude", "notes", "fidelity")}, indent=1))


def plot(out, small, cls, names, types, pal, frac, edges, t_ctrl, curves, lon, lat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 8.5))
    ext = [-180, 180, -90, 90]
    ax[0, 0].imshow(small, extent=ext, aspect="auto")
    for n, (lo, la) in LANDMARKS.items():
        ax[0, 0].plot(lo, la, "c+", ms=12); ax[0, 0].text(lo + 2, la, n, color="c", fontsize=7)
    ax[0, 0].set_title("still + landmark check (cyan + should sit on the coast)")
    vis = np.zeros(small.shape, np.uint8)
    for nme in types:
        vis[cls == names.index(nme)] = pal[nme]
    vis[cls == names.index("neutral")] = (70, 70, 70)
    ax[0, 1].imshow(vis, extent=ext, aspect="auto"); ax[0, 1].set_title("classified (palette from Aerosols_cbar.png)")
    x = (edges[:-1] + edges[1:]) / 2
    for k, nme in enumerate(types):
        ax[1, 0].plot(x, frac[k], color=np.array(pal[nme]) / 255, lw=1.5, label=nme)
    ax[1, 0].legend(); ax[1, 0].set_xlabel("longitude (sweep position)"); ax[1, 0].set_title("area fraction per 2 deg column = loudness")
    for k_, c in curves.items():
        ax[1, 1].plot(t_ctrl, c, label=k_)
    ax[1, 1].legend(); ax[1, 1].set_xlabel("s"); ax[1, 1].set_title("controls")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=95); plt.close(fig)


if __name__ == "__main__":
    main()
