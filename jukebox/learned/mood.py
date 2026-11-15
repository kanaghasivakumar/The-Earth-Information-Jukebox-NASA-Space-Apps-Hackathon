"""Video frames -> style weights over a small fixed vocabulary of music prompts ("mood").

    python -m jukebox.learned.mood 5313            # writes outputs/5313_mood/{weights.json, weights.png}
    python -m jukebox.learned.mood 3912 --tau 0.02 --beta 2.0

The mood layer is ATMOSPHERE ONLY. It never sets a pitch and never edits a data note; the data
melody stays fixed and is checked again on the final mix (magenta_mood.py).

Method
1. Frames: decoded from the cached SVS movie at `fps` (default 2) with ffmpeg; movie time == track time
   for the three pilot tracks (5209: 30 s @30p, 5313: 43.3 s @30p, 3912: 66.5 s; checked in the
   builders' reports).
2. Each anchor has two texts: `music` (sent to Magenta RT's MusicCoCa) and `visual` (what a frame
   that should evoke it looks like). SigLIP (google/siglip-base-patch16-224, text-aligned) scores
   every frame against the `visual` texts; the cosine similarities are partly de-biased per anchor
   (subtract `debias` x the anchor's mean over the track, so a word that always scores high does not
   win everywhere) and passed through softmax(cos / tau).  -> `w_image` (vision only)
3. Data blend: anchors may be linked to a controls.json curve (or a data series), e.g. 5313
   city <- anthrophony, forest <- biophony. logit_k += beta * curve_k(t). -> `w` (used for music)
4. Exponential smoothing (`smooth_s`) so the texture does not jump frame to frame.
Correlations of BOTH w_image and w with the data curves are reported, so the vision part is
judged on its own (the blend correlates with data by construction).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WT = ROOT.parent                                    # /projects/e32706/omb8654/jukebox-wt
INTEGRATE = WT / "integrate" / "outputs"
HF_HOME = "/projects/e32706/omb8654/hf_cache"
SIGLIP_ID = "google/siglip-base-patch16-224"

# movie sources (read-only caches in the sibling worktrees; ingest.py layout data/raw/<id>/)
MOVIES = {
    "5209": WT / "datasets-continuous/data/raw/5209/GISTEMP_Zonal_1880-2023_C.mp4",
    "5313": WT / "form/data/raw/5313/nightlights_flat_series_and_change_wZooms_13_EIC_2160p30_h265.mp4",
    "3912": WT / "voices/data/raw/3912/ecco2_sst_flow.webmhd.webm",
}

# ------------------------------------------------------------------ vocabularies
# name: (music prompt for MusicCoCa, visual description for SigLIP, data link or None)
# data link: (curve name, sign); curve names are controls.json curves or the derived series below.
VOCAB = {
    "ocean": {   # 3912 sea surface temperature + currents
        "deep":    ("deep ocean ambient drone", "dark blue deep ocean water", ("geophony", +1)),
        "tropic":  ("warm tropical marimba", "warm red and orange tropical sea", ("warm", +1)),
        "icy":     ("icy glassy pads", "cold icy blue and white polar water", ("warm", -1)),
        "flow":    ("gentle flowing water ambient", "swirling ocean currents and eddies", None),
        "storm":   ("storm percussion", "turbulent stormy chaotic swirls", None),
        "solemn":  ("solemn orchestral strings", "a dark empty map", None),
    },
    "lights": {  # 5313 night lights 2012-2023
        "city":    ("night city synthwave", "bright city lights at night seen from space", ("anthrophony", +1)),
        "forest":  ("forest birdsong ambient", "dark forest and countryside at night", ("biophony", +1)),
        "dark":    ("dark empty silence", "a black empty night with no lights", None),
        "glow":    ("warm glowing electric piano", "warm orange glow of lights", None),
        "change":  ("tense pulsing electronic", "a map of change in purple and orange", None),
        "solemn":  ("solemn orchestral strings", "a calm map of the earth", None),
    },
    "climate": { # 5209 zonal temperature anomalies 1880-2023
        "icy":     ("icy glassy pads", "cold blue colors", ("anomaly", -1)),
        "warm":    ("warm tropical marimba", "warm red and orange colors", ("anomaly", +1)),
        "heat":    ("tense rising cinematic drums", "hot dark red heat", ("anomaly", +1)),
        "calm":    ("calm cool ambient", "pale white and light blue colors", None),
        "solemn":  ("solemn orchestral strings", "a chart with lines on a dark background", None),
        "wind":    ("airy wind ambient", "an empty grey chart", None),
    },
}
FAMILY = {"3912": "ocean", "5313": "lights", "5209": "climate"}


# ------------------------------------------------------------------ frames
def ffmpeg() -> str:
    f = shutil.which("ffmpeg") or shutil.which("ffmpeg", path=str(Path(sys.executable).parent))
    if not f:
        raise SystemExit("ffmpeg not found")
    return f


def decode(path: Path, fps: float, width: int = 448) -> np.ndarray:
    ff = ffmpeg()
    probe = subprocess.run([ff.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                           check=True, capture_output=True, text=True).stdout.strip()
    w0, h0 = map(int, probe.split(",")[:2])
    h = int(round(width * h0 / w0 / 2)) * 2
    raw = subprocess.run([ff, "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:{h}:flags=area",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, width, 3)


def load_frames(tid: str, out: Path, fps: float) -> tuple[np.ndarray, np.ndarray]:
    cache = out / f"frames_{fps:g}fps.npy"
    if cache.exists():
        fr = np.load(cache)
    else:
        fr = decode(MOVIES[tid], fps)
        np.save(cache, fr)
    # sample at the middle of each 1/fps interval (ffmpeg's fps filter picks the nearest frame)
    return fr, (np.arange(len(fr)) + 0.5) / fps


# ------------------------------------------------------------------ SigLIP
class Siglip:
    def __init__(self, model_id: str = SIGLIP_ID):
        os.environ.setdefault("HF_HOME", HF_HOME)
        import torch
        from transformers import AutoModel, AutoProcessor
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        self.torch = torch
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.id = model_id

    def text(self, texts: list[str]) -> np.ndarray:
        with self.torch.inference_mode():
            t = self.proc(text=texts, padding="max_length", return_tensors="pt")
            e = self.model.get_text_features(**t)
            e = getattr(e, "pooler_output", e)
        e = e.float().numpy()
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def images(self, frames: np.ndarray, batch: int = 16) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(frames), batch):
                x = self.proc(images=list(frames[i:i + batch]), return_tensors="pt")
                e = self.model.get_image_features(**x)
                e = getattr(e, "pooler_output", e)
                out.append(e.float().numpy())
        e = np.concatenate(out)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


# ------------------------------------------------------------------ data curves
def data_curves(tid: str, t: np.ndarray, src: Path) -> dict[str, np.ndarray]:
    """controls.json curves + track-specific data series, all resampled to t and scaled to 0..1."""
    c = json.loads((src / "controls.json").read_text())
    tc = np.asarray(c["t"], float)
    out = {k: np.interp(t, tc, np.nan_to_num(np.asarray(v, float))) for k, v in c["curves"].items()
           if np.ptp(np.asarray(v, float)) > 1e-6}
    tl = json.loads((src / "timeline_data.json").read_text())
    lead = {"5209": "lead global", "5313": "years", "3912": "lead"}[tid]
    ln = sorted([n for n in tl if n["voice"] == lead and n.get("value") is not None], key=lambda n: n["start"])
    lt = np.array([0.5 * (n["start"] + n["end"]) for n in ln])
    lv = np.array([n["value"] for n in ln], float)
    series = np.interp(t, lt, lv)
    out[{"5209": "anomaly", "5313": "light", "3912": "warm"}[tid]] = series
    if tid == "3912":   # seasons: northern minus southern mid-latitude band SST (features.npz zonal, 6 bands N->S)
        z = np.load(src / "features.npz")
        zt, zon = z["timestamps"].astype(float), z["zonal"].astype(float)
        nh, sh = np.nanmean(zon[:, 0:3], 1), np.nanmean(zon[:, 3:6], 1)
        ok = np.isfinite(nh) & np.isfinite(sh)
        out["north_minus_south"] = np.interp(t, zt[ok], (nh - sh)[ok])
    return {k: (v - v.min()) / (np.ptp(v) + 1e-12) for k, v in out.items()}


# ------------------------------------------------------------------ weights
def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def ema(w: np.ndarray, dt: float, tau_s: float) -> np.ndarray:
    if tau_s <= 0:
        return w
    a = 1 - np.exp(-dt / tau_s)
    out = np.empty_like(w)
    cur = w[0]
    for i in range(len(w)):
        cur = cur + a * (w[i] - cur)
        out[i] = cur
    # run backwards too (zero phase): offline, so no lag against the picture
    cur = out[-1]
    for i in range(len(w) - 1, -1, -1):
        cur = cur + a * (out[i] - cur)
        out[i] = cur
    return out / out.sum(1, keepdims=True)


def corr(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def detrend(x, t):
    x = np.asarray(x, float)
    return x - np.polyval(np.polyfit(t, x, 1), t)


def linked_stats(w_img, w, curves, names, vocab, t, segments=None) -> dict:
    """For each data-linked anchor: r (vision only / blended), r after removing a linear trend from
    both (a chart that fills in over time correlates with any trending series), the anchor's
    correlation with time itself, and r inside each named segment of the video."""
    out = {}
    for k, n in enumerate(names):
        dl = vocab[n][2]
        if not (dl and dl[0] in curves):
            continue
        s, c = dl[1], curves[dl[0]]
        f = lambda v: None if v is None else s * v
        d = {"curve": dl[0], "sign": s,
             "r_image_only": f(corr(w_img[:, k], c)), "r_blended": f(corr(w[:, k], c)),
             "r_image_only_detrended": f(corr(detrend(w_img[:, k], t), detrend(c, t))),
             "r_image_only_vs_time": corr(w_img[:, k], t), "r_curve_vs_time": f(corr(c, t))}
        if segments:
            d["segments"] = {}
            for sg in segments:
                m = (t >= sg["start"]) & (t <= sg["end"])
                if m.sum() >= 5:
                    d["segments"][sg["name"]] = {"n": int(m.sum()), "r_image_only": f(corr(w_img[m, k], c[m])),
                                                 "r_blended": f(corr(w[m, k], c[m]))}
        out[n] = d
    return out


def mood_weights(tid: str, *, fps: float = 2.0, tau: float = 0.01, debias: float = 0.7, beta: float = 2.0,
                 smooth_s: float = 1.5, out: Path | None = None, src: Path | None = None, plot: bool = True) -> dict:
    tid = str(tid)
    out = Path(out or ROOT / "outputs" / f"{tid}_mood")
    src = Path(src or INTEGRATE / tid)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames, t = load_frames(tid, out, fps)
    duration = float(json.loads((src / "track.json").read_text())["duration_s"])
    keep = t <= duration + 0.5 / fps
    frames, t = frames[keep], t[keep]
    vocab = VOCAB[FAMILY[tid]]
    names = list(vocab)
    model = Siglip()
    T = model.text([vocab[k][1] for k in names])
    I = model.images(frames)
    cos = I @ T.T                                                   # (n_frames, K)
    cos_d = cos - debias * cos.mean(0, keepdims=True)
    w_img = softmax(cos_d / tau)
    curves = data_curves(tid, t, src)
    link = np.zeros_like(cos)
    for k, n in enumerate(names):
        dl = vocab[n][2]
        if dl and dl[0] in curves:
            link[:, k] = curves[dl[0]] if dl[1] > 0 else 1 - curves[dl[0]]
    w_mix = softmax(cos_d / tau + beta * link)
    dt = 1 / fps
    w_img_s, w_s = ema(w_img, dt, smooth_s), ema(w_mix, dt, smooth_s)
    # correlations: every anchor x every data curve (vision only, and blended)
    cors = {"image_only": {n: {c: corr(w_img_s[:, k], v) for c, v in curves.items()} for k, n in enumerate(names)},
            "blended": {n: {c: corr(w_s[:, k], v) for c, v in curves.items()} for k, n in enumerate(names)}}
    segments = json.loads((src / "track.json").read_text()).get("segments")
    linked = linked_stats(w_img_s, w_s, curves, names, vocab, t, segments)
    res = {"id": tid, "family": FAMILY[tid], "model": SIGLIP_ID, "fps": fps, "tau": tau, "debias": debias,
           "beta": beta, "smooth_s": smooth_s, "duration_s": duration,
           "anchors": [{"name": n, "music": vocab[n][0], "visual": vocab[n][1],
                        "data_link": list(vocab[n][2]) if vocab[n][2] else None} for n in names],
           "t": np.round(t, 3).tolist(),
           "weights": np.round(w_s, 4).tolist(), "weights_image_only": np.round(w_img_s, 4).tolist(),
           "cos": np.round(cos, 4).tolist(),
           "cos_range": {n: [float(cos[:, k].min()), float(cos[:, k].max())] for k, n in enumerate(names)},
           "mean_weight": {n: float(w_s[:, k].mean()) for k, n in enumerate(names)},
           "mean_weight_image_only": {n: float(w_img_s[:, k].mean()) for k, n in enumerate(names)},
           "data_curves": {k: np.round(v, 4).tolist() for k, v in curves.items()},
           "linked_correlations": linked, "correlations": cors, "seconds": round(time.time() - t0, 1),
           "frames": int(len(frames)), "frame_hw": list(frames.shape[1:3])}
    (out / "weights.json").write_text(json.dumps(res))
    if plot:
        _plot(out, res, w_img_s, w_s, curves, names, frames)
    return res


def _plot(out, res, w_img, w, curves, names, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray(res["t"])
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=False,
                           gridspec_kw={"height_ratios": [1.1, 1.6, 1.6, 1.2]})
    idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    ax[0].imshow(np.concatenate([frames[i] for i in idx], 1)); ax[0].axis("off")
    ax[0].set_title(f"{res['id']}: frames at t = " + ", ".join(f"{t[i]:.0f}" for i in idx) + " s")
    cmap = plt.get_cmap("tab10")
    ax[1].stackplot(t, w_img.T, labels=names, colors=[cmap(i) for i in range(len(names))], alpha=0.85)
    ax[1].set_ylabel("weight"); ax[1].set_title("SigLIP only (vision)"); ax[1].legend(fontsize=7, ncol=6, loc="upper left")
    ax[2].stackplot(t, w.T, labels=names, colors=[cmap(i) for i in range(len(names))], alpha=0.85)
    ax[2].set_ylabel("weight"); ax[2].set_title("blended with data curves (sent to Magenta RT)")
    for k, v in curves.items():
        ax[3].plot(t, v, label=k)
    ax[3].set_ylabel("0..1"); ax[3].set_xlabel("s"); ax[3].legend(fontsize=7, ncol=4); ax[3].set_title("data curves")
    for a in ax[1:]:
        a.set_xlim(t[0], t[-1])
    fig.tight_layout(); fig.savefig(out / "weights.png", dpi=80); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--debias", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--smooth", type=float, default=1.5)
    a = ap.parse_args(argv)
    for tid in a.ids:
        r = mood_weights(tid, fps=a.fps, tau=a.tau, debias=a.debias, beta=a.beta, smooth_s=a.smooth)
        print(json.dumps({"id": tid, "mean_weight_image_only": r["mean_weight_image_only"],
                          "linked": r["linked_correlations"], "cos_range": r["cos_range"], "s": r["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
