"""Constellation for the jukebox navigation view: every frame of every local dataset in one map.

    srun -A p32576 -p short -t 00:30:00 --mem=16G -c 4 \
        /projects/e32706/omb8654/conda/envs/jukebox/bin/python scripts/constellation.py

Embeds data/raw/*/frames.npy (DINOv2-small, handcrafted fallback), projects with UMAP if installed,
else t-SNE (2D) + PCA (3D). Each point also carries its section label from jukebox.form.sections,
so the web view can draw a dataset as a path of coloured sections.
Writes outputs/constellation.json and outputs/constellation.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", "/projects/e32706/omb8654/hf_cache")

from jukebox.form.embed import embed_frames, handcrafted  # noqa: E402
from jukebox.form.sections import analyse  # noqa: E402


def project(X: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    from sklearn.decomposition import PCA
    Z = X - X.mean(0)
    p3 = PCA(3, random_state=seed).fit_transform(Z)
    try:
        import umap  # type: ignore
        p2 = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1).fit_transform(Z)
        how = "umap"
    except ImportError:
        from sklearn.manifold import TSNE
        Z50 = PCA(min(50, Z.shape[1], len(Z) - 1), random_state=seed).fit_transform(Z)
        p2 = TSNE(2, perplexity=min(30, max(5, len(Z) // 20)), init="pca", random_state=seed).fit_transform(Z50)
        how = "tsne"
    return p2, p3, how


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--method", default="auto", choices=["auto", "dinov2", "handcrafted"])
    ap.add_argument("--max-frames", type=int, default=400, help="per dataset (evenly subsampled)")
    a = ap.parse_args(argv)

    rows, Xs, meta = [], [], {}
    method = None
    for d in sorted(p for p in a.data.iterdir() if (p / "frames.npy").exists()):
        frames = np.load(d / "frames.npy", mmap_mode="r")
        t = np.load(d / "timestamps.npy").astype(float)
        idx = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), a.max_frames)).round().astype(int))
        f = np.ascontiguousarray(frames[idx])
        if a.method == "handcrafted":
            emb = handcrafted(f, standardize=False)   # shared space across datasets
        else:
            emb = embed_frames(f, a.method)
            if emb.method == "handcrafted":
                emb = handcrafted(f, standardize=False)
        method = emb.method
        form = analyse(emb.X, t[idx])
        lab = np.empty(len(idx), object)
        for s in form.sections:
            lab[s.start_frame:s.end_frame] = s.label
        m = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        meta[d.name] = {"title": m.get("title"), "frames": int(len(idx)), "duration_s": float(t[-1] - t[0]),
                        "sections": [list(x) for x in form.tuples()],
                        "returns": [s.index for s in form.returns]}
        for j, i in enumerate(idx):
            rows.append({"dataset": d.name, "frame": int(i), "t": float(t[i]), "section": lab[j],
                         "region": int(form.frame_regions[j])})
        Xs.append(emb.X)
        print(d.name, m.get("title"), len(idx), emb.method, form.tuples(), flush=True)

    X = np.concatenate(Xs)
    if method == "handcrafted":   # mixed raw scales: standardise globally (same transform for all datasets)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    p2, p3, how = project(X)
    for r, a2, a3 in zip(rows, p2, p3):
        r["xy"] = [round(float(v), 4) for v in a2]
        r["xyz_pca"] = [round(float(v), 4) for v in a3]
    out = {"embedding": method, "projection_2d": how, "projection_3d": "pca", "datasets": meta, "points": rows,
           "note": "Latent space for navigation/form only; positions never set pitch."}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "constellation.json").write_text(json.dumps(out))
    plot(a.out / "constellation.png", rows, p2, p3, meta, method, how)


def plot(path, rows, p2, p3, meta, method, how):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = list(meta)
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(14, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    for k, name in enumerate(ds):
        sel = np.array([r["dataset"] == name for r in rows])
        tt = np.array([r["t"] for r in rows])[sel]
        lbl = f"{name}: {(meta[name]['title'] or '')[:32]}"
        ax.plot(p2[sel, 0], p2[sel, 1], "-", color=cmap(k), lw=.4, alpha=.5)
        ax.scatter(p2[sel, 0], p2[sel, 1], c=[cmap(k)], s=6 + 18 * (tt - tt.min()) / (np.ptp(tt) + 1e-9), label=lbl)
        ax3.plot(p3[sel, 0], p3[sel, 1], p3[sel, 2], "-", color=cmap(k), lw=.5)
        # mark section starts
        secs = meta[name]["sections"]
        for s0, _, lab in secs:
            j = np.argmin(np.abs(tt - s0))
            ax.annotate(lab, p2[sel][j], fontsize=7, color=cmap(k))
    ax.set_title(f"constellation ({method}, {how}); dot size = time, letters = section starts")
    ax.legend(fontsize=7, loc="best")
    ax3.set_title("PCA 3D")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
