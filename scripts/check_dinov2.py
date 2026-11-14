"""Sanity check for the plain-torch DINOv2: invariances a correct ViT should show.
    srun ... python scripts/check_dinov2.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jukebox.form.embed import embed_frames  # noqa: E402

a = np.load(ROOT / "data/raw/3912/frames.npy")[[0, 60]]
others = [np.load(p, mmap_mode="r")[[10]] for p in sorted((ROOT / "data/raw").glob("*/frames.npy")) if "3912" not in str(p)]
import cv2  # noqa: E402
shift = np.roll(a[:1], 6, axis=2)
bright = np.clip(a[:1].astype(int) + 20, 0, 255).astype(np.uint8)
oth = [cv2.resize(np.asarray(o[0]), (256, 128)) for o in others]
stack = np.concatenate([a, shift, bright, np.stack(oth)])
t0 = time.time()
e = embed_frames(stack, "dinov2")
print("method", e.method, e.info, f"{time.time() - t0:.1f}s")
S = e.X @ e.X.T
names = ["3912 f0", "3912 f60", "f0 shifted 6px", "f0 brighter"] + [p.parent.name for p in sorted((ROOT / "data/raw").glob("*/frames.npy")) if "3912" not in str(p)]
for n, row in zip(names, S[0]):
    print(f"cos(f0, {n:16s}) = {row:.3f}")
