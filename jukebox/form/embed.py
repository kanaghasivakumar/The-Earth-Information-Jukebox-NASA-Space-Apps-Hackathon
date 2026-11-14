"""Frame embeddings for musical FORM (sections, motifs, navigation). Never used for pitch.

    from jukebox.form.embed import embed_frames
    emb = embed_frames(frames)          # frames: (T, H, W, 3) uint8 RGB
    emb.X, emb.method                   # (T, D) float32, "dinov2-small" or "handcrafted"

Default: DINOv2-small (facebook/dinov2-small, 22M params, CPU friendly), CLS token
concatenated with the mean patch token, L2-normalised. Weights come from the HF hub cache;
the forward pass is plain torch because the env's transformers requires torch >= 2.6.
Fallback: colour histograms + a coarse colour layout + Farneback optical-flow statistics.
The fallback needs only numpy + OpenCV, so it always works offline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

DINO_ID = "facebook/dinov2-small"
DEFAULT_HF_HOME = "/projects/e32706/omb8654/hf_cache"


@dataclass
class Embedding:
    X: np.ndarray                     # (T, D) float32
    method: str
    info: dict = field(default_factory=dict)


def embed_frames(frames: np.ndarray, method: str = "auto", *, mask: np.ndarray | None = None,
                 batch: int = 16, max_side: int = 448, device: str = "cpu") -> Embedding:
    """Embed every frame of a (T, H, W, 3) uint8 RGB stack.

    method: "auto" (DINOv2, falling back to handcrafted on any failure), "dinov2", "handcrafted".
    mask:   optional (H, W) bool; False pixels (burned-in labels, colorbars) are set to black first.
    """
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (T, H, W, 3), got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if mask is not None:
        frames = frames * np.asarray(mask, bool)[None, :, :, None].astype(np.uint8)
    if method in ("auto", "dinov2"):
        try:
            return _dinov2(frames, batch=batch, max_side=max_side, device=device)
        except Exception as e:  # no network, no weights, no torch ...
            if method == "dinov2":
                raise
            emb = handcrafted(frames)
            emb.info["fallback_reason"] = f"{type(e).__name__}: {e}"[:300]
            return emb
    if method == "handcrafted":
        return handcrafted(frames)
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- DINOv2

def _target_hw(h: int, w: int, max_side: int, patch: int = 14) -> tuple[int, int]:
    """Keep aspect ratio, short side >= 224, long side <= max_side, multiples of the patch."""
    s = max(224 / min(h, w), 1e-9)
    s = min(s, max_side / max(h, w)) if max(h, w) * s > max_side else s
    return max(patch, int(round(h * s / patch)) * patch), max(patch, int(round(w * s / patch)) * patch)


def _dinov2_weights() -> tuple[dict, dict]:
    """Load facebook/dinov2-small config + safetensors via huggingface_hub (cache first)."""
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    paths = {}
    for name in ("config.json", "model.safetensors"):
        try:
            paths[name] = hf_hub_download(DINO_ID, name, local_files_only=True)
        except Exception:
            paths[name] = hf_hub_download(DINO_ID, name)
    return json.loads(open(paths["config.json"]).read()), load_file(paths["model.safetensors"])


class _DinoV2:
    """Minimal DINOv2 ViT forward pass in plain torch.

    The env's transformers refuses torch < 2.6 ("Disabling PyTorch"), so we run the published
    weights ourselves: patch embed -> [CLS] + bicubic-interpolated position embeddings ->
    pre-norm blocks with LayerScale -> final LayerNorm. Same maths as HF Dinov2Model (eval mode).
    """

    def __init__(self, cfg: dict, w: dict, device: str = "cpu"):
        import torch
        self.cfg, self.dev = cfg, device
        self.w = {k: v.to(device).float() for k, v in w.items()}
        self.L, self.H, self.p = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["patch_size"]
        self.eps = cfg.get("layer_norm_eps", 1e-6)
        self.torch = torch

    def __call__(self, x):
        import torch.nn.functional as F
        w, p = self.w, self.p
        B, _, h, wd = x.shape
        gh, gw = h // p, wd // p
        x = F.conv2d(x, w["embeddings.patch_embeddings.projection.weight"],
                     w["embeddings.patch_embeddings.projection.bias"], stride=p)
        x = x.flatten(2).transpose(1, 2)                                    # (B, N, C)
        pos = w["embeddings.position_embeddings"]
        n0 = int(round((pos.shape[1] - 1) ** 0.5))
        pp = pos[:, 1:].reshape(1, n0, n0, -1).permute(0, 3, 1, 2)
        if (gh, gw) != (n0, n0):
            pp = F.interpolate(pp, size=(gh, gw), mode="bicubic", align_corners=False)
        pp = pp.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
        cls = w["embeddings.cls_token"].expand(B, -1, -1) + pos[:, :1]
        x = self.torch.cat([cls, x + pp], 1)
        C = x.shape[-1]
        for i in range(self.L):
            k = f"encoder.layer.{i}."
            y = F.layer_norm(x, (C,), w[k + "norm1.weight"], w[k + "norm1.bias"], self.eps)
            q, kk, v = (F.linear(y, w[k + f"attention.attention.{n}.weight"], w[k + f"attention.attention.{n}.bias"])
                        .reshape(B, -1, self.H, C // self.H).transpose(1, 2) for n in ("query", "key", "value"))
            a = F.scaled_dot_product_attention(q, kk, v).transpose(1, 2).reshape(B, -1, C)
            a = F.linear(a, w[k + "attention.output.dense.weight"], w[k + "attention.output.dense.bias"])
            x = x + w[k + "layer_scale1.lambda1"] * a
            y = F.layer_norm(x, (C,), w[k + "norm2.weight"], w[k + "norm2.bias"], self.eps)
            y = F.linear(F.gelu(F.linear(y, w[k + "mlp.fc1.weight"], w[k + "mlp.fc1.bias"])),
                         w[k + "mlp.fc2.weight"], w[k + "mlp.fc2.bias"])
            x = x + w[k + "layer_scale2.lambda1"] * y
        return F.layer_norm(x, (C,), w["layernorm.weight"], w["layernorm.bias"], self.eps)


def _dinov2(frames: np.ndarray, *, batch: int, max_side: int, device: str) -> Embedding:
    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    import cv2
    import torch

    cfg, weights = _dinov2_weights()
    model = _DinoV2(cfg, weights, device)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    T, H, W, _ = frames.shape
    th, tw = _target_hw(H, W, max_side)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    out = []
    with torch.inference_mode():
        for i in range(0, T, batch):
            chunk = [cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA if tw < W else cv2.INTER_CUBIC)
                     for f in frames[i:i + batch]]
            x = (np.stack(chunk).astype(np.float32) / 255 - mean) / std
            x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
            h = model(x)                                         # (B, 1 + N, 384)
            feat = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
            out.append(feat.float().cpu().numpy())
    X = np.concatenate(out).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return Embedding(X, "dinov2-small", {"model": DINO_ID, "impl": "plain-torch (jukebox.form.embed._DinoV2)",
                                         "input_hw": [th, tw], "dim": int(X.shape[1])})


# --------------------------------------------------------------------------- fallback

def _flow_stats(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    import cv2
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hist = np.bincount((ang.ravel() / (2 * np.pi) * 8).astype(int) % 8, weights=mag.ravel(), minlength=8)
    hist = hist / (hist.sum() + 1e-9)
    # global translation (camera pan proxy) vs. residual motion (content)
    dx, dy = float(np.median(flow[..., 0])), float(np.median(flow[..., 1]))
    return np.r_[mag.mean(), mag.std(), np.percentile(mag, 90), dx, dy, hist].astype(np.float32)


def handcrafted(frames: np.ndarray, *, small: int = 64, standardize: bool = True) -> Embedding:
    """HSV histograms (global) + 4x8 Lab colour layout + optical-flow statistics.

    standardize=True z-scores each feature within this clip (good for segmenting one video);
    use False when several videos must share one space (constellation).
    """
    import cv2
    T, H, W, _ = frames.shape
    sw = small
    sh = max(8, int(round(small * H / W)))
    hists, layouts, flows = [], [], []
    prev = None
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        hh = np.histogram(hsv[..., 0], 18, (0, 180))[0]
        hs = np.histogram(hsv[..., 1], 8, (0, 256))[0]
        hv = np.histogram(hsv[..., 2], 8, (0, 256))[0]
        hists.append(np.r_[hh, hs, hv].astype(np.float32) / (H * W))
        lab = cv2.cvtColor(cv2.resize(f, (8, 4), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)
        layouts.append(lab.astype(np.float32).ravel() / 255)
        g = cv2.cvtColor(cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        flows.append(_flow_stats(prev, g) if prev is not None else None)
        prev = g
    if T > 1:
        flows[0] = flows[1]
    else:
        flows[0] = np.zeros(13, np.float32)
    blocks = [np.stack(hists), np.stack(layouts), np.stack(flows)]
    parts = []
    for b in blocks:  # z-score per feature, then weight each block equally
        z = (b - b.mean(0)) / (b.std(0) + 1e-6) if standardize else b
        parts.append(z / np.sqrt(b.shape[1]))
    X = np.concatenate(parts, 1).astype(np.float32)
    return Embedding(X, "handcrafted", {"blocks": {"hsv_hist": 34, "lab_layout": 96, "flow": 13},
                                        "dim": int(X.shape[1])})
