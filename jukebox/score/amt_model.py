"""Minimal GPT-2 forward pass for the Anticipatory Music Transformer checkpoints.

Why not `transformers`? The shared env has transformers 5.x with torch 2.5.1, and
transformers 5 refuses to enable PyTorch below 2.6 ("Models won't be available").
The AMT checkpoints (stanford-crfm/music-{small,medium}-800k, Apache-2.0) are plain
GPT-2, so ~80 lines of torch load them directly, with a KV cache for fast sampling.
Checked against transformers 4.46 GPT2LMHeadModel logits (see tests/test_infill.py).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", "/projects/e32706/omb8654/hf_cache")


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class Block(nn.Module):
    def __init__(self, d, h, idx, eps, inv_layer_scale):
        super().__init__()
        self.h, self.dh = h, d // h
        self.scale = 1.0 / math.sqrt(self.dh) / ((idx + 1) if inv_layer_scale else 1)
        self.ln_1 = nn.LayerNorm(d, eps=eps)
        self.ln_2 = nn.LayerNorm(d, eps=eps)
        self.c_attn = nn.Linear(d, 3 * d)
        self.attn_proj = nn.Linear(d, d)
        self.c_fc = nn.Linear(d, 4 * d)
        self.mlp_proj = nn.Linear(4 * d, d)

    def forward(self, x, past):
        B, T, D = x.shape
        q, k, v = self.c_attn(self.ln_1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        if past is not None:
            k = torch.cat([past[0], k], 2)
            v = torch.cat([past[1], v], 2)
        L = k.shape[2]
        if T == 1:
            mask = None
        else:  # causal mask offset by the cached length
            mask = torch.ones(T, L, dtype=torch.bool, device=x.device).tril(L - T)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        x = x + self.attn_proj(a.transpose(1, 2).reshape(B, T, D))
        x = x + self.mlp_proj(gelu_new(self.c_fc(self.ln_2(x))))
        return x, (k, v)


class GPT2(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, eps = cfg["n_embd"], cfg["layer_norm_epsilon"]
        self.n_positions = cfg["n_positions"]
        self.wte = nn.Embedding(cfg["vocab_size"], d)
        self.wpe = nn.Embedding(cfg["n_positions"], d)
        self.h = nn.ModuleList(Block(d, cfg["n_head"], i, eps, cfg.get("scale_attn_by_inverse_layer_idx", False))
                               for i in range(cfg["n_layer"]))
        self.ln_f = nn.LayerNorm(d, eps=eps)
        self.name = cfg.get("_name", "gpt2")

    @property
    def device(self):
        return self.wte.weight.device

    def forward(self, ids: torch.Tensor, past=None):
        """ids (1, T) -> logits (1, T, V), new cache (list of (k, v))."""
        p0 = 0 if past is None else past[0][0].shape[2]
        pos = torch.arange(p0, p0 + ids.shape[1], device=ids.device)
        x = self.wte(ids) + self.wpe(pos)[None]
        cache = []
        for i, blk in enumerate(self.h):
            x, kv = blk(x, None if past is None else past[i])
            cache.append(kv)
        x = self.ln_f(x)
        return x @ self.wte.weight.T, cache


def _convert(sd: dict, n_layer: int) -> dict:
    """HF GPT-2 names (Conv1D weights are (in, out)) -> this module (Linear (out, in))."""
    out = {}
    for k, v in sd.items():
        k = k.removeprefix("transformer.")
        if v is None or k == "token_out_embeddings":  # export artifact (None); output head is tied to wte
            continue
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias") or k.startswith("lm_head"):
            continue
        k = (k.replace("attn.c_attn", "c_attn").replace("attn.c_proj", "attn_proj")
              .replace("mlp.c_fc", "c_fc").replace("mlp.c_proj", "mlp_proj"))
        if k.endswith(".weight") and any(s in k for s in ("c_attn", "attn_proj", "c_fc", "mlp_proj")):
            v = v.T.contiguous()
        out[k] = v
    return out


_CACHE: dict[str, GPT2] = {}


def load(name: str = "stanford-crfm/music-medium-800k", device: str = "cpu") -> GPT2:
    """Load (and memoize) an AMT checkpoint from the HF hub or a local directory."""
    key = f"{name}@{device}"
    if key in _CACHE:
        return _CACHE[key]
    if Path(name).is_dir():
        cfg_p, w_p = Path(name) / "config.json", Path(name) / "pytorch_model.bin"
    else:
        from huggingface_hub import hf_hub_download
        cfg_p, w_p = hf_hub_download(name, "config.json"), hf_hub_download(name, "pytorch_model.bin")
    cfg = json.loads(Path(cfg_p).read_text())
    cfg["_name"] = name
    m = GPT2(cfg)
    sd = torch.load(w_p, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(_convert(sd, cfg["n_layer"]), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    m.eval().to(device)
    _CACHE[key] = m
    return m
