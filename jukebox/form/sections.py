"""Musical form from frame embeddings: sections, recurring regions (motif IDs), returns.

    from jukebox.form.sections import analyse
    form = analyse(emb.X, t)               # t: frame times in seconds (video timeline)
    form.tuples()                          # [(start_s, end_s, "A"), (.., .., "B"), ...]
    form.to_json()                         # sections, labels, recurrences, returns, frame regions

Pipeline
1. light temporal smoothing of the embedding trajectory (removes flow-line flicker),
2. centred cosine self-similarity matrix (SSM),
3. Foote novelty: Gaussian-tapered checkerboard kernel slid along the SSM diagonal,
   boundaries = novelty peaks (prominence + minimum section length),
4. regions: PCA -> k-means on frames (k by silhouette), then dwell + hysteresis smoothing
   (same rule as eic-vae/music/memory.py), so motif IDs do not flicker,
5. section label = majority region inside the section; a section whose label already occurred
   earlier, with a different label in between, is a RETURN (the score should recall that motif).

Form is structure only. Nothing here may set a data note's pitch.
"""
from __future__ import annotations

import string
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Section:
    index: int
    start: float
    end: float
    start_frame: int
    end_frame: int            # exclusive
    label: str                # "A", "B", ... in order of first appearance
    region: int               # region (motif) id the label maps to
    boundary_novelty: float   # novelty peak height at this section's start (0 for the first)
    is_return: bool = False
    returns_to: int | None = None   # index of the first section with this label


@dataclass
class Form:
    sections: list[Section]
    novelty: np.ndarray
    ssm: np.ndarray
    frame_regions: np.ndarray        # hysteresis-smoothed region id per frame
    frame_regions_raw: np.ndarray
    times: np.ndarray
    k: int
    silhouette: float | None
    repetition: dict
    params: dict = field(default_factory=dict)

    def tuples(self) -> list[tuple[float, float, str]]:
        return [(s.start, s.end, s.label) for s in self.sections]

    @property
    def returns(self) -> list[Section]:
        return [s for s in self.sections if s.is_return]

    def recurrences(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for s in self.sections:
            out.setdefault(s.label, []).append(s.index)
        return out

    def to_json(self) -> dict:
        runs = _runs(self.frame_regions, self.times)
        seen, region_returns = set(), []
        for r in runs:
            if r["region"] in seen:
                region_returns.append(r)
            seen.add(r["region"])
        return {
            "sections": [asdict(s) for s in self.sections],
            "tuples": [list(x) for x in self.tuples()],
            "labels": sorted({s.label for s in self.sections}),
            "label_to_region": {s.label: s.region for s in self.sections},
            "recurrences": self.recurrences(),
            "returns": [{"section": s.index, "label": s.label, "time": s.start, "first_section": s.returns_to}
                        for s in self.returns],
            "regions": {"k": self.k, "silhouette": self.silhouette, "runs": runs, "returns": region_returns,
                        "per_frame": self.frame_regions.astype(int).tolist()},
            "novelty": np.round(self.novelty, 4).tolist(),
            "times": np.round(self.times, 4).tolist(),
            "repetition": self.repetition,
            "params": self.params,
        }


# --------------------------------------------------------------------------- building blocks

def smooth(X: np.ndarray, frames: float) -> np.ndarray:
    """Gaussian smoothing along time (sigma in frames); edges use nearest values."""
    if frames <= 0 or len(X) < 3:
        return X.copy()
    r = int(np.ceil(3 * frames))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / frames) ** 2)
    k /= k.sum()
    P = np.pad(X, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(P[:, j], k, mode="valid") for j in range(X.shape[1])], 1)


def self_similarity(X: np.ndarray) -> np.ndarray:
    """Centred cosine similarity, in [-1, 1]. Centring removes the 'everything looks alike' offset."""
    Z = X - X.mean(0, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    return Z @ Z.T


def checkerboard(L: int) -> np.ndarray:
    """(2L, 2L) Foote kernel: +1 within-past/within-future blocks, -1 across, Gaussian taper."""
    ax = np.arange(-L, L) + 0.5
    g = np.exp(-0.5 * (ax / (0.5 * L)) ** 2)
    sign = np.sign(ax)
    return np.outer(g, g) * np.outer(sign, sign)


def foote_novelty(S: np.ndarray, L: int) -> np.ndarray:
    """Novelty in [0, 1]. The SSM is extended by edge-replicating the trajectory so the ends are not
    artificially novel."""
    T = len(S)
    idx = np.clip(np.arange(-L, T + L), 0, T - 1)
    P = S[np.ix_(idx, idx)]
    K = checkerboard(L)
    nov = np.array([np.sum(K * P[i:i + 2 * L, i:i + 2 * L]) for i in range(T)])
    nov = np.maximum(nov, 0)
    return nov / (nov.max() + 1e-12)


def pick_boundaries(nov: np.ndarray, min_len: int, prominence: float, max_sections: int | None) -> list[int]:
    from scipy.signal import find_peaks
    T = len(nov)
    peaks, props = find_peaks(nov, distance=max(1, min_len), prominence=prominence)
    keep = [(p, pr) for p, pr in zip(peaks, props["prominences"]) if min_len <= p <= T - min_len]
    if max_sections and len(keep) > max_sections - 1:
        keep = sorted(keep, key=lambda x: -x[1])[:max_sections - 1]
    return sorted(int(p) for p, _ in keep)


def repetition_lag(S: np.ndarray, t: np.ndarray, min_lag_frac: float = 0.1, min_strength: float = 0.1) -> dict:
    """Mean similarity along each SSM diagonal; the strongest off-zero peak is a repetition period
    (e.g. a seasonal cycle). Returned so a human can check it against the dates in the video."""
    T = len(S)
    if T < 8:
        return {"lag_frames": None, "lag_s": None, "strength": None, "lag_profile": []}
    prof = np.array([np.mean(np.diagonal(S, k)) for k in range(T // 2 + 1)])
    lo = max(2, int(min_lag_frac * T))
    from scipy.signal import find_peaks
    pk, pr = find_peaks(prof[lo:], prominence=0.02)
    if not len(pk):
        return {"lag_frames": None, "lag_s": None, "strength": None, "lag_profile": np.round(prof, 4).tolist()}
    best = int(pk[np.argmax(prof[lo:][pk])]) + lo
    if prof[best] < min_strength:   # a bump in a decaying profile is not a cycle
        return {"lag_frames": None, "lag_s": None, "strength": float(prof[best]), "lag_profile": np.round(prof, 4).tolist()}
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    return {"lag_frames": best, "lag_s": best * dt, "strength": float(prof[best]),
            "lag_profile": np.round(prof, 4).tolist()}


def dwell_filter(labels: np.ndarray, t: np.ndarray, min_dwell: float, hysteresis: float) -> np.ndarray:
    """Switch region only after the old one lasted `min_dwell` s and the new one persisted
    `hysteresis` s (mirrors eic-vae/music/memory.py::_dwell_filter)."""
    if len(labels) < 2:
        return labels.copy()
    out = np.empty_like(labels)
    active, since = int(labels[0]), float(t[0])
    cand, cand_since = None, since
    for i, (raw, now) in enumerate(zip(labels.astype(int), t.astype(float))):
        if raw == active:
            cand = None
        else:
            if cand != raw:
                cand, cand_since = raw, now
            if now - since >= min_dwell and now - cand_since >= hysteresis:
                active, since, cand = raw, now, None
        out[i] = active
    # the filter reports a switch at the moment it is confirmed; shift it back to when the
    # candidate started, so regions line up with the video rather than lagging by `hysteresis`.
    return _backdate(out, labels.astype(int))


def _backdate(filtered: np.ndarray, raw: np.ndarray) -> np.ndarray:
    out = filtered.copy()
    for i in range(1, len(out)):
        if out[i] != out[i - 1]:
            j = i
            while j > 0 and raw[j - 1] == out[i] and out[j - 1] != out[i]:
                j -= 1
            out[j:i] = out[i]
    return out


def regions(X: np.ndarray, t: np.ndarray, *, k: int | None = None, k_max: int = 6, min_dwell: float = 2.0,
            hysteresis: float = 1.0, seed: int = 0, min_silhouette: float = 0.12):
    """PCA -> k-means region ids per frame (k by silhouette unless given), then dwell/hysteresis."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    T = len(X)
    Z = X - X.mean(0)
    nc = int(min(16, T - 1, Z.shape[1]))
    Z = PCA(nc, random_state=seed).fit_transform(Z) if nc >= 2 else Z
    best = (1, None, np.zeros(T, int))
    if k is None:
        for kk in range(2, min(k_max, T - 1) + 1):
            lab = KMeans(kk, n_init=10, random_state=seed).fit_predict(Z)
            if len(set(lab)) < 2:
                continue
            sil = float(silhouette_score(Z, lab))
            if best[1] is None or sil > best[1]:
                best = (kk, sil, lab)
        if best[1] is None or best[1] < min_silhouette:   # no real cluster structure: one region
            best = (1, best[1], np.zeros(T, int))
    elif k > 1:
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
        best = (k, float(silhouette_score(Z, lab)) if len(set(lab)) > 1 else None, lab)
    kk, sil, raw = best
    return kk, sil, raw.astype(int), dwell_filter(raw.astype(int), t, min_dwell, hysteresis)


def _runs(lab: np.ndarray, t: np.ndarray) -> list[dict]:
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append({"region": int(lab[s]), "start": float(t[s]),
                        "end": float(t[i] if i < len(lab) else t[-1] + dt), "frames": i - s})
            s = i
    return out


# --------------------------------------------------------------------------- main entry

def analyse(X: np.ndarray, t, *, kernel_s: float | None = None, smooth_s: float = 0.5,
            min_section_s: float | None = None, prominence: float = 0.1, max_sections: int | None = 12,
            k: int | None = None, min_dwell_s: float | None = None, hysteresis_s: float | None = None,
            seed: int = 0) -> Form:
    """Segment one chronological embedding trajectory X (T, D) with frame times t (s)."""
    X = np.asarray(X, np.float64)
    t = np.asarray(t, float).reshape(-1)
    if X.ndim != 2 or len(X) != len(t):
        raise ValueError("X must be (T, D) and match t")
    T = len(t)
    dt = float(np.median(np.diff(t))) if T > 1 else 1.0
    dur = float(t[-1] - t[0] + dt)
    kernel_s = kernel_s or float(np.clip(dur / 10, 2 * dt, 16.0))
    min_section_s = min_section_s or max(kernel_s, 2 * dt)
    min_dwell_s = min_section_s / 2 if min_dwell_s is None else min_dwell_s
    hysteresis_s = min_dwell_s / 2 if hysteresis_s is None else hysteresis_s
    L = max(1, int(round(kernel_s / dt / 2)))
    Xs = smooth(X, smooth_s / dt)
    S = self_similarity(Xs)
    nov = foote_novelty(S, L) if T > 2 else np.zeros(T)
    bounds = pick_boundaries(nov, int(round(min_section_s / dt)), prominence, max_sections) if T > 4 else []
    kk, sil, raw, reg = regions(Xs, t, k=k, min_dwell=min_dwell_s, hysteresis=hysteresis_s, seed=seed)

    edges = [0] + bounds + [T]
    secs, first_of_label, region_to_label = [], {}, {}
    letters = string.ascii_uppercase
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        r = int(np.bincount(reg[a:b]).argmax())
        if r not in region_to_label:
            region_to_label[r] = letters[len(region_to_label) % 26]
        lab = region_to_label[r]
        end = float(t[b]) if b < T else float(t[-1] + dt)
        s = Section(i, float(t[a]), end, int(a), int(b), lab, r, float(nov[a]) if i else 0.0)
        if lab in first_of_label and secs[-1].label != lab:
            s.is_return, s.returns_to = True, first_of_label[lab]
        elif lab in first_of_label:   # same label continues across a boundary: a variation, not a return
            s.returns_to = first_of_label[lab]
        first_of_label.setdefault(lab, i)
        secs.append(s)

    params = {"kernel_s": kernel_s, "kernel_half_frames": L, "smooth_s": smooth_s, "min_section_s": min_section_s,
              "prominence": prominence, "max_sections": max_sections, "min_dwell_s": min_dwell_s,
              "hysteresis_s": hysteresis_s, "seed": seed, "dt": dt}
    return Form(secs, nov, S, reg, raw, t, kk, sil, repetition_lag(S, t), params)
