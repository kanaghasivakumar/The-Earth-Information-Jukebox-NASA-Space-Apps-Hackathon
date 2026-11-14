"""Frame-pair dataset for NASA RGB visualizations and Farneback optical flow."""

from __future__ import annotations
import hashlib, json, os, tempfile, re
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageSequence

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CACHE_VERSION = 3


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


class NASAVisualizationDataset(Dataset):
    """Each item is a consecutive sampled frame pair: input [RGB(3), flow(2)]."""

    def __init__(
        self,
        root,
        size=64,
        fps=5.0,
        sequence_fps=30.0,
        flow_scale=1.0,
        cache_dir=None,
        overlay_mask=None,
        align=False,
        sources=None,
    ):
        self.root = Path(root)
        self.size = int(size)
        self.fps = float(fps)
        self.sequence_fps = float(sequence_fps)
        self.flow_scale = float(flow_scale)
        if self.fps <= 0 or self.sequence_fps <= 0 or self.flow_scale <= 0:
            raise ValueError("fps, sequence_fps, and flow_scale must be positive")
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else (self.root.parent if self.root.is_file() else self.root) / ".vae_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.align = bool(align)
        self.mask_path = Path(overlay_mask).resolve() if overlay_mask else None
        self.mask = self._load_mask(self.mask_path) if self.mask_path else None
        paths = [Path(p) for p in sources] if sources is not None else self._discover()
        self.sources = []
        self.samples = []
        for path in paths:
            if not path.exists():
                continue
            frames = self._get_frames(path)
            if not frames:
                continue
            sid = len(self.sources)
            self.sources.append(str(path.resolve()))
            self.samples.extend((sid, *row) for row in frames)
        if not self.samples:
            raise ValueError(
                f"No readable frame pairs found under {self.root}. Add MP4/MOV/WebM/GIF files or image sequences with at least two sampled frames."
            )

    @staticmethod
    def _load_mask(path):
        try:
            with Image.open(path) as im:
                return np.asarray(im.convert("L")) > 0
        except Exception as e:
            raise ValueError(f"Cannot read overlay mask {path}: {e}") from e

    def _discover(self):
        if not self.root.exists():
            return []
        if self.root.is_file():
            return (
                [self.root]
                if self.root.suffix.lower() in VIDEO_EXTS | {".gif"} | IMAGE_EXTS
                else []
            )
        vids = sorted(
            (
                p
                for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS | {".gif"}
            ),
            key=natural_key,
        )
        dirs = sorted(
            {
                p.parent
                for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            },
            key=natural_key,
        )
        groups = [
            d
            for d in dirs
            if d != self.root
            and any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir() if p.is_file())
        ]
        loose = any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in self.root.iterdir()
        )
        return vids + groups + ([self.root] if loose else [])

    def _cache_key(self, source):
        files = (
            sorted([p for p in source.iterdir() if p.is_file()])
            if source.is_dir()
            else [source]
        )
        manifest = [
            (
                str(p.relative_to(source) if source.is_dir() else p.name),
                p.stat().st_size,
                p.stat().st_mtime_ns,
            )
            for p in files
        ]
        settings = {
            "v": CACHE_VERSION,
            "manifest": manifest,
            "size": self.size,
            "fps": self.fps,
            "sequence_fps": self.sequence_fps,
            "flow_scale": self.flow_scale,
            "mask": str(self.mask_path),
            "mask_mtime": self.mask_path.stat().st_mtime_ns if self.mask_path else None,
            "mask_size": self.mask_path.stat().st_size if self.mask_path else None,
            "align": self.align,
        }
        return self.cache_dir / (
            hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
            + ".npz"
        )

    def _raw(self, source):
        if source.is_dir():
            for i, p in enumerate(
                sorted(
                    (
                        q
                        for q in source.iterdir()
                        if q.is_file() and q.suffix.lower() in IMAGE_EXTS
                    ),
                    key=natural_key,
                )
            ):
                try:
                    with Image.open(p) as im:
                        yield i / self.sequence_fps, i, np.asarray(im.convert("RGB"))
                except Exception:
                    continue
            return
        if source.suffix.lower() == ".gif":
            elapsed = 0.0
            idx = 0
            try:
                with Image.open(source) as im:
                    for fr in ImageSequence.Iterator(im):
                        dur = (
                            max(
                                1,
                                int(
                                    fr.info.get(
                                        "duration", im.info.get("duration", 100)
                                    )
                                ),
                            )
                            / 1000.0
                        )
                        yield elapsed, idx, np.asarray(fr.convert("RGB"))
                        elapsed += dur
                        idx += 1
            except Exception:
                return
            return
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return
        native = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(native) or native <= 0:
            native = 30.0
        idx = 0
        next_t = 0.0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                t = idx / native
                yield t, idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                idx += 1
        finally:
            cap.release()

    def _get_frames(self, source):
        cp = self._cache_key(source)
        if cp.exists():
            try:
                with np.load(cp, allow_pickle=False) as archive:
                    timestamps = archive["timestamps"]
                    indices = archive["indices"]
                    rgb_frames = archive["rgb"]
                    flow_frames = archive["flow"]
                return [
                    (
                        float(timestamps[i]),
                        int(indices[i]),
                        rgb_frames[i],
                        flow_frames[i],
                    )
                    for i in range(len(indices))
                ]
            except (OSError, ValueError, KeyError):
                pass
        selected = []
        next_t = 0.0
        for t, idx, arr in self._raw(source):
            if t + 1e-9 < next_t:
                continue
            try:
                rgb = cv2.resize(
                    arr, (self.size, self.size), interpolation=cv2.INTER_AREA
                )
                selected.append((float(t), int(idx), rgb.astype(np.uint8)))
                next_t += 1 / self.fps
            except Exception:
                continue
        if len(selected) < 2:
            return []
        rgbs = np.stack([r[2] for r in selected])
        if self.align and len(rgbs) > 1:
            ref = cv2.cvtColor(rgbs[0], cv2.COLOR_RGB2GRAY)
            aligned = [rgbs[0]]
            for rgb in rgbs[1:]:
                warp = np.eye(2, 3, dtype=np.float32)
                try:
                    cv2.findTransformECC(
                        ref,
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                        warp,
                        cv2.MOTION_TRANSLATION,
                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5),
                    )
                    rgb = cv2.warpAffine(
                        rgb,
                        warp,
                        (self.size, self.size),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                except cv2.error:
                    pass
                aligned.append(rgb)
            rgbs = np.stack(aligned)
        mask = (
            cv2.resize(
                self.mask.astype(np.uint8),
                (self.size, self.size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if self.mask is not None
            else None
        )
        rows = []
        for i in range(1, len(rgbs)):
            a = cv2.cvtColor(rgbs[i - 1], cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2GRAY)
            flow = (
                cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                / self.flow_scale
            )
            flow = np.clip(flow, -1, 1).astype(np.float32)
            rgb = rgbs[i].copy()
            if mask is not None:
                rgb[~mask] = 0
                flow[~mask] = 0
            rows.append((selected[i][0], selected[i][1], rgb, flow.transpose(2, 0, 1)))
        try:
            fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".npz")
            os.close(fd)
            np.savez_compressed(
                tmp,
                timestamps=np.array([x[0] for x in rows]),
                indices=np.array([x[1] for x in rows]),
                rgb=np.stack([x[2] for x in rows]),
                flow=np.stack([x[3] for x in rows]),
            )
            os.replace(tmp, cp)
        except OSError:
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sid, timestamp, frame, rgb, flow = self.samples[index]
        rgb_t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        flow_t = torch.from_numpy(flow.copy()).float()
        return {
            "input": torch.cat((rgb_t, flow_t), 0),
            "image": rgb_t,
            "flow": flow_t,
            "timestamp": torch.tensor(timestamp, dtype=torch.float64),
            "source_id": torch.tensor(sid, dtype=torch.long),
        }


def source_video_split(dataset, val_fraction=0.2, seed=42):
    ids = np.arange(len(dataset.sources))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    if len(ids) < 2:
        raise ValueError("Source holdout requires at least two readable sources.")
    n = max(1, min(len(ids) - 1, round(len(ids) * val_fraction)))
    valset = set(ids[:n])
    tr = [i for i, (sid, *_) in enumerate(dataset.samples) if sid not in valset]
    va = [i for i, (sid, *_) in enumerate(dataset.samples) if sid in valset]
    return tr, va


def exploratory_time_block_split(dataset, source_id=0, val_fraction=0.2, gap=2):
    """Exploratory one-source split: contiguous regions separated by `gap` excluded samples."""
    ix = [i for i, (sid, *_) in enumerate(dataset.samples) if sid == source_id]
    cut = max(1, min(len(ix) - gap - 1, round((len(ix) - gap) * (1 - val_fraction))))
    tr = ix[:cut]
    va = ix[cut + gap :]
    if not tr or not va:
        raise ValueError(
            "Time split/gap leaves an empty side; reduce gap or adjust validation fraction."
        )
    return tr, va
