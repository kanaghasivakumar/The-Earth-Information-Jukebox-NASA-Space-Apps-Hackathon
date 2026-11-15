#!/usr/bin/env python3
"""Export a 30-second stretch of the real pipeline for the web walkthrough.

The stretch starts at 0:57 and crosses from one pattern group into the next, so the
dot makes a long, visible move across the map.

Writes frontend/public/jukebox/pipeline.json plus frame thumbnails and two sprite
sheets. Everything is read from saved results: video frames, VAE latent means,
pattern-group labels, the decoder's reconstructions and the generated MIDI notes.
Nothing is simulated.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pretty_midi
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'eic-sonification' / 'frontend' / 'public' / 'jukebox'
SEGS = [(57.0, 30.0)]  # (source start seconds, length): one continuous stretch crossing from the purple to the blue group
FRAME_TIMES = [59.0, 65.0, 71.0, 77.0, 83.0]  # source seconds of the five frames used in the close-up
COLS, ROWS, TILE = 15, 10, 64

results = ROOT / 'results'
ts = np.load(results / 'timestamps.npy')
z = np.load(results / 'latent_means.npy')
dims = np.load(results / 'selected_dims.npy')[:2]
labels = np.load(results / 'ambient' / 'cluster_labels.npy')
video = next((ROOT / 'data' / 'raw').glob('*.webm'))

lo, hi = np.percentile(z, 2, axis=0), np.percentile(z, 98, axis=0)
zn = np.clip((z - lo) / np.maximum(hi - lo, 1e-9), 0, 1)

from dataset import NASAVisualizationDataset  # noqa: E402
from models.vae import ConvVAE  # noqa: E402

ckpt = torch.load(ROOT / 'checkpoints' / 'best_vae.pt', map_location='cpu', weights_only=False)
cfg, prep = ckpt['model_config'], dict(ckpt['preprocessing'])
ds = NASAVisualizationDataset(ROOT / 'data' / 'raw', **prep, cache_dir=ROOT / 'data' / 'cache')
model = ConvVAE(latent_dim=int(cfg['latent_dim']), image_size=int(cfg.get('image_size', cfg.get('size', 64))))
model.load_state_dict(ckpt.get('model', ckpt.get('model_state_dict', ckpt.get('model_state'))))
model.eval()
convs = [model.encoder[1], model.encoder[3], model.encoder[5]]

(OUT / 'pipeline').mkdir(parents=True, exist_ok=True)

# ---- every sample in the tour, in order ----
sample_idx = []
seg_of = []
for s, (start, length) in enumerate(SEGS):
    ii = np.where((ts >= start) & (ts < start + length))[0]
    sample_idx.extend(ii.tolist())
    seg_of.extend([s] * len(ii))
sample_idx = np.array(sample_idx)
assert len(sample_idx) <= COLS * ROWS, len(sample_idx)
times = [round(float(ts[i]), 2) for i in sample_idx]
dt = round(float(np.median(np.diff(ts))), 3)


def pooled(a, k):
    c, h, w = a.shape
    return a.reshape(c, h // k, k, w // k, k).mean(axis=(2, 4))


# ---- frames for the close-up ----
frames = []
for k, ft in enumerate(FRAME_TIMES):
    i = int(np.argmin(np.abs(ts - ft)))
    gi = int(np.where(sample_idx == i)[0][0])
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{ts[i]:.2f}', '-i', str(video), '-frames:v', '1',
                    '-vf', 'scale=200:200', '-q:v', '5', str(OUT / 'pipeline' / f'f{k}.jpg')], check=True)
    cur, prev = ds[i], ds[i - 1]
    Image.fromarray((cur['image'].permute(1, 2, 0).numpy() * 255).astype('uint8')).save(OUT / 'pipeline' / f'p{k}.png')
    Image.fromarray((prev['image'].permute(1, 2, 0).numpy() * 255).astype('uint8')).save(OUT / 'pipeline' / f'q{k}.png')
    f16 = pooled(cur['flow'].numpy(), 4)
    frames.append({'t': round(float(ts[i]), 2), 'gi': gi, 'img': f'/jukebox/pipeline/f{k}.jpg',
                   'latent': zn[i].round(3).tolist(),
                   'xy': zn[i, dims].round(3).tolist(), 'region': int(labels[i]), 'flow16': f16.round(3).tolist()})

latent_map = [[*zn[i, dims].round(3).tolist(), int(labels[i])] for i in np.arange(0, len(ts), 3)]

# ---- notes of every style across the tour, with times on the tour clock ----
notes = {}
for style in ('ambient', 'cinematic', 'classical', 'jazz', 'electronic'):
    pm = pretty_midi.PrettyMIDI(str(results / style / 'earth_jukebox.mid'))
    rows = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            for s, (start, length) in enumerate(SEGS):
                if start <= n.start < start + length:
                    g0 = sum(L for _, L in SEGS[:s])
                    rows.append([round(g0 + n.start - start, 2), round(min(n.end, start + length) - n.start, 2), n.pitch, n.velocity])
    rows.sort()
    notes[style] = rows

# ---- the VAE in action: original vs. what the decoder rebuilds from only the eight numbers ----
orig_sheet = Image.new('RGB', (COLS * TILE, ROWS * TILE))
recon_sheet = Image.new('RGB', (COLS * TILE, ROWS * TILE))
with torch.no_grad():
    for n, i in enumerate(sample_idx):
        item = ds[int(i)]
        rgb, _ = model.decode(torch.from_numpy(z[i]).float()[None])
        pos = ((n % COLS) * TILE, (n // COLS) * TILE)
        orig_sheet.paste(Image.fromarray((item['image'].permute(1, 2, 0).numpy() * 255).astype('uint8')), pos)
        recon_sheet.paste(Image.fromarray((rgb[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype('uint8')), pos)
orig_sheet.save(OUT / 'pipeline' / 'sheet_orig.png', optimize=True)
recon_sheet.save(OUT / 'pipeline' / 'sheet_recon.png', optimize=True)

mv = np.linalg.norm(np.diff(z, axis=0, prepend=z[:1]), axis=1)[sample_idx]
mv = np.clip((mv - mv.min()) / max(mv.max() - mv.min(), 1e-9), 0, 1)
path = [[*zn[i, dims].round(3).tolist(), int(labels[i]), s] for i, s in zip(sample_idx, seg_of)]
lat = [zn[i].round(2).tolist() for i in sample_idx]

(OUT / 'pipeline.json').write_text(json.dumps({
    'segs': [[s, L] for s, L in SEGS], 'span': float(sum(L for _, L in SEGS)), 'cols': COLS, 'rows': ROWS,
    'frames': frames, 'map': latent_map, 'notes': notes, 'movement': mv.round(3).tolist(),
    'path': path, 'lat': lat, 'times': times, 'dt': dt, 'dims': dims.tolist(),
}, separators=(',', ':')))
print({s: len(v) for s, v in notes.items()}, 'samples', len(sample_idx), 'groups', sorted({p[2] for p in path}))
