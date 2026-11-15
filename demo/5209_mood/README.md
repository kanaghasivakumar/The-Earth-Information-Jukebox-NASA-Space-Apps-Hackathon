# 5209 mood pilot: Zonal temperature anomalies 1880–2023

A learned **mood texture** under the unchanged integrated arrangement: video frames → SigLIP style weights over a fixed prompt vocabulary (blended with the track's data curves) → Magenta RealTime 2 (`mrt2_small`, CPU) generating continuously over the timeline. The mood never sets a pitch. The data melody and all other stems are exactly the `integrate` stems.

| File | What it is |
|---|---|
| `mix_mood.mp3` | integrated arrangement + mood stem, −16 LUFS (compare with `demo/5209/mix.mp3`) |
| `mood.mp3` | the mood stem alone, at its mix level (after ducking) |
| `weights.png` | frames, SigLIP-only weights, blended weights, data curves |
| `mood.png` | weights sent to the model, mood and mix spectrograms (cyan: lead pitch) |
| `summary.json` | the numbers below plus prompts and settings |

## Prompts (anchor: music prompt ← visual text, data link)

- **icy**: “icy glassy pads” ← “cold blue colors” — data: −anomaly
- **warm**: “warm tropical marimba” ← “warm red and orange colors” — data: +anomaly
- **heat**: “tense rising cinematic drums” ← “hot dark red heat” — data: +anomaly
- **calm**: “calm cool ambient” ← “pale white and light blue colors”
- **solemn**: “solemn orchestral strings” ← “a chart with lines on a dark background”
- **wind**: “airy wind ambient” ← “an empty grey chart”

## Fidelity (note-level pyin round trip, same method as `arrange.py`)

| Audio | R² | pitch within ½ st |
|---|---|---|
| full mix, no mood (integrate) | 0.966 | 1.00 |
| full mix + mood | 0.966 | 1.00 |
| shuffled control mix, no mood | -0.551 | 0.94 |
| shuffled control mix + mood | -0.634 | 0.94 |

R² change from the mood: -0.000. Mood sits -14 dB under the melody's active RMS before the sidechain (ducked 0.9 dB on average).

## Does the mood follow the picture?

Correlation of each data-linked anchor's weight with its data curve. *Vision only* is SigLIP alone; *blended* is what was sent to the model (it includes the data curve, so it correlates by construction). *Detrended* removes a linear trend from both series first.

| Anchor ↔ curve | vision only | vision, detrended | blended |
|---|---|---|---|
| icy ↔ 1 − anomaly | 0.77 | -0.42 | 0.83 |
| warm ↔ anomaly | 0.77 | 0.14 | 0.87 |
| heat ↔ anomaly | 0.91 | 0.58 | 0.97 |

## Does the generated audio follow the weights?

MusicCoCa audio embeddings of the raw mood (11 clips of 10 s, every 2 s) against the prompts' text embeddings: mean r over anchors between “clip ↔ prompt similarity” and the weight the model was given = 0.17 (same audio against time-reversed weights: -0.08). The closest prompt is the most-weighted prompt in 82% of clips (chance 17%).

| Anchor | r | share of clips closest |
|---|---|---|
| icy | 0.95 | 73% |
| warm | 0.94 | 27% |
| heat | -0.68 | 0% |
| calm | -0.74 | 0% |
| solemn | 0.42 | 0% |
| wind | 0.10 | 0% |

## Key

Notes conditioning: `key` (out-of-scale pitches and everything from one semitone under the lead's lowest note upward set OFF; harmony chord tones below that FREE). Chroma share on the track's scale: 0.77 (chance 0.58).
Chroma cosine with the harmony stem per 2 s: 0.80 in place vs 0.50 averaged over the 11 transpositions; best shift 0 st; shift 0 best in 80% of segments → no systematic clash.

## Runtime

mrt2_small on CPU: 182 s for 32 s of audio (346 s per audio minute, 0.17× real time), plus 18 s load/compile.

Reproduce:

```bash
srun -A p32576 -p short -t 00:30:00 --mem=24G -c 8 scripts/mood/weights.sh 5209
srun -A p32576 -p short -t 02:00:00 --mem=48G -c 8 scripts/mood/render.sh mrt2_small 5209   # CPU
sbatch scripts/mood/gpu_base.sbatch                                                     # GPU, base
```

Nothing here has been listened to; every judgement is from the metrics above.
