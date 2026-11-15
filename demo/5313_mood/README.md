# 5313 mood pilot: Night lights 2012–2023

A learned **mood texture** under the unchanged integrated arrangement: video frames → SigLIP style weights over a fixed prompt vocabulary (blended with the track's data curves) → Magenta RealTime 2 (`mrt2_small`, CPU) generating continuously over the timeline. The mood never sets a pitch. The data melody and all other stems are exactly the `integrate` stems.

| File | What it is |
|---|---|
| `mix_mood.mp3` | integrated arrangement + mood stem, −16 LUFS (compare with `demo/5313/mix.mp3`) |
| `mood.mp3` | the mood stem alone, at its mix level (after ducking) |
| `weights.png` | frames, SigLIP-only weights, blended weights, data curves |
| `mood.png` | weights sent to the model, mood and mix spectrograms (cyan: lead pitch) |
| `summary.json` | the numbers below plus prompts and settings |

## Prompts (anchor: music prompt ← visual text, data link)

- **city**: “night city synthwave” ← “bright city lights at night seen from space” — data: +anthrophony
- **forest**: “forest birdsong ambient” ← “dark forest and countryside at night” — data: +biophony
- **dark**: “dark empty silence” ← “a black empty night with no lights”
- **glow**: “warm glowing electric piano” ← “warm orange glow of lights”
- **change**: “tense pulsing electronic” ← “a map of change in purple and orange”
- **solemn**: “solemn orchestral strings” ← “a calm map of the earth”

## Fidelity (note-level pyin round trip, same method as `arrange.py`)

| Audio | R² | pitch within ½ st |
|---|---|---|
| full mix, no mood (integrate) | 0.977 | 1.00 |
| full mix + mood | 0.977 | 1.00 |
| shuffled control mix, no mood | -0.877 | 0.92 |
| shuffled control mix + mood | -0.873 | 0.92 |

R² change from the mood: -0.000. Mood sits -14 dB under the melody's active RMS before the sidechain (ducked 2.2 dB on average).

## Does the mood follow the picture?

Correlation of each data-linked anchor's weight with its data curve. *Vision only* is SigLIP alone; *blended* is what was sent to the model (it includes the data curve, so it correlates by construction). *Detrended* removes a linear trend from both series first.

| Anchor ↔ curve | vision only | vision, detrended | blended |
|---|---|---|---|
| city ↔ anthrophony | -0.29 | -0.36 | -0.21 |
| … inside “annual series” (n=19) | -0.96 | | 0.93 |
| … inside “change map” (n=22) | -0.45 | | -0.49 |
| … inside “annual series + zooms” (n=30) | 0.28 | | 0.55 |
| … inside “change map + zooms” (n=13) | -0.45 | | -0.42 |
| forest ↔ biophony | 0.49 | 0.51 | 0.70 |
| … inside “annual series” (n=19) | 0.82 | | 0.96 |
| … inside “change map” (n=22) | 0.55 | | 0.52 |
| … inside “annual series + zooms” (n=30) | 0.64 | | 0.93 |
| … inside “change map + zooms” (n=13) | 0.47 | | 0.48 |

## Does the generated audio follow the weights?

MusicCoCa audio embeddings of the raw mood (18 clips of 10 s, every 2 s) against the prompts' text embeddings: mean r over anchors between “clip ↔ prompt similarity” and the weight the model was given = 0.65 (same audio against time-reversed weights: -0.56). The closest prompt is the most-weighted prompt in 94% of clips (chance 17%).

| Anchor | r | share of clips closest |
|---|---|---|
| city | 0.92 | 50% |
| forest | 0.50 | 0% |
| dark | 0.92 | 0% |
| glow | -0.22 | 0% |
| change | 0.82 | 33% |
| solemn | 0.94 | 17% |

## Key

Notes conditioning: `key` (out-of-scale pitches and everything from one semitone under the lead's lowest note upward set OFF; harmony chord tones below that FREE). Chroma share on the track's scale: 0.75 (chance 0.58).
Chroma cosine with the harmony stem per 2 s: 0.81 in place vs 0.60 averaged over the 11 transpositions; best shift 0 st; shift 0 best in 50% of segments → no systematic clash.

## Runtime

mrt2_small on CPU: 309 s for 45 s of audio (411 s per audio minute, 0.15× real time), plus 19 s load/compile.

Reproduce:

```bash
srun -A p32576 -p short -t 00:30:00 --mem=24G -c 8 scripts/mood/weights.sh 5313
srun -A p32576 -p short -t 02:00:00 --mem=48G -c 8 scripts/mood/render.sh mrt2_small 5313   # CPU
sbatch scripts/mood/gpu_base.sbatch                                                     # GPU, base
```

Nothing here has been listened to; every judgement is from the metrics above.
