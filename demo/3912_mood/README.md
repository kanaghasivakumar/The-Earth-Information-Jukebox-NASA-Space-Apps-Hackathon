# 3912 mood pilot: Sea surface currents and temperature

A learned **mood texture** under the unchanged integrated arrangement: video frames → SigLIP style weights over a fixed prompt vocabulary (blended with the track's data curves) → Magenta RealTime 2 (`mrt2_small`, CPU) generating continuously over the timeline. The mood never sets a pitch. The data melody and all other stems are exactly the `integrate` stems.

| File | What it is |
|---|---|
| `mix_mood.mp3` | integrated arrangement + mood stem, −16 LUFS (compare with `demo/3912/mix.mp3`) |
| `mood.mp3` | the mood stem alone, at its mix level (after ducking) |
| `weights.png` | frames, SigLIP-only weights, blended weights, data curves |
| `mood.png` | weights sent to the model, mood and mix spectrograms (cyan: lead pitch) |
| `summary.json` | the numbers below plus prompts and settings |

## Prompts (anchor: music prompt ← visual text, data link)

- **deep**: “deep ocean ambient drone” ← “dark blue deep ocean water” — data: +geophony
- **tropic**: “warm tropical marimba” ← “warm red and orange tropical sea” — data: +warm
- **icy**: “icy glassy pads” ← “cold icy blue and white polar water” — data: −warm
- **flow**: “gentle flowing water ambient” ← “swirling ocean currents and eddies”
- **storm**: “storm percussion” ← “turbulent stormy chaotic swirls”
- **solemn**: “solemn orchestral strings” ← “a dark empty map”

## Fidelity (note-level pyin round trip, same method as `arrange.py`)

| Audio | R² | pitch within ½ st |
|---|---|---|
| full mix, no mood (integrate) | 0.987 | 0.98 |
| full mix + mood | 0.987 | 0.98 |
| shuffled control mix, no mood | -1.026 | 1.00 |
| shuffled control mix + mood | -1.154 | 0.95 |

R² change from the mood: 0.000. Mood sits -14 dB under the melody's active RMS before the sidechain (ducked 0.6 dB on average).

## Does the mood follow the picture?

Correlation of each data-linked anchor's weight with its data curve. *Vision only* is SigLIP alone; *blended* is what was sent to the model (it includes the data curve, so it correlates by construction). *Detrended* removes a linear trend from both series first.

| Anchor ↔ curve | vision only | vision, detrended | blended |
|---|---|---|---|
| deep ↔ geophony | 0.07 | 0.07 | 0.72 |
| tropic ↔ warm | 0.12 | -0.31 | 0.86 |
| icy ↔ 1 − warm | -0.21 | -0.26 | 0.87 |

## Does the generated audio follow the weights?

MusicCoCa audio embeddings of the raw mood (30 clips of 10 s, every 2 s) against the prompts' text embeddings: mean r over anchors between “clip ↔ prompt similarity” and the weight the model was given = 0.24 (same audio against time-reversed weights: -0.14). The closest prompt is the most-weighted prompt in 97% of clips (chance 17%).

| Anchor | r | share of clips closest |
|---|---|---|
| deep | -0.17 | 3% |
| tropic | 0.63 | 0% |
| icy | 0.52 | 0% |
| flow | 0.02 | 97% |
| storm | 0.33 | 0% |
| solemn | 0.09 | 0% |

## Key

Notes conditioning: `key` (out-of-scale pitches and everything from one semitone under the lead's lowest note upward set OFF; harmony chord tones below that FREE). Chroma share on the track's scale: 0.68 (chance 0.58).
Chroma cosine with the harmony stem per 2 s: 0.79 in place vs 0.54 averaged over the 11 transpositions; best shift 0 st; shift 0 best in 47% of segments → no systematic clash.

## Runtime

mrt2_small on CPU: 464 s for 68 s of audio (409 s per audio minute, 0.15× real time), plus 21 s load/compile.

Reproduce:

```bash
srun -A p32576 -p short -t 00:30:00 --mem=24G -c 8 scripts/mood/weights.sh 3912
srun -A p32576 -p short -t 02:00:00 --mem=48G -c 8 scripts/mood/render.sh mrt2_small 3912   # CPU
sbatch scripts/mood/gpu_base.sbatch                                                     # GPU, base
```

Nothing here has been listened to; every judgement is from the metrics above.
