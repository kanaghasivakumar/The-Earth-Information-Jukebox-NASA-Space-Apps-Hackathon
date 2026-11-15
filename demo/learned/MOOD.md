# Learned mood layer (pilot): video frames → style weights → Magenta RealTime

**Question.** Can a learned "mood" texture, steered by what the video shows, sit under the data melody
without hurting it? This is the honest version of Magenta's "Space DJ" idea, where weighted text prompts steer a
live music model. Here the weights come from the frames and the data curves, not from a DJ.

**Short answer.** The layer is **safe**: the full-mix round trip does not move (ΔR² ≤ 0.0004 on all three
pilots), the mood stays in the track's key, and Magenta RT follows the weights it is given.
But the **vision half does not reliably track what the data shows**. SigLIP sees section changes and overall
trends, not the story the data tells (for example, more light on the map). The mood follows the data only where
the data curve is blended in. **Recommendation: do not expand the vision-steered mood to more tracks.**
If a mood layer is kept, drive its weights from the data curves, label it "atmosphere, no data", and listen to it first.

## What was built

```
frames (ffmpeg, 2 fps) ─► SigLIP base-patch16-224 ─► cos(frame, visual text of each anchor)
                                                       │ minus 0.7 × the anchor's mean over the track, softmax(·/0.01)
controls.json / data curve ─► + 2.0 × curve on linked anchors (city ← anthrophony, forest ← biophony, …)
                                                       │ zero-phase smoothing 1.5 s ─► weights.json
weights ─► Σ w_k · MusicCoCa(music prompt_k)  per 0.4 s chunk ─► Magenta RT 2 (one streaming state, 2 s pre-roll)
harmony chords + key ─► notes conditioning: out-of-scale pitches OFF, lead register OFF, chord tones FREE
mood stem ─► -14 dB under the melody, 80 Hz HPF, same sidechain as arrange.py ─► + integrate stems ─► -16 LUFS ─► pyin round trip
```

- `jukebox/learned/mood.py`: frames → weights (`outputs/<id>_mood/weights.json`, `weights.png`). Each anchor
  has a *music* prompt (for MusicCoCa) and a *visual* text (for SigLIP), because SigLIP scores pictures, not
  genres. The vocabularies are `ocean` (3912), `lights` (5313) and `climate` (5209), six anchors each.
- `jukebox/learned/magenta_mood.py`: generation, key conditioning, mix, fidelity, key check, and a
  check of whether the audio follows the weights (MusicCoCa audio embeddings vs the prompts).
- `scripts/mood/{weights.sh,render.sh,gpu_base.sbatch,ablation.sh,stress.sh,export_demo.py}`; env `mrt`
  (`/projects/e32706/omb8654/conda/envs/mrt`: magenta-rt 2.0.3, jax 0.10.2 with CUDA 12, torch 2.15 CPU,
  transformers 5.23). Weights are in `/projects/e32706/omb8654/magenta_home` (about 12 GB) and the env is about 8 GB.

```bash
srun -A p32576 -p short -t 00:30:00 --mem=24G -c 8 scripts/mood/weights.sh 5313 3912 5209
srun -A p32576 -p short -t 02:00:00 --mem=48G -c 8 scripts/mood/render.sh mrt2_small 5313   # per track, CPU
sbatch scripts/mood/gpu_base.sbatch                                                     # base model, GPU
source scripts/mood/env.sh && python scripts/mood/export_demo.py                        # demo/<id>_mood + table
```

## Results

The integrate stems are unchanged. "Without mood" is the integrate `mix.wav`, measured again with the same
code (it reproduces the integrate numbers exactly). Control is the shuffled-lead mix, scored against the real
data. Weight–data r uses the anchor the story predicts should move. It is reported for vision alone, then
detrended (a linear trend removed from both series), then blended (data included, so this one correlates by
construction). The "audio follows weights" column is the mean r over anchors between each 10 s clip's MusicCoCa
similarity to a prompt and that prompt's weight, with the same audio against time-reversed weights in brackets.

| Track | Model (device) | Full-mix R² without mood | with mood | Control R² (mix + mood; no mood) | Weight–data r: anchor, vision only (detrended) / blended | Audio follows weights: mean r (reversed) | Gen. s per audio min | Key clash? |
|---|---|---|---|---|---|---|---|---|
| 5313 | mrt2_small (CPU) | 0.977 | 0.977 | −0.87; −0.88 | city↔anthrophony: −0.29 (−0.36) / −0.21 | 0.65 (−0.56) | 411 | no (scale share 0.75) |
| 3912 | mrt2_small (CPU) | 0.987 | 0.987 | −1.15; −1.03 | tropic↔global SST: 0.12 (−0.31) / 0.86 | 0.24 (−0.14) | 409 | no (scale share 0.68) |
| 5209 | mrt2_small (CPU) | 0.966 | 0.966 | −0.63; −0.55 | warm↔anomaly: 0.77 (0.14) / 0.87 | 0.17 (−0.08) | 346 | no (scale share 0.77) |

`mood_table.md` is the same table written by `export_demo.py`. It gains base-model rows when the GPU job (or
the CPU base run) finishes; see "Pending" below. Per-track detail is in `demo/<id>_mood/README.md`.

### Does the mood track the visuals?

- **5313 night lights: no.** The anchor that should rise as lights spread ("night city synthwave") does not.
  Vision alone, it correlates −0.29 with the anthrophony curve, and −0.96 inside the "annual series"
  segment, where it is drowned by "solemn" and "dark". SigLIP does find the video's **sections**: the change
  map turns on "tense pulsing electronic" and the zooms turn on "city". That is a cut detector, not light
  growth. The forest anchor does follow biophony (vision 0.49, detrended 0.51, 0.82 inside the annual series),
  but the city anchor does not. With the data blend, city follows anthrophony inside the annual series
  (0.93), because the data put it there.
- **3912 SST: weak.** "Flowing water" wins about 50 % of the weight everywhere, because the picture barely
  changes globally. On the seasons (northern minus southern band SST), vision gives the tropical anchor
  r 0.48 and the icy anchor −0.04. Against the global-mean SST the melody plays, the tropical anchor gets 0.12
  (−0.31 detrended). The data blend makes tropical and icy follow global SST (0.86 / 0.87). That curve peaks in
  March, so the blended "warm" anchor runs against the northern seasons (−0.39).
- **5209 zonal anomalies: trend only.** Vision correlates 0.77–0.91 with the anomaly, but the chart fills in
  over time, so most of that is shared trend (the weights vs time give |r| 0.83–0.94). After detrending, only
  "hot dark red heat" keeps a real signal (0.58). "Warm" drops to 0.14, and "icy" goes the wrong way (−0.42).

### Does Magenta RT obey the weights and the key?

- **Weights: yes, when they move.** In 5313 the generated audio's MusicCoCa similarity follows the weights
  (mean r 0.65, against −0.56 for reversed weights). The most-weighted prompt is the closest prompt in
  94 % of clips (chance 17 %). In 3912 and 5209 the weights hardly move, so the r values are small (0.24, 0.17).
- **Key: yes, with notes conditioning.** With the 128-pitch notes input (out-of-scale pitches and the lead
  register OFF, harmony chord tones FREE), the mood's chroma matches the harmony stem better in place than at
  any of the 11 transpositions: cos 0.79–0.81 against 0.50–0.60. Its chroma share on the track's scale is
  0.68–0.77 (chance 0.58). Ablation (3912) without notes gives cos 0.64 against 0.60, a best shift of 10 semitones and a scale share of
  0.61, so it barely beats chance.

### What protects the lead (3912 stress test, small model)

| Mood level vs melody | Sidechain | Notes conditioning | Full-mix R² | Lead notes within ½ st |
|---|---|---|---|---|
| −14 dB (default) | yes | key | 0.987 | 98 % |
| −14 dB | yes | none | 0.987 | 100 % |
| −6 dB | yes | key | 0.987 | 100 % |
| −6 dB | **no** | key | 0.974 | 93 % |
| 0 dB | **no** | key | 0.722 | 79 % |
| 0 dB | **no** | none | 0.846 | 81 % |

The sidechain and the level protect fidelity. With ducking, a mood 6 dB louder ends up only 3.5 dB louder
in the mix. Notes conditioning keeps the mood in key, but it does not protect the lead: a mood that is in key
at equal level can be *worse* for pyin than a random one, because its partials land on the lead's pitch classes.
For comparison, the integrate mix measures 0.987 (no mood).

### Runtime

| Model | Device | s per audio minute | Real-time factor |
|---|---|---|---|
| mrt2_small (230 M) | CPU, 8 cores (Quest `short`) | 346–411 | 0.15–0.17× |
| mrt2_base (2.4 B) | CPU, 8 cores | about 4,500 (5313: 757 s for the first 10 s) | about 0.013× |
| mrt2_base (2.4 B) | GPU (gengpu) | pending | pending |

## Caveats

- **Nothing has been listened to.** "Mood follows the weights" is measured by MusicCoCa, the same model
  family that conditioned the generation, so it is a self-consistency check and not a perceptual one.
- The mood is quiet: about 17 dB under the lead's active RMS after ducking, which is about the level of the
  recorded voices. A louder mood was not tested by ear.
- The vocabularies, visual texts, τ = 0.01, debias = 0.7 and β = 2 were set once by hand, with no tuning
  against the correlations. A different vocabulary would give different numbers.
- 3912's geophony curve is frame-to-frame movement, and its "warm" curve is the global-mean SST from colorbar
  inversion (±1–2 °C absolute; see `demo/3912/README.md`).
- Magenta RT 2 adds about 2 s of pre-roll (discarded) and changes style at 0.4 s chunk boundaries. It was
  trained with style changes about every 8 s, so faster weight changes may be smoothed by the model.

## Pending

- GPU job **8745305** (`scripts/mood/gpu_base.sbatch`, `-A e32706 -p gengpu`, base model, all three pilots)
  was still queued at the time of writing; its estimated start was 2026-11-15 07:12. When it finishes, run
  `python scripts/mood/export_demo.py`. The demo folders switch to the base render, and `mood_table.md` gains the
  base rows.

## Credits slide: licences

| Component | Licence | Credit / citation |
|---|---|---|
| Magenta RealTime 2 code (`magenta-rt` 2.0.3, github.com/magenta/magenta-realtime) | Apache-2.0 | © 2026 Google LLC |
| Magenta RealTime 2 weights, `google/magenta-realtime-2` (mrt2_small, mrt2_base, MusicCoCa, SpectroStream) | **CC BY 4.0** (attribution required) | "Magenta RealTime 2 by Google (CC BY 4.0)". Cite *Live Music Models*, Magenta team, arXiv:2508.04651 (NeurIPS Creative AI 2025). Google claims no rights in generated outputs, and users are responsible for their use. The terms also require responsible use with no infringing content. |
| SigLIP `google/siglip-base-patch16-224` | Apache-2.0 | Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV 2023 |
| JAX, Flax, transformers | Apache-2.0 | |
| PyTorch | BSD-3-Clause | |
| NASA SVS visualizations 5209, 5313, 3912 | NASA media guidelines (not copyrighted; credit requested) | NASA Scientific Visualization Studio |

The mood stems are generated audio, and Google claims no rights in them. The project's other credits (AMT, FluidR3_GM,
recorded corpora) are unchanged; see `jukebox/voices/ATTRIBUTION.md` and `jukebox/render/SOUNDFONT.md`.
