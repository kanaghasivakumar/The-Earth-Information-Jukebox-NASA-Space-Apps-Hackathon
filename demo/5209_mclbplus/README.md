# 5209 with MuseControlLite B+ (no cymbals, bass kept, with Earth voices)

The data lead, the bands and every recorded Earth-voice bed stay exactly at their arranged levels; a
MuseControlLite accompaniment (with its bass) shares the AMT harmony's slot (blend: both 3 dB down, so the AMT bass line stays). **Nothing here has
been listened to**: judged by pyin, HPSS and spectrograms only. Method and table: `demo/learned/README.md`.

- **MCL take:** the pilot's existing render (wosdd-all, control CFG 1.5, seed 42; the mode-B augment
  bed the user rated), not regenerated. Prompt: "Orchestral, warming, solemn strings and piano, slow cinematic build, flute melody".
- **Cymbal removal (disclosed post-processing):** HPSS (margin 1) on the MCL stem only; its percussive
  component above 5 kHz is turned down 15 dB (1 kHz ramp). The data lead, bands and recorded voices
  are untouched, and so is everything under 4 kHz (the bass).

| measure | deterministic mix | old augment | B+ |
|---|---|---|---|
| full-mix R² vs data | 0.966 | 0.966 | 0.967 |
| notes within ½ semitone | 1.00 | 1.00 | 1.00 |
| shuffled control R² (B+: end to end) | -0.551 (lead swap) | -0.705 | -0.638 |
| voice-bed share of K-weighted energy | 0.043 | 0.036 | 0.042 |
| HF (5–16 kHz) percussive energy, full mix (re total, A-weighted) | -35.3 dB | -35.9 dB | -35.4 dB |
| whole-mix 40–250 Hz level at -23 LUFS (bass) | 26.3 dB | 25.5 dB | 25.1 dB |
| HPSS percussive share, full mix | 0.003 | 0.003 | 0.003 |
| MCL stem 40–250 Hz level in the mix (dB) | – | -45.4 | -47.7 |

MCL stem alone, before / after cleanup: HF (5–16 kHz) percussive energy -43.9 dB / -53.8 dB, all HF energy -30.8 dB / -32.7 dB, HPSS percussive share 0.006 / 0.006, 40–250 Hz 27.2 / 27.2 dB.

Files: `mix.mp3` (B+), `control.mp3` (shuffled lead + accompaniment generated from it, identical chain)
`accompaniment.mp3` (the cleaned MCL stem at its level in the mix), `spectrogram.png`, `bplus_report.json`.

Credits: Stable Audio Open 1.0 (Stability AI Community License, "Powered by Stability AI"), MuseControlLite adapters and code (MIT, Tsai et al. 2025), recorded voices as in `jukebox/voices/ATTRIBUTION.md`.
