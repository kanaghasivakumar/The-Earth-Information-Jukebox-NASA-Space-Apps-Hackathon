# 5058 with MuseControlLite B+ (no cymbals, bass kept, with Earth voices)

The data lead, the bands and every recorded Earth-voice bed stay exactly at their arranged levels; a
MuseControlLite accompaniment (with its bass) shares the AMT harmony's slot (blend: both 3 dB down, so the AMT bass line stays). **Nothing here has
been listened to**: judged by pyin, HPSS and spectrograms only. Method and table: `demo/learned/README.md`.

- **MCL take:** rendered for B+ on CPU (fp32): wosdd-all adapters on Stable Audio Open 1.0, 50 steps,
  text CFG 7.0, control CFG 1.5, seed 42, rhythm control x0.5, melody control x0.0.
- **Prompt:** "Somber ambient in G major, slow strings and warm pads, fading light, warm bass line, no cymbals, no hi-hats, no crash, no ride"
- **Negative prompt:** "cymbals, hi-hats, crash cymbal, ride cymbal, splash cymbal, shaker, tambourine, bright snare"
- **Cymbal removal (disclosed post-processing):** HPSS (margin 1) on the MCL stem only; its percussive
  component above 5 kHz is turned down 15 dB (1 kHz ramp). The data lead, bands and recorded voices
  are untouched, and so is everything under 4 kHz (the bass).

| measure | deterministic mix | old augment | B+ |
|---|---|---|---|
| full-mix R² vs data | 0.988 | – | 0.988 |
| notes within ½ semitone | 1.00 | – | 1.00 |
| shuffled control R² (B+: end to end) | 0.097 (lead swap) | – | 0.115 |
| voice-bed share of K-weighted energy | 0.131 | – | 0.129 |
| HF (5–16 kHz) percussive energy, full mix (re total, A-weighted) | -27.5 dB | – | -27.8 dB |
| whole-mix 40–250 Hz level at -23 LUFS (bass) | 27.6 dB | – dB | 28.8 dB |
| HPSS percussive share, full mix | 0.062 | – | 0.055 |
| MCL stem 40–250 Hz level in the mix (dB) | – | – | -38.7 |

MCL stem alone, before / after cleanup: HF (5–16 kHz) percussive energy -65.6 dB / -70.1 dB, all HF energy -64.7 dB / -69.3 dB, HPSS percussive share 0.001 / 0.001, 40–250 Hz 36.9 / 36.9 dB.

Files: `mix.mp3` (B+), `control.mp3` (shuffled lead + accompaniment generated from it, identical chain)
`accompaniment.mp3` (the cleaned MCL stem at its level in the mix), `spectrogram.png`, `bplus_report.json`.

Credits: Stable Audio Open 1.0 (Stability AI Community License, "Powered by Stability AI"), MuseControlLite adapters and code (MIT, Tsai et al. 2025), recorded voices as in `jukebox/voices/ATTRIBUTION.md`.
