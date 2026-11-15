# 5669 with MuseControlLite B+ (no cymbals, bass kept, with Earth voices)

The data lead, the bands and every recorded Earth-voice bed stay exactly at their arranged levels; a
MuseControlLite accompaniment (with its bass) takes the AMT harmony's place (replace). **Nothing here has
been listened to**: judged by pyin, HPSS and spectrograms only. Method and table: `demo/learned/README.md`.

- **MCL take:** rendered for B+ on CPU (fp32): wosdd-all adapters on Stable Audio Open 1.0, 50 steps,
  text CFG 7.0, control CFG 1.5, seed 42, rhythm control x0.5, melody control x1.0.
- **Prompt:** "Airy ambient, drifting pads and soft strings, hazy and slow, warm bass line, no cymbals, no hi-hats, no crash, no ride"
- **Negative prompt:** "cymbals, hi-hats, crash cymbal, ride cymbal, splash cymbal, shaker, tambourine, bright snare"
- **Cymbal removal (disclosed post-processing):** HPSS (margin 1) on the MCL stem only; its percussive
  component above 5 kHz is turned down 15 dB (1 kHz ramp). The data lead, bands and recorded voices
  are untouched, and so is everything under 4 kHz (the bass).

| measure | deterministic mix | old augment | B+ |
|---|---|---|---|
| voice-bed share of K-weighted energy | 0.115 | – | 0.114 |
| HF (5–16 kHz) percussive energy, full mix (re total, A-weighted) | -27.1 dB | – | -27.1 dB |
| whole-mix 40–250 Hz level at -23 LUFS (bass) | 32.2 dB | – dB | 32.3 dB |
| HPSS percussive share, full mix | 0.018 | – | 0.018 |
| MCL stem 40–250 Hz level in the mix (dB) | – | – | -41.5 |

MCL stem alone, before / after cleanup: HF (5–16 kHz) percussive energy -71.3 dB / -78.8 dB, all HF energy -69.5 dB / -74.2 dB, HPSS percussive share 0.007 / 0.007, 40–250 Hz 35.4 / 35.4 dB.

Per aerosol type, r between note data fraction and its level in the mix (deterministic → B+): black carbon 0.45 → 0.44, dust 0.89 → 0.87, sea salt 0.35 → 0.37, sulfate 0.61 → 0.63.

Files: `mix.mp3` (B+)
`accompaniment.mp3` (the cleaned MCL stem at its level in the mix), `spectrogram.png`, `bplus_report.json`.

Credits: Stable Audio Open 1.0 (Stability AI Community License, "Powered by Stability AI"), MuseControlLite adapters and code (MIT, Tsai et al. 2025), recorded voices as in `jukebox/voices/ATTRIBUTION.md`.
