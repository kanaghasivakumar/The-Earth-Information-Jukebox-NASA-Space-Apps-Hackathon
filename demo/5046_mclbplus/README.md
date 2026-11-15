# 5046 with MuseControlLite B+ (no cymbals, bass kept, with Earth voices)

The data lead, the bands and every recorded Earth-voice bed stay exactly at their arranged levels; a
MuseControlLite accompaniment (with its bass) shares the AMT harmony's slot (blend: both 3 dB down, so the AMT bass line stays). **Nothing here has
been listened to**: judged by pyin, HPSS and spectrograms only. Method and table: `demo/learned/README.md`.

- **MCL take:** rendered for B+ on CPU (fp32): wosdd-all adapters on Stable Audio Open 1.0, 50 steps,
  text CFG 7.0, control CFG 1.5, seed 42, rhythm control x0.5, melody control x1.0.
- **Prompt:** "Arctic ambient, glassy pads and slow strings, cold and spacious, warm bass line, no cymbals, no hi-hats, no crash, no ride"
- **Negative prompt:** "cymbals, hi-hats, crash cymbal, ride cymbal, splash cymbal, shaker, tambourine, bright snare"
- **Cymbal removal (disclosed post-processing):** HPSS (margin 1) on the MCL stem only; its percussive
  component above 5 kHz is turned down 15 dB (1 kHz ramp). The data lead, bands and recorded voices
  are untouched, and so is everything under 4 kHz (the bass).

| measure | deterministic mix | old augment | B+ |
|---|---|---|---|
| full-mix R² vs data | 0.990 | – | 0.990 |
| notes within ½ semitone | 1.00 | – | 1.00 |
| shuffled control R² (B+: end to end) | -1.330 (lead swap) | – | -1.247 |
| voice-bed share of K-weighted energy | 0.036 | – | 0.035 |
| HF (5–16 kHz) percussive energy, full mix (re total, A-weighted) | -36.5 dB | – | -36.7 dB |
| whole-mix 40–250 Hz level at -23 LUFS (bass) | 25.2 dB | – dB | 23.9 dB |
| HPSS percussive share, full mix | 0.001 | – | 0.001 |
| MCL stem 40–250 Hz level in the mix (dB) | – | – | -50.2 |

MCL stem alone, before / after cleanup: HF (5–16 kHz) percussive energy -64.7 dB / -81.3 dB, all HF energy -56.1 dB / -58.0 dB, HPSS percussive share 0.002 / 0.002, 40–250 Hz 25.7 / 25.7 dB.

Files: `mix.mp3` (B+), `control.mp3` (shuffled lead + accompaniment generated from it, identical chain)
`accompaniment.mp3` (the cleaned MCL stem at its level in the mix), `spectrogram.png`, `bplus_report.json`.

Credits: Stable Audio Open 1.0 (Stability AI Community License, "Powered by Stability AI"), MuseControlLite adapters and code (MIT, Tsai et al. 2025), recorded voices as in `jukebox/voices/ATTRIBUTION.md`.
