# 3912 with MuseControlLite B+ (no cymbals, bass kept, with Earth voices)

The data lead, the bands and every recorded Earth-voice bed stay exactly at their arranged levels; a
MuseControlLite accompaniment (with its bass) shares the AMT harmony's slot (blend: both 3 dB down, so the AMT bass line stays). **Nothing here has
been listened to**: judged by pyin, HPSS and spectrograms only. Method and table: `demo/learned/README.md`.

- **MCL take:** the pilot's existing render (wosdd-all, control CFG 1.5, seed 42; the mode-B augment
  bed the user rated), not regenerated. Prompt: "Oceanic ambient, flowing, soft pads and strings, gentle waves, flute melody".
- **Cymbal removal (disclosed post-processing):** HPSS (margin 1) on the MCL stem only; its percussive
  component above 5 kHz is turned down 15 dB (1 kHz ramp). The data lead, bands and recorded voices
  are untouched, and so is everything under 4 kHz (the bass).

| measure | deterministic mix | old augment | B+ |
|---|---|---|---|
| full-mix R² vs data | 0.987 | 0.987 | 0.987 |
| notes within ½ semitone | 0.98 | 1.00 | 1.00 |
| shuffled control R² (B+: end to end) | -1.026 (lead swap) | -1.028 | -1.026 |
| voice-bed share of K-weighted energy | 0.027 | 0.022 | 0.026 |
| HF (5–16 kHz) percussive energy, full mix (re total, A-weighted) | -42.3 dB | -42.6 dB | -42.4 dB |
| whole-mix 40–250 Hz level at -23 LUFS (bass) | 26.0 dB | 25.4 dB | 24.9 dB |
| HPSS percussive share, full mix | 0.001 | 0.001 | 0.001 |
| MCL stem 40–250 Hz level in the mix (dB) | – | -50.3 | -52.9 |

MCL stem alone, before / after cleanup: HF (5–16 kHz) percussive energy -41.2 dB / -47.9 dB, all HF energy -22.6 dB / -23.5 dB, HPSS percussive share 0.002 / 0.002, 40–250 Hz 22.6 / 22.6 dB.

Seam: largest 1 s log-mel jump within ±2 s of the crossfade 7.74 dB (2.81 x the track's median, percentile 90). pilot take reused (triage): its 4 s crossfade at 32-36 s is unchanged.

Files: `mix.mp3` (B+), `control.mp3` (shuffled lead + accompaniment generated from it, identical chain)
`accompaniment.mp3` (the cleaned MCL stem at its level in the mix), `spectrogram.png`, `bplus_report.json`.

Credits: Stable Audio Open 1.0 (Stability AI Community License, "Powered by Stability AI"), MuseControlLite adapters and code (MIT, Tsai et al. 2025), recorded voices as in `jukebox/voices/ATTRIBUTION.md`.
