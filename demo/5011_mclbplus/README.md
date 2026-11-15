# 5011 with MuseControlLite B+ (no cymbals, bass kept, with Earth voices)

The data lead, the bands and every recorded Earth-voice bed stay exactly at their arranged levels; a
MuseControlLite accompaniment (with its bass) takes the AMT harmony's place (replace). **Nothing here has
been listened to**: judged by pyin, HPSS and spectrograms only. Method and table: `demo/learned/README.md`.

- **MCL take:** rendered for B+ on CPU (fp32): wosdd-all adapters on Stable Audio Open 1.0, 50 steps,
  text CFG 7.0, control CFG 1.5, seed 42, rhythm control x0.5, melody control x1.0.
- **Prompt:** "Stormy ambient, dark strings and low synth pads, slow and brooding, warm bass line, no cymbals, no hi-hats, no crash, no ride"
- **Negative prompt:** "cymbals, hi-hats, crash cymbal, ride cymbal, splash cymbal, shaker, tambourine, bright snare"
- **Cymbal removal (disclosed post-processing):** HPSS (margin 1) on the MCL stem only; its percussive
  component above 5 kHz is turned down 15 dB (1 kHz ramp). The data lead, bands and recorded voices
  are untouched, and so is everything under 4 kHz (the bass).

| measure | deterministic mix | old augment | B+ |
|---|---|---|---|
| full-mix R² vs data | 0.983 | – | 0.983 |
| notes within ½ semitone | 1.00 | – | 1.00 |
| shuffled control R² (B+: end to end) | -1.180 (lead swap) | – | -1.092 |
| voice-bed share of K-weighted energy | 0.035 | – | 0.034 |
| HF (5–16 kHz) percussive energy, full mix (re total, A-weighted) | -31.0 dB | – | -31.1 dB |
| whole-mix 40–250 Hz level at -23 LUFS (bass) | 30.4 dB | – dB | 30.4 dB |
| HPSS percussive share, full mix | 0.010 | – | 0.010 |
| MCL stem 40–250 Hz level in the mix (dB) | – | – | -44.4 |

MCL stem alone, before / after cleanup: HF (5–16 kHz) percussive energy -52.8 dB / -67.4 dB, all HF energy -52.2 dB / -61.4 dB, HPSS percussive share 0.001 / 0.001, 40–250 Hz 28.9 / 28.9 dB.

Data drum hits (5011): onset salience at the hits 6.02 vs 6.09 in the deterministic mix; MCL onsets away from hits 0.59 per s.

Seam: largest 1 s log-mel jump within ±2 s of the crossfade 9.68 dB (1.67 x the track's median, percentile 86).

Files: `mix.mp3` (B+), `control.mp3` (shuffled lead + accompaniment generated from it, identical chain)
`accompaniment.mp3` (the cleaned MCL stem at its level in the mix), `spectrogram.png`, `bplus_report.json`.

Credits: Stable Audio Open 1.0 (Stability AI Community License, "Powered by Stability AI"), MuseControlLite adapters and code (MIT, Tsai et al. 2025), recorded voices as in `jukebox/voices/ATTRIBUTION.md`.
