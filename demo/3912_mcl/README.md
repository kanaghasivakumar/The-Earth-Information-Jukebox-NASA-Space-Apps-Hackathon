# 3912 with MuseControlLite (pilot, `wosdd-all_g1.5`)

Learned layer on top of the deterministic track. **Nothing here has been listened to**: it was judged by
the pyin round trip and spectrograms only. See `demo/learned/README.md` for the method and the comparison.

- **Prompt:** "Oceanic ambient, flowing, soft pads and strings, gentle waves, flute melody"
- **Melody control:** legato tone of the data lead notes (fidelity voices only), 43 notes; top-4 CQT bins per channel (8 x frames), MIDI+1, hop 512.
- **Dynamics control:** half the lead's own loudness envelope, half the voices' `controls.json` curves (geophony).
- **Rhythm control:** beat every data frame (0.50 s), downbeat every 4.
- **Segments:** 2 x <= 47 s, 4 s equal-power crossfade.
- **Model:** wosdd-all adapters on Stable Audio Open 1.0, 50 steps, text CFG 7.0, control CFG 1.5, seed 42, fp16, NVIDIA A100-SXM4-80GB: 10.5 s for 68.0 s of audio.

| file | what | R² vs data | within ½ semitone | shuffled control R² |
|---|---|---|---|---|
| `A_generated.mp3` | mode A: generated audio alone | 0.985 | 1.00 | -1.118 |
| `A_generated_shuffled_control.mp3` | same prompt, seed and settings, melody control from the shuffled lead | | | |
| `B_replace_mix.mp3` | mode B: data-lead stem + bands + events + generated (replaces AMT harmony and recorded voices) | 0.987 | 1.00 | -1.111 end to end, -0.690 lead swap |
| `B_augment_mix.mp3` | mode B: every current stem + generated, 13 dB under the lead | 0.987 | 1.00 | -1.028 end to end, -0.654 lead swap |
| (`demo/3912/mix.mp3`) | current integrated mix, same measurement | 0.987 | 0.98 | -1.026 |

Other seeds, mode A (R², control R²): `wosdd-all_g1.5_seed123` 0.985, -1.035; `wosdd-all_g1.5_seed7` 0.985, -1.118.

Share of spectral energy outside the lead's harmonics while lead notes sound: generated 0.06, current integrated mix 0.06.

`spectrogram.png`: generated CQT with the data notes and the pyin track. `controls.png`: the three controls.
`mcl_report.json`: every number above plus the controls summary.

Credits: Stable Audio Open 1.0 (Stability AI Community License, "Powered by Stability AI"), MuseControlLite adapters and code (MIT, Tsai et al. 2025). See `demo/learned/README.md`.
