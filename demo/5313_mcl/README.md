# 5313 with MuseControlLite (pilot, `wosdd-all_g1.5`)

Learned layer on top of the deterministic track. **Nothing here has been listened to**: it was judged by
the pyin round trip and spectrograms only. See `demo/learned/README.md` for the method and the comparison.

- **Prompt:** "Night city electronic ambient, warm synth pads, soft pulse, glowing, flute melody"
- **Melody control:** legato tone of the data lead notes (fidelity voices only), 24 notes; top-4 CQT bins per channel (8 x frames), MIDI+1, hop 512.
- **Dynamics control:** half the lead's own loudness envelope, half the voices' `controls.json` curves (geophony, anthrophony, biophony).
- **Rhythm control:** downbeat at each year boundary (24), 4 beats per year (0.15 s).
- **Segments:** 1 x <= 47 s.
- **Model:** wosdd-all adapters on Stable Audio Open 1.0, 50 steps, text CFG 7.0, control CFG 1.5, seed 42, fp16, NVIDIA A100-SXM4-80GB: 5.2 s for 44.8 s of audio.

| file | what | R² vs data | within ½ semitone | shuffled control R² |
|---|---|---|---|---|
| `A_generated.mp3` | mode A: generated audio alone | 0.975 | 1.00 | -1.155 |
| `A_generated_shuffled_control.mp3` | same prompt, seed and settings, melody control from the shuffled lead | | | |
| `B_replace_mix.mp3` | mode B: data-lead stem + bands + events + generated (replaces AMT harmony and recorded voices) | 0.976 | 1.00 | -0.915 end to end, -0.864 lead swap |
| `B_augment_mix.mp3` | mode B: every current stem + generated, 13 dB under the lead | 0.977 | 1.00 | -0.915 end to end, -0.872 lead swap |
| (`demo/5313/mix.mp3`) | current integrated mix, same measurement | 0.977 | 1.00 | -0.877 |

Other seeds, mode A (R², control R²): `wosdd-all_g1.5_seed123` 0.974, -1.172; `wosdd-all_g1.5_seed7` 0.974, -1.160.

Share of spectral energy outside the lead's harmonics while lead notes sound: generated 0.21, current integrated mix 0.11.

`spectrogram.png`: generated CQT with the data notes and the pyin track. `controls.png`: the three controls.
`mcl_report.json`: every number above plus the controls summary.

Credits: Stable Audio Open 1.0 (Stability AI Community License, "Powered by Stability AI"), MuseControlLite adapters and code (MIT, Tsai et al. 2025). See `demo/learned/README.md`.
