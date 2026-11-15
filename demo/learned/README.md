# Learned layer pilot: MuseControlLite on top of the data melody

**Question.** Can one learned generative layer make the tracks sound produced without breaking the fidelity
guarantee (the data melody comes back from the final audio)?

**Short answer.** Yes for the melody, with conditions. With the melody control built from the data lead alone,
the generated audio returns the data melody about as well as the deterministic FluidSynth tracks
(R² 0.963–0.986 over 3 seeds, every lead note within ½ semitone, shuffled control R² −0.66 to −1.18). The
catch is what "generated" means here. With the melody-only HF adapters the output is mostly the melody
re-voiced. The melody + dynamics + rhythm adapters (`wosdd-all`) add an accompaniment about as strong as the
current mixes' (see "richness" below). When we add chord tones to the melody control to get more harmony,
fidelity drops in 4 of 6 track × adapter cases (A-mode R² 0.64–0.89). Mode B (generated audio as an accompaniment bed under the
existing flute stem) keeps the integrated mix's R² unchanged on all three tracks. **Nothing has been listened
to.** All judgements come from pyin and spectrograms.

## What was built

- `jukebox/learned/musecontrol.py`: turns a track into controls, generates, and evaluates.
  - **Melody control.** The data lead notes (the fidelity voice only) are rendered as a clean, gap-free tone:
    each note is held to the next onset, with 4 harmonics. Then comes MuseControlLite's own feature: a
    261.6 Hz high-pass, a 128-bin CQT from MIDI 0, and the top-4 bins per frame and channel. This was checked
    against upstream `compute_melody` and agrees on 100 % of frames. The rendered flute stem was not used: its
    rests and reverb tails give the control arbitrary top-4 bins.
  - **Dynamics control** (woSDD-all adapters only). Upstream's `compute_dynamics` format (mel energy in dB,
    hop 160, Savitzky–Golay 279). It is half the lead's own loudness envelope and half the mean of the
    non-zero `controls.json` voice curves, mapped to −24..0 dB.
  - **Rhythm control** (woSDD-all only). madmom-style beat and downbeat activations, synthesised from the
    data's time grid:
    - 5209: one year per data step, with a beat every 2 years and a downbeat every 8.
    - 5313: a downbeat at each year boundary and 4 beats per year. The change-sweep and zoom sections get no
      beats.
    - 3912: a beat at each 0.5 s data frame and a downbeat every 4 beats.
  - **Prompts.** Written by hand from the track metadata, for example 5209: "Orchestral, warming, solemn strings
    and piano, slow cinematic build, flute melody".
  - **Segments.** Up to 47 s each, overlapping by 4 s with an equal-power crossfade (3912 is 2 × 36 s). Each
    segment's controls are cut from the same timeline and padded to the 47.55 s latent window.
  - **Mode A.** The generated audio alone. The same note-level, register-bounded pyin round trip as
    `arrange.fidelity` (same notes, PitchMap, window and offset). The shuffled control uses the same prompt,
    seed and settings, with the melody control built from arrange's shuffled lead (seed 0). Diagnostics:
    octave-folded pitch, MuseControlLite's chroma accuracy, and "richness" (the share of 100 Hz–8 kHz energy
    outside ±60 cents of the lead's first 10 harmonics while lead notes sound).
  - **Mode B.** The current stems plus the generated audio, ducked by arrange's sidechain (8 dB margin, at most
    9 dB, one gain for the whole accompaniment bus), normalised to −16 LUFS, then arrange's limiter.
    - *replace*: lead + bands + events + generated (−10 dB vs the lead). This replaces the AMT harmony and
      the recorded voices.
    - *augment*: every current stem + generated (−13 dB).

    Two controls: *lead swap* (the shuffled lead over the same accompaniment, like arrange's `mix_control`)
    and *end to end* (the shuffled lead over the accompaniment generated from the shuffled lead).
- Two adapter sets, both on `stabilityai/stable-audio-open-1.0`:
  - `hf-melody`: HF `fundwotsai2001/Text-to-Music_control_family/MuseControlLite`, melody only.
  - `wosdd-all`: the GitHub release's melody + dynamics + rhythm adapters (Google Drive, `woSDD-all`).
- Variants: control CFG 1.0 / 1.5 / 2.5 with text CFG 7, 50 steps, fp16. Seeds 7 and 123 were also run at
  1.5. `_full` means the melody control also contains the arranged AMT harmony and the band stems, 6 dB under
  the lead.

## Results

R² is the data value recovered from pyin pitch at each lead-note onset (`arrange.fidelity`). "Within ½ st" is
the share of lead notes within half a semitone of the written pitch. Control R² for mode B is end to end
(lead swap in brackets). "s per audio min" is generation wall time only: the one-off pipeline load (~110 s
cold, ~3 s warm) is excluded, and so is the CPU-side evaluation.

| track | model / control | mode | R² | within ½ st | control R² | non-lead energy | s per audio min | device |
|---|---|---|---|---|---|---|---|---|
| 5209 | current integrated mix (FluidSynth flute + AMT + voices) | reference | 0.966 | 1.00 | -0.551 | 0.10 | – | CPU |
| 5209 | `hf-melody` CFG 1.5, lead-only control | A generated alone | 0.965 (seeds 7/123: 0.965, 0.965) | 1.00 | -0.970 | 0.05 | 9.2 | A100 |
| 5209 | `hf-melody` CFG 1.5, lead-only control | B-replace | 0.966 | 1.00 | -0.657 (-0.563) | – | 9.2 | A100 |
| 5209 | `hf-melody` CFG 1.5, lead-only control | B-augment | 0.966 | 1.00 | -0.758 (-0.515) | – | 9.2 | A100 |
| 5209 | `hf-melody` CFG 1.5, lead + chords control | A generated alone | 0.682 | 0.96 | -0.627 | 0.07 | 11.4 | H100 |
| 5209 | `wosdd-all` CFG 1.5, lead-only control | A generated alone | 0.964 (seeds 7/123: 0.964, 0.963) | 1.00 | -0.922 | 0.12 | 9.9 | A100 |
| 5209 | `wosdd-all` CFG 1.5, lead-only control | B-replace | 0.966 | 1.00 | -0.657 (-0.858) | – | 9.9 | A100 |
| 5209 | `wosdd-all` CFG 1.5, lead-only control | B-augment | 0.966 | 1.00 | -0.705 (-0.552) | – | 9.9 | A100 |
| 5209 | `wosdd-all` CFG 1.5, lead + chords control | A generated alone | 0.869 | 0.99 | -0.601 | 0.18 | 48.0 | H100 |
| 5313 | current integrated mix (FluidSynth flute + AMT + voices) | reference | 0.977 | 1.00 | -0.877 | 0.11 | – | CPU |
| 5313 | `hf-melody` CFG 1.5, lead-only control | A generated alone | 0.975 (seeds 7/123: 0.974, 0.976) | 1.00 | -1.152 | 0.02 | 6.5 | A100 |
| 5313 | `hf-melody` CFG 1.5, lead-only control | B-replace | 0.976 | 1.00 | -0.888 (-0.344) | – | 6.5 | A100 |
| 5313 | `hf-melody` CFG 1.5, lead-only control | B-augment | 0.977 | 1.00 | -0.915 (-0.863) | – | 6.5 | A100 |
| 5313 | `hf-melody` CFG 1.5, lead + chords control | A generated alone | 0.640 | 0.88 | -0.499 | 0.07 | 4.0 | H100 |
| 5313 | `wosdd-all` CFG 1.5, lead-only control | A generated alone | 0.975 (seeds 7/123: 0.974, 0.974) | 1.00 | -1.155 | 0.21 | 7.0 | A100 |
| 5313 | `wosdd-all` CFG 1.5, lead-only control | B-replace | 0.976 | 1.00 | -0.915 (-0.864) | – | 7.0 | A100 |
| 5313 | `wosdd-all` CFG 1.5, lead-only control | B-augment | 0.977 | 1.00 | -0.915 (-0.872) | – | 7.0 | A100 |
| 5313 | `wosdd-all` CFG 1.5, lead + chords control | A generated alone | 0.975 | 1.00 | -0.478 | 0.26 | 4.2 | H100 |
| 3912 | current integrated mix (FluidSynth flute + AMT + voices) | reference | 0.987 | 0.98 | -1.026 | 0.06 | – | CPU |
| 3912 | `hf-melody` CFG 1.5, lead-only control | A generated alone | 0.985 (seeds 7/123: 0.985, 0.985) | 1.00 | -1.114 | 0.03 | 8.5 | A100 |
| 3912 | `hf-melody` CFG 1.5, lead-only control | B-replace | 0.987 | 0.98 | -1.110 (-0.684) | – | 8.5 | A100 |
| 3912 | `hf-melody` CFG 1.5, lead-only control | B-augment | 0.987 | 0.98 | -1.025 (-0.678) | – | 8.5 | A100 |
| 3912 | `hf-melody` CFG 1.5, lead + chords control | A generated alone | 0.986 | 1.00 | -1.113 | 0.07 | 5.2 | H100 |
| 3912 | `wosdd-all` CFG 1.5, lead-only control | A generated alone | 0.985 (seeds 7/123: 0.985, 0.985) | 1.00 | -1.118 | 0.06 | 9.2 | A100 |
| 3912 | `wosdd-all` CFG 1.5, lead-only control | B-replace | 0.987 | 1.00 | -1.111 (-0.690) | – | 9.2 | A100 |
| 3912 | `wosdd-all` CFG 1.5, lead-only control | B-augment | 0.987 | 1.00 | -1.028 (-0.654) | – | 9.2 | A100 |
| 3912 | `wosdd-all` CFG 1.5, lead + chords control | A generated alone | 0.890 | 0.98 | -1.121 | 0.09 | 5.6 | H100 |
| 5209 (first 12 s) | `hf-melody` CPU smoke test, 4 steps | A | see note | | | | 353 (4 steps) | CPU 8 cores fp32 |
| 5209 (first 12 s) | `wosdd-all` CPU smoke test, 4 steps | A | see note | | | | 354 (4 steps) | CPU 8 cores fp32 |

The full grid (CFG 1.0/1.5/2.5, both adapters, both controls, seeds) is in `all_results.md`.

## Reading the numbers

- **Mode A, lead-only control: the melody is held as tightly as the deterministic tracks.** On 5209, 5313 and
  3912 the R² is within 0.003 of the integrated mix, all notes are within ½ semitone, and the shuffled
  controls stay strongly negative. On this evidence the fidelity guarantee survives.
- **`hf-melody` gives a thin result.** Its generated audio is essentially the melody re-voiced
  (piano/synth-like attacks, 2–5 % non-lead energy at CFG 1.5; the integrated mixes have 6–11 %). Long held
  lead notes come out as near-pure drones (5313, 7–21 s). `wosdd-all` puts clearly more underneath: 6–21 %
  non-lead energy (5209 0.12 vs the mix's 0.10, 5313 0.21 vs 0.11, 3912 0.06 vs 0.06), with bass/percussive
  texture in the CQT (`demo/<id>_mcl/spectrogram.png`). The metric only counts energy, though: whether it
  sounds produced is untested.
- **Asking for more harmony through the control loosens the melody.** With `_full` (chord tones in the
  control), A-mode R² drops to 0.68 (5209 `hf-melody`), 0.64 (5313 `hf-melody`), 0.87 (5209 `wosdd-all`)
  and 0.89 (3912 `wosdd-all`). That is well below ~0.97. It holds only for 5313 `wosdd-all` (0.975) and
  3912 `hf-melody` (0.986). Richness rises only a little (5209 0.12 → 0.18, 5313 0.21 → 0.26).
- **Mode B is safe everywhere.** Its R² equals the current integrated mix's to the third decimal, because
  the deterministic flute stem is still on top and the sidechain keeps the bed 8 dB under it. The lead-swap
  control can be weaker (5313 `hf-melody` B-replace: −0.34, against −0.89 end to end), because the generated
  bed itself carries the real melody. The end-to-end control (shuffled data through the whole pipeline) is
  the fair one, and it stays at −0.66 to −1.11.
- **Not tested.** Musical quality, seams at the 3912 segment boundary (~32–36 s, where the texture visibly
  changes), and prompt adherence. The pilot proves fidelity, not taste.
- **Runtime.**
  - GPU: about 5–10 s per 47 s window at 50 steps on A100/H100, or about 4–12 s of compute per minute of
    audio (the first call is slower).
  - CPU (8 cores, fp32): ~70 s per 47.55 s window at 4 steps; 50 steps would take ~15 min per window. Every
    call computes the full 47.55 s latent, however short the clip. Four steps sound unusable (smoke-test
    mode A R² 0.48 on 12 s).

## Recommendation

1. **Expand, but only as mode A with the lead-only control, or as mode B, and keep the deterministic mix as
   the reference track.** Use `wosdd-all` at control CFG 1.5: it is the richest setting that still holds
   R² ≥ 0.963 on all three pilots and three seeds.
2. **Tracks.** It works best where the lead moves often:
   - 5209: the proof track, so play the learned version next to it, never instead of it.
   - 3912
   - 5176 and 4950 (same single-flute structure, so expect the same result)

   On 5313 the long held "years" notes become drones. Use mode B there, or condition on the whole melody group
   (years + change-sweep) first.
3. **Don't** chase richness by putting chords into the melody control (`_full`). It costs R² 0.1–0.3.
   Don't use it for 5669/5011 (no single lead) or 5058 (0.12 s notes, below the control's 512-hop resolution).
4. **Before it goes in the pitch, someone has to listen.** Check the crossfade seam on 3912 and whether the
   wosdd-all percussion clashes with the data rhythm.

## Licences (for the credits slide)

| Component | Licence | What applies to our outputs |
|---|---|---|
| Stable Audio Open 1.0 weights (`stabilityai/stable-audio-open-1.0`, gated; we accepted it as user `kanagha`) | Stability AI Community License (5 July 2024) | Free for research and non-commercial use, and commercial use under US$1M revenue with registration. We own the outputs (§4c(iii)). Outputs may not be used to train a foundation model (§4b). The Acceptable Use Policy applies. When distributing a product that uses the model, show **"Powered by Stability AI"** and ship the licence and notice. Put the line on the credits slide and the web About text anyway. The weights are not redistributed. |
| SAO training data | Freesound (CC0 / CC-BY / CC-Sampling+) and Free Music Archive. Attribution CSVs are in the model repo | Nothing extra for outputs. Cite as "trained on Freesound and FMA (CC)". |
| MuseControlLite code and adapters (Tsai et al., ICML 2025; GitHub `fundwotsai2001/MuseControlLite`; HF `fundwotsai2001/Text-to-Music_control_family`) | MIT | Keep the MIT notice if we ship the code. Cite the paper (arXiv 2506.18729). The adapters were trained on MTG-Jamendo (CC licences). |
| diffusers (`fundwotsai2001/diffusers@v0.31.0-release` fork) and the T5-base text encoder inside SAO | Apache-2.0 | Notice only if redistributed. |
| Everything deterministic (data melody, FluidR3_GM, AMT harmony, recorded voices) | As in the existing credits | Unchanged |

## Reproduce

```bash
bash scripts/mcl_env_setup.sh                     # env "mcl" (torch 2.4.1 cu121, diffusers fork, madmom): 6.7 GB
# weights (gated SAO: accept the licence on HF first): 5.3 GB diffusers parts + 340 MB adapters
HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HOME=/projects/e32706/omb8654/hf_cache python -c "from huggingface_hub import snapshot_download as s; s('stabilityai/stable-audio-open-1.0', allow_patterns=['model_index.json','LICENSE.md','projection_model/*','scheduler/*','text_encoder/*','tokenizer/*','transformer/*','vae/*']); s('fundwotsai2001/Text-to-Music_control_family', allow_patterns=['MuseControlLite/*','*.py'])"
# woSDD-all adapters: gdown the 4 files of Drive folder 1Q9B333jcq1czA11JKTbM-DHANJ8YqGbP/woSDD-all into
#   /projects/e32706/omb8654/assets/MuseControlLite/checkpoints/woSDD-all/
sbatch scripts/mcl_gpu.sbatch                                                  # lead-only control, CFG 1.0/1.5/2.5
sbatch --export=ALL,COND=full,GUIDANCE=1.5 scripts/mcl_gpu.sbatch              # chord tones in the control
sbatch --export=ALL,GUIDANCE=1.5,SEEDS="7 123" scripts/mcl_gpu.sbatch          # seed check
srun -A p32576 -p short -t 04:00:00 --mem=48G -c 8 bash scripts/mcl_cpu_smoke.sh 5209   # CPU smoke test
python scripts/mcl_summary.py 5209 5313 3912 --export wosdd-all_g1.5 wosdd-all_g1.5 wosdd-all_g1.5
```

The inputs are copied from `jukebox-wt/integrate/outputs/<id>` (unchanged). The full outputs are in
`outputs/<id>_mcl/` (git-ignored). Small MP3s, the spectrogram and the report are in `demo/<id>_mcl/`.
