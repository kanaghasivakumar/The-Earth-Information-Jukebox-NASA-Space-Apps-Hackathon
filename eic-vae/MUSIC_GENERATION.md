# NASA Earth Jukebox

The standalone music pipeline reuses the saved VAE and NASA frame-pair dataset. It does not train, replace, or modify the VAE. The encoder's posterior mean is extracted once per invocation, then the same latent and visual-feature trajectories are reused for every requested style.

## Run

From `eic-vae/`, install requirements and keep the old default composition:

```sh
python3 -m pip install -r requirements.txt
python3 generate_music.py --checkpoint checkpoints/best_vae.pt --data data/raw
```

Choose one style or generate several in one pass:

```sh
python3 generate_music.py --data data/raw --style jazz --memory
python3 generate_music.py --data data/raw --styles ambient cinematic classical jazz electronic --memory
```

The original no-style invocation retains the existing default score and output paths. Explicit styles are written to `results/<style>/`; multiple styles share one VAE inference pass and write `results/style_comparison.csv`.

## Styles and configuration

The registry provides `ambient`, `cinematic`, `classical`, `jazz`, and `electronic`. Presets configure tempo, scale, harmony, instruments, rhythmic patterns, bass, dynamics, and motif shape. Pass `--tempo`, `--key`, and `--scale` to override a preset. `--timing-mode match_video` follows the source timestamps; `--timing-mode fixed --target-duration N` resamples the controls and clips events at N seconds.

An optional YAML file can set defaults. Command-line values override it:

```yaml
music:
  styles: [ambient, cinematic, jazz]
  tempo: null
  key: C
  scale: null
  timing_mode: match_video
  seed: 42
memory:
  enabled: true
  clusters: 4
  motif_bars: 2
  variation_strength: 0.25
  minimum_cluster_duration: 2.0
  hysteresis: 0.5
output:
  directory: results
  soundfont: null
```

Run it with `python3 generate_music.py --config music_config.yml`. JSON configuration is also accepted.

## Musical memory

Enable memory with `--memory` (or `memory.enabled: true`). The full latent trajectory is standardized by coordinate and clustered deterministically with scikit-learn K-means; the default request is four clusters. The effective cluster count is reduced for constant or short trajectories. Timestamp-based dwell and hysteresis suppress brief cluster changes. Cluster labels describe recurring model-space regions; they are not scientific categories.

Each cluster is assigned a style-shaped, scale-relative motif. When a region returns at a motif phrase boundary, its motif is reused with a deterministic, bounded variation. Set `--variation-strength 0` for exact recalls or any value from 0 to 1 for increasingly varied returns. `--clusters`, `--motif-bars`, `--minimum-cluster-duration`, `--cluster-hysteresis`, and `--variation-strength` are command-line overrides.

With memory enabled, each style directory also includes `earth_jukebox_memory.png`, raw/stabilized cluster-label arrays, and the usual feature and piano-roll plots. The memory plot shows the latent trajectory (directly in 2D or projected to two principal components), cluster assignments, transition times, active motifs, and returning motifs (`R`).

## Comparisons and files

Each style produces `earth_jukebox.mid`, `earth_jukebox.wav`, `earth_jukebox_features.png`, `earth_jukebox_piano_roll.png`, and a short description. `style_comparison.csv` reports total note count, average note duration, pitch range, rhythmic density, average velocity, motif repeats/variations, stabilized cluster transitions, and exploratory correlations between score energy/movement and percussion/flow. Correlation does not establish scientific validity.

The pipeline uses FluidSynth if installed with a readable local SoundFont supplied through `--soundfont /path/to/local.sf2`. Otherwise it exports MIDI and renders a basic 44.1 kHz stereo WAV with the built-in synthesizer. It never downloads a SoundFont or music samples.

The shared scientific conditioning remains the VAE latent trajectory, latent movement, and optical-flow magnitude. Styles interpret these same signals differently. Latent dimensions are learned visual patterns with no guaranteed physical meaning; named EIC facts require explicit metadata and sonification mappings, not cluster labels alone.
