# Three Voices of the Planet — build spec

> Every note traces back to a value on a NASA colorbar. The planet plays in three voices:
> **geophony** (ocean, thunder, wind), **anthrophony** (city hum), **biophony** (forest chorus).
> The data writes the melody; a model writes the harmony; a listener can check any note.

## 1. Why this wins
| Judging axis | What we show |
|---|---|
| Technical depth | Colorbar inversion → physical values; VAE latent trajectory + motif memory (from `eic-vae/`); Anticipatory Music Transformer infill; corpus-based synthesis; binaural render |
| Musicality | Scale-quantized melodies, model-written harmony, real recorded timbres, recurring motifs |
| Fidelity | Round-trip R² (re-extract pitch/loudness from audio → regress on the data) and a shuffled-data control |
| Accessibility | Sweep/scrub by keyboard, spoken legend + calibration tones, ARIA live values, braille score export |
| Story | Biophony gets drowned out by anthrophony as night lights spread 2012→2023 |

## 2. Architecture

```
SVS API ──► ingest ──► data layer ──────────────► score layer ──► voice layer ──► mix/render ──► jukebox web
(catalog)   (frames)   colorbar→values            data notes fixed   3 corpora +     stems, binaural   sweep, scrub,
                       zonal/regional curves      + AMT infill       FluidSynth      timeline JSON     legend, a11y
                       events / categories        + VAE motif memory tonal layer
                       VAE latent trajectory ─────► form / sections
                                                                         └──► proof: R², shuffled control
```

### 2.1 Data layer (`jukebox/data/`)
- `catalog.py` — SVS API (`https://svs.gsfc.nasa.gov/api/<id>`) → per-page metadata, smallest usable movie, colorbar image, frame-folder URLs, dates.
- `ingest.py` — download the smallest movie, `ffmpeg` → frames at N fps, downsample to a 128×64 (equirectangular) grid. Mask burned-in labels/colorbars.
- `colorbar.py` — sample the colorbar PNG along its long axis → LUT of (CIELAB colour → value). Invert each pixel by nearest LUT entry; reject pixels with ΔE above a threshold as "no data" (land, labels, cloud). Report quantization error. Categorical maps (aerosols) use nearest-palette classification.
- `features.py` — per frame: global mean / p10 / p90, **zonal means in 6 latitude bands** (Planetary Bands), regional means, frame-to-frame change, event detection (lightning: bright transient blobs → (t, lon, lat, intensity)), category fractions. Output a `features.npz` that is a superset of what `eic-vae/music/composer.py` expects (`timestamps`, `activity`, `movement`, `flow_magnitude`, `latent_controls`).
- VAE latent trajectory (reuse `eic-vae`) is used only for **form**: k-means regions → sections + recurring motifs. It never sets pitch.

### 2.2 Score layer (`jukebox/score/`)
- Value → pitch: linear over ~3 octaves (St George et al. 2017), snapped to a scale; polarity toggle (default up = more).
- One voice per latitude band (Planetary Bands, Warming World). Longitude → pan.
- Events → percussion at exact times; categories → instrument programs.
- **Anticipatory Music Transformer** (`stanford-crfm/music-medium-800k`) infills harmony/bass around the fixed data notes (data notes passed as `controls`).
- Output: MIDI + MusicXML (+ braille via FreeDots, stretch).

### 2.3 Voice layer (`jukebox/voices/`)
- Corpora (CC0/CC-BY only, attribution list in `voices/ATTRIBUTION.md`):
  - geophony: MBARI Pacific Sound hydrophone (CC-BY 4.0, `s3://pacific-sound-16khz`), thunder, wind, rain
  - anthrophony: traffic, city hum, power-line hum
  - biophony: dawn chorus, insects, frogs
- Concatenative grain selection: data value → target descriptor (loudness, centroid, flatness) → nearest grains (librosa features, KD-tree).
- Mix level per voice is data-driven (e.g. night-light fraction → anthrophony gain; canopy height → biophony density).

### 2.4 Proof (`jukebox/proof/`)
- Re-extract pitch (pyin) and RMS from the rendered melody stem → R² against the input curve.
- Render the same pipeline on time-shuffled data → A/B pair + metric drop.

### 2.5 Jukebox web (`web/`)
Static site, Web Audio. Pick a visualization → video + synced audio stems. Keyboard: ←/→ scrub, space play, L legend, S sweep mode, P polarity. ARIA live region speaks date + value under the cursor. Spoken legend plays calibration tones (min / mid / max). Headphone binaural toggle.

## 3. Datasets and mappings (first pass)
| SVS id | Visualization | Data type | Melody / pitch | Rhythm / events | Voice | Notes |
|---|---|---|---|---|---|---|
| 3912 | Sea surface currents + temperature | continuous, 0–32 °C colorbar | SST per latitude band | eddy count → motif entries | geophony (hydrophone) | use `orig/` frames, no colorbar overlay |
| 5313 | Night lights 2012–2023 | diverging (purple = more light) | change per region, rising/falling | year boundaries | anthrophony vs biophony | headline story |
| 5011 | Lightning from ISS 2017–2023 | point events | — | each flash = hit at lon/lat | geophony (thunder) | MP4 only |
| 4950 | GEDI forest height | continuous, 0–50 m | canopy height per band | — | biophony density | 60p, 3000 frames |
| 5669 | GEOS aerosols | categorical (salt/dust/smoke/sulfate) | — | — | one instrument per type | single still → sweep |
| 5097 | Hurricane Ian (teammate's VAE) | RGB + flow | latent form | flow → activity | geophony (wind) | already wired in `eic-vae` |

## 3b. What the EIC metadata gives us (see `docs/EIC_CATALOG.md`, 79 pages)
| Metadata (SVS API field) | Musical use |
|---|---|
| `datasets[].date_range`, captions, frame dates | real time axis: day = beat, month = bar, year = phrase; seasonal cycles become meter |
| colorbar images (20 pages) + value ranges in descriptions | pitch range and units for the spoken legend |
| `datasets[].type` (Observed 15 · Model 23 · Compilation 13 · Event List 2) | **provenance → timbre**: observed = acoustic/recorded, model = synthesized. The audience can hear which is measured and which is simulated |
| `keywords` (Human Dimensions 15, Atmosphere 16, Oceans 7, Carbon Cycle and Ecosystems 5, …) | **automatic voice assignment**: Human Dimensions → anthrophony; Atmosphere/Oceans/Hazards → geophony; Ecosystems/Agriculture/Biodiversity → biophony |
| `missions` (Landsat, GPM, OCO-2, GRACE, Aqua/Terra…) | a short signature motif per satellite, sounded when its data enters |
| Captions on 13 pages (narration + "captions_silent" descriptions on 4950, 4993) | spoken legend / audio description text; lyric source |
| Event-list datasets (HURDAT2 tracks, VIIRS fire pixels, ISS lightning) | percussion and melodic tracks with exact timing |

**Datasets the catalog surfaced that fit better than the original list:**
- **5209 Zonal Climate Anomalies 1880–2023 (GISTEMP)**: literally the *Planetary Bands* setup. NASA GISS publishes the zonal numbers as CSV, so the visual and the actual data can be paired. This is the hero for the proof slide.
- **Earth Now dashboard 5147–5154**: daily GEOS-FP/GEOS-CF fields (temperature, wind, precipitation, humidity, PM2.5, O₃, CO, NOx), each with a colorbar and frame sets, updated daily. This gives an **"Earth today" song**: 8 layers of today's planet, regenerated each day. It's the jukebox's live track.
- **5046/5064 Daily Arctic/Antarctic sea ice by year**: one phrase per year, so a shrinking melody is audible.
- **5119 OCO-2 + Keeling curve**: the seasonal sawtooth becomes a rhythm on a rising drone.
- **5058 Florida night lights before/after Hurricane Ian**, with **5097 2022 Hurricane Season** (the teammate's VAE input, dated 2022-06-01 → 11-15): a city voice that goes silent after landfall.
- **14700 Biodiversity (Arctic Animal Movement Archive)**: real animal tracks for the biophony voice.
- **13417 Landsat Croplands** already ships a NASA sonification, so there is precedent inside EIC. Cite it.

## 4. Workstreams (parallel)
1. **Data** — catalog, ingest, colorbar inversion, features (CPU, `short` partition).
2. **Score** — mapping + AMT infill + FluidSynth render (GPU optional; AMT medium runs on CPU slowly).
3. **Voices** — corpus collection, descriptors, grain engine, attribution.
4. **Web + pitch** — jukebox UI, accessibility, deck, user test.

## 5. Milestones
- M1 (first 6 h): one dataset (3912) end-to-end → 30 s WAV with data melody + hydrophone voice.
- M2: proof metrics + shuffled control on 3912.
- M3: 5313 night lights with three-voice story; web jukebox with two tracks.
- M4: lightning + GEDI + aerosols; AMT harmony; spoken legend.
- M5: deck, user test (5–8 people, ≥1 BLV if reachable), polish.

## 6. Open questions
- Does the underlying data exist as files (ECCO2, VIIRS Black Marble, GEDI L3)? If yes, prefer it over pixel inversion for the "proof" slide.
- 5313 colorbar: no numeric range on the page — treat as relative change, say so.
- Licensing: NC model weights are fine for a hackathon; list them on the credits slide.
