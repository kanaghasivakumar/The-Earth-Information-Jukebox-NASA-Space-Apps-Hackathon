# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Next.js (App Router) web app in eic-vae/eic-sonification/frontend, talking to a local FastAPI generator (eic-vae/server.py) for uploads.

## Users

Hackathon judges and NASA staff skimming for two minutes; blind and low-vision listeners; students and teachers. All want to hear music made from NASA Earth visualizations and understand what they are hearing.

## Product Purpose

Earth Jukebox turns frames from NASA scientific visualizations into learned visual patterns and then into style-shaped music. The showcase succeeds when a viewer can play the result and follow the connection between the source imagery, changes in the visual signal, and the music over time.

## Positioning

The music is generated from temporal features and learned patterns in NASA visualization frames, with musical style choices shaping the arrangement. It is an artistic sonification of imagery, not an original soundtrack, physical sound, weather forecast, or automated emotion reading.

Alongside it, eight data-faithful tracks (NASA SVS 5209, 5313, 3912, 5011, 4950, 5669, 5058, 5176) work the other way round: values are read back out of each visualization through its colour bar, every melody note is one value, and a pitch tracker checks that the finished mix gives the data back (R² 0.97–0.99, against about −0.3 to −1.3 for the same notes shuffled). Keep the two labels apart: "artistic sonification" for the hurricane season, "data-faithful" with its measured fidelity for the data tracks.

## Operating Context

The VAE and score generator run locally from command-line tools. Generated audio, MIDI, latent arrays, and diagnostic images are saved in `results/`. The web app plays the saved outputs without any server; uploading a new visualization needs the local generator running.

## Capabilities and Constraints

- The dataset supports video including WebM, GIF, and ordered image sequences.
- The VAE produces learned visual representations; their dimensions are not named physical quantities.
- Visual movement and optical-flow signals can influence musical activity, while key, chords, instruments, motifs, and style are artistic composition choices.
- The showcase explains existing generated files. It does not train the VAE or claim to infer emotion.
- The current source animation is Hurricane Ian (2022), credited in the project README to NASA/Goddard Space Flight Center Scientific Visualization Studio.

## Evidence on Hand

- Local source animation: `data/raw/hurr2022_v6_2160p30.webm`.
- Existing style outputs: WAV and MIDI files plus latent and cluster arrays under `results/`.
- Existing feature, piano-roll, and latent-memory visualizations under `results/`.

## Product Principles

- Let people hear the generated result while following its source visualization.
- Explain model signals and musical choices in plain language.
- Keep observations from the visualization distinct from artistic interpretation.
- Use existing local artifacts; do not require model training or a remote service to listen.

## Accessibility & Inclusion

The showcase is intended for non-technical viewers. Use plain-language labels, keyboard-accessible playback and seeking, readable contrast, and reduced-motion support.

## Brand commitments

The interface does not need to look NASA-branded. The user disliked the saturated Space Apps blue with neon yellow, and then a green/mint accent on a teal-black ground. Calm, listening-first, not a space cliche (no starfields, neon glow, glass).
