# Earth Jukebox

Earth Jukebox turns a NASA hurricane-season animation into music. A variational autoencoder learns compact patterns from the animation's frames, and a composer turns how those patterns change into five styles of music: ambient, cinematic, classical, jazz and electronic. The web app plays the music in step with the footage, shows how a picture becomes music, and lets you upload your own Earth visualization.

Built for the NASA Space Apps Challenge. This is an independent project and is not endorsed by NASA. Animation credit: NASA/Goddard Space Flight Center Scientific Visualization Studio, [Hurricane Ian (2022), SVS 5097](https://svs.gsfc.nasa.gov/5097/).

It is an artistic sonification of imagery, not a forecast or a measurement. The model's eight numbers are learned patterns, not temperature, rainfall or wind speed.

## What you need

- [`uv`](https://docs.astral.sh/uv/) (installs Python 3.14 automatically)
- Node.js 20 or newer and npm
- `ffmpeg` and `ffprobe` on your PATH (used for uploads and for exporting audio and video)

## Quickstart

```sh
uv sync
cd eic-vae/eic-sonification/frontend && npm install && cd ..
npm run dev:all        # API on http://localhost:8000, web app on http://localhost:3000
```

Open http://localhost:3000. The bundled hurricane season plays without the API; the API is only needed for the Upload page.

## What is where

| Folder | Contents |
|---|---|
| `eic-vae/` | The pipeline (the repo's main folder): frame loading, the trained VAE, the composer and five styles, the FastAPI server (`server.py`), export scripts and tests |
| `eic-vae/eic-sonification/frontend/` | The Next.js app: `/` player, `/how-it-works`, `/upload`, `/accessibility` |
| `eic-vae/eic-sonification/backend/` | Entry point that serves `eic-vae/server.py` |
| `plans/` | Improvement plans for the codebase, with a status table |

## Get the data

The demo plays from the media already in `eic-vae/eic-sonification/frontend/public/jukebox`, so you need nothing else to run it.

To regenerate the music or the exports you need the source video, which is too large for git. Download the hurricane animation from the [SVS 5097 page](https://svs.gsfc.nasa.gov/5097/) into `eic-vae/data/raw/` (that folder is git-ignored). Then follow `eic-vae/MUSIC_GENERATION.md` to generate music, and refresh the web exports from the repo root:

```sh
uv run python eic-vae/export_web.py        # audio, video, tracks.json, signals.json
uv run python eic-vae/export_pipeline.py   # data for the How it works walkthrough
```

## Tests

```sh
cd eic-vae/eic-sonification && npm run verify      # type check, lint and the Python tests
```

The Python tests alone: `uv run pytest -q`.

## Optional: the AI music director

Setting `GEMINI_API_KEY` in your shell lets `generate_music.py` ask an AI model to choose a style from a text description. Everything else works without it. See `eic-vae/MUSIC_GENERATION.md`.
