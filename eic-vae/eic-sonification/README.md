# Earth Jukebox web app

A synchronized player for the music generated in `../eic-vae`, plus an upload panel that turns your own Earth visualization into music.

- `frontend/` Next.js app with four routes under one shared layout (`app/(site)/`): `/` the player, `/how-it-works` the walkthrough, `/upload` your own visualization, `/accessibility` keyboard controls and screen reader support. An upload is held in the shared layout, so it survives moving between pages.
- `backend/main.py` entry point for the FastAPI app in `../eic-vae/server.py`, which wraps the eic-vae pipeline (VAE inference, latent pattern groups, the five music styles, MP3 and video export).

## Run

From the repo root, once: `uv sync`, then `cd eic-sonification/frontend && npm install`.

```sh
cd eic-sonification && npm run dev:all     # API on :8000, web on :3000
```

Or separately:

```sh
.venv/bin/python -m uvicorn main:app --app-dir eic-sonification/backend --port 8000
cd eic-sonification/frontend && npm run dev
```

## API

| Route | Purpose |
|---|---|
| `GET /api/health` | Status and available styles |
| `POST /api/jobs` | Multipart `file` (mp4, mov, webm, avi, gif; 300 MB, 10 min) with optional `styles` (comma separated), `tempo` (30-240), `key` (e.g. `C`, `F#`), `scale` (`major`, `minor`, `dorian`, `pentatonic`). Returns `{job_id}` |
| `GET /api/jobs/{id}` | Status, stage, progress; on completion the playable source (video, poster, tracks, signals) |
| `GET /api/media/{id}/web/...` | Generated audio and video |

Jobs run one at a time and are stored under `eic-vae/jobs/` (git-ignored). The VAE is never trained or modified.

## Refresh the bundled hurricane season

```sh
.venv/bin/python eic-vae/export_web.py        # audio, video, tracks.json, signals.json
.venv/bin/python eic-vae/export_pipeline.py   # data for the pipeline walkthrough
```
