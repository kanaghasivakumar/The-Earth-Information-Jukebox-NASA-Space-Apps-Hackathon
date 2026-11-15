"""Earth Jukebox backend.

A thin entry point for the FastAPI app that wraps the eic-vae music pipeline
(eic-vae/server.py). Run from the repo root:

    .venv/bin/python -m uvicorn main:app --app-dir eic-vae/eic-sonification/backend --port 8000
"""
import sys
from pathlib import Path

VAE = Path(__file__).resolve().parents[2]  # eic-vae/, the folder that holds the pipeline and server.py
sys.path.insert(0, str(VAE))

from server import app  # noqa: E402,F401  (re-exported for uvicorn)
