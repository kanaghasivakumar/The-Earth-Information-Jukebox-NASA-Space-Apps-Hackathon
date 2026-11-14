#!/usr/bin/env bash
# Fetch extra small SVS movies for the form/constellation layer (~150 MB total).
#   srun -A p32576 -p short -t 00:40:00 --mem=16G -c 4 bash scripts/fetch_form_datasets.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/projects/e32706/omb8654/conda/envs/jukebox/bin/python}
SVS=https://svs.gsfc.nasa.gov/vis/a000000/a005000
[ -f data/raw/5046/frames.npy ] || $PY -m jukebox.data.ingest 5046 --fps 2 --width 256 --url $SVS/a005046/arctic_sea_ice_moving_two_year_w_dates.webm
[ -f data/raw/5313/frames.npy ] || $PY -m jukebox.data.ingest 5313 --fps 2 --width 256
[ -f data/raw/5097/frames.npy ] || $PY -m jukebox.data.ingest 5097 --fps 2 --width 256 --url $SVS/a005097/hurr2022_v6_2160p30.webm
