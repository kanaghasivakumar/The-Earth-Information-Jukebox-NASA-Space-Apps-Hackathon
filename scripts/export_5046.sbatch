#!/bin/bash
#SBATCH -A p32576
#SBATCH -p short
#SBATCH -t 00:20:00
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH -J exp5046
#SBATCH -o logs_5046/%x-%j.out
# Export the arranged 5046 track to web/tracks, demo/5046 and the Next app.
set -e
cd /projects/e32706/omb8654/jukebox-wt/track-5046
BIN=/projects/e32706/omb8654/conda/envs/jukebox/bin
export PYTHONPATH=. PATH=$BIN:$PATH
python -W ignore scripts/export_track.py outputs/5046 --meta scripts/track_meta/5046.json
mkdir -p demo/5046
ffmpeg -v error -y -i outputs/5046/mix.wav -codec:a libmp3lame -b:a 128k demo/5046/mix.mp3
cp outputs/5046/arrange_report.json outputs/5046/track.json outputs/5046/timeline.json demo/5046/
cp outputs/5046/arrange.png demo/5046/arrange.png
python scripts/check_web.py
python scripts/export_next_app.py --integrate outputs
du -sh web/tracks/5046 eic-vae/eic-sonification/frontend/public/jukebox/tracks/5046
