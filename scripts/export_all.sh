#!/bin/bash
# Export every arranged track (outputs/<id>/ from jukebox.render.arrange) into web/tracks/ in story
# order, and refresh demo/<id>/mix.mp3 (+ arrange_report.json) with the integrated mix.
#   srun -A p32576 -p short -t 00:20:00 --mem=8G -c 2 scripts/export_all.sh
set -e
cd "$(dirname "$0")/.."
BIN=/projects/e32706/omb8654/conda/envs/jukebox/bin
for id in 5209 5313 3912 5011 4950 5669 5058 5176; do
  PYTHONPATH=. $BIN/python -W ignore scripts/export_track.py outputs/$id --meta scripts/track_meta/$id.json "$@"
  mkdir -p demo/$id
  $BIN/ffmpeg -v error -y -i outputs/$id/mix.wav -codec:a libmp3lame -b:a 128k demo/$id/mix.mp3
  cp outputs/$id/arrange_report.json demo/$id/arrange_report.json
done
cp outputs/3912/timeline.json demo/3912/timeline_arranged.json
du -sh web/tracks
