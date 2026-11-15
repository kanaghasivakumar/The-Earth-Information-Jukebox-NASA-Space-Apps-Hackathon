#!/bin/bash
# Ablation on CPU (small model): no notes conditioning, and louder mixes (-6 dB under the melody).
# Usage: scripts/mood/ablation.sh <id>
source /projects/e32706/omb8654/jukebox-wt/magenta/scripts/mood/env.sh
export JAX_PLATFORMS=cpu
id=$1
python -m jukebox.learned.magenta_mood $id --size mrt2_small --notes none
python -m jukebox.learned.magenta_mood $id --mix-only --tag small-nonotes --mood-db -6 --label small-nonotes-loud
python -m jukebox.learned.magenta_mood $id --mix-only --tag small --mood-db -6 --label small-loud
