#!/bin/bash
# Magenta RT mood stem + mix + fidelity. Usage: scripts/mood/render.sh <size> <ids...> [-- extra args]
#   scripts/mood/render.sh mrt2_small 5313 3912 5209
source /projects/e32706/omb8654/jukebox-wt/magenta/scripts/mood/env.sh
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
size=$1; shift
python -m jukebox.learned.magenta_mood "$@" --size "$size"
