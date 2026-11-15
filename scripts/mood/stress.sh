#!/bin/bash
# Stress test of the mix safeguards on an existing small render (CPU, mix only).
# Usage: scripts/mood/stress.sh <id>
source /projects/e32706/omb8654/jukebox-wt/magenta/scripts/mood/env.sh
export JAX_PLATFORMS=cpu
id=$1
for db in -6 0; do
  python -m jukebox.learned.magenta_mood $id --mix-only --tag small-nonotes --mood-db $db --no-duck --label small-nonotes-noduck${db}
  python -m jukebox.learned.magenta_mood $id --mix-only --tag small --mood-db $db --no-duck --label small-noduck${db}
done 2>&1 | grep r2_with
