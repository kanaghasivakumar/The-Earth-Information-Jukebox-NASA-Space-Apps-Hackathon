#!/bin/bash
# Frame -> style weights (SigLIP, CPU). Usage: scripts/mood/weights.sh 5313 3912 5209
source /projects/e32706/omb8654/jukebox-wt/magenta/scripts/mood/env.sh
python -m jukebox.learned.mood "$@"
