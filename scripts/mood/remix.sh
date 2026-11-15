#!/bin/bash
# Re-run mix + fidelity + checks on existing mood renders. Usage: scripts/mood/remix.sh <tag> <ids...>
source /projects/e32706/omb8654/jukebox-wt/magenta/scripts/mood/env.sh
export JAX_PLATFORMS=cpu
tag=$1; shift
python -m jukebox.learned.magenta_mood "$@" --mix-only --tag $tag 2>&1 | grep -v XNNPACK
