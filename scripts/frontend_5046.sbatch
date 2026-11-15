#!/bin/bash
#SBATCH -A p32576
#SBATCH -p short
#SBATCH -t 00:30:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -J fe5046
#SBATCH -o logs_5046/%x-%j.out
# NOTE: compute nodes have no npm (this job fails with exit 127); run these commands on the login node instead.
# Lint, test and build the Next app (node_modules symlinked from ../ui; Turbopack cannot run on Quest).
cd /projects/e32706/omb8654/jukebox-wt/track-5046/eic-vae/eic-sonification/frontend
export NEXT_TELEMETRY_DISABLED=1
echo "== lint"; npm run lint; echo "lint exit $?"
echo "== test"; npm test 2>&1 | tail -25; echo "test exit ${PIPESTATUS[0]}"
echo "== build"; npx next build --webpack 2>&1 | tail -40; echo "build exit ${PIPESTATUS[0]}"
