#!/bin/bash
#SBATCH -A p32576
#SBATCH -p short
#SBATCH -t 00:45:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -J bplus-final
#SBATCH -o /projects/e32706/omb8654/jukebox-wt/mcl-bplus/logs/bplus_final_%j.log
# After every B+ post job: export to the Next app, then lint, vitest and the webpack build. Results in logs/.
cd /projects/e32706/omb8654/jukebox-wt/mcl-bplus
PY=/projects/e32706/omb8654/conda/envs/jukebox/bin/python
set -o pipefail
$PY scripts/export_next_app.py > logs/export_next_app.log 2>&1; echo "export_next_app exit $?" | tee logs/final_status.txt
cd eic-vae/eic-sonification/frontend
npm run lint > ../../../logs/lint.log 2>&1; echo "lint exit $?" | tee -a ../../../logs/final_status.txt
npm test > ../../../logs/test.log 2>&1; echo "test exit $?" | tee -a ../../../logs/final_status.txt
NEXT_TELEMETRY_DISABLED=1 npx next build --webpack > ../../../logs/build.log 2>&1; echo "build exit $?" | tee -a ../../../logs/final_status.txt
date >> ../../../logs/final_status.txt
