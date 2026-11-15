#!/bin/bash
#SBATCH -A p32576
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH -o /projects/e32706/omb8654/jukebox-wt/mcl-bplus/logs_mcl/bplus_post_%x_%j.log
# B+ post-processing for one track (stitch if rendered here, decymbal, mix, fidelity, export to demo/<id>_mclbplus):
#   sbatch -J bpp-<id> [--dependency=afterany:<render jobs>] scripts/bplus_post.sbatch <id>
cd /projects/e32706/omb8654/jukebox-wt/mcl-bplus
export PYTHONPATH=. OMP_NUM_THREADS=4
PY=/projects/e32706/omb8654/conda/envs/jukebox/bin/python
echo "start $1 $(date)"
case $1 in 5209|5313|3912) ;; *)
  $PY -W ignore -c "import sys; from jukebox.learned import bplus as B; m=B.missing_units('$1'); print('missing', m); sys.exit(1 if m else 0)" || exit 1
  $PY -W ignore -m jukebox.learned.bplus stitch $1 ;;
esac
$PY -W ignore -m jukebox.learned.bplus mix $1 && $PY -W ignore -m jukebox.learned.bplus export $1
echo "done $1 $(date)"
