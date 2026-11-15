#!/bin/bash
#SBATCH -A p32576
#SBATCH -p short
#SBATCH -t 01:30:00
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH -o /projects/e32706/omb8654/jukebox-wt/mcl-bplus/logs_mcl/bplus_render_%x_%j.log
# One B+ render unit on CPU:  sbatch -J bpc-<id>-<v>-<which>-<k> scripts/bplus_render.sbatch <id> <v> <which> <k>
cd /projects/e32706/omb8654/jukebox-wt/mcl-bplus
export PYTHONPATH=. HF_HOME=/projects/e32706/omb8654/hf_cache OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
echo "start $(date)"
/projects/e32706/omb8654/conda/envs/mcl/bin/python -W ignore -m jukebox.learned.bplus render $1 --unit $2 $3 $4 --device cpu
echo "done $(date)"
