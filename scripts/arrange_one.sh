#!/bin/bash
# Arrange one track on a compute node:  srun ... scripts/arrange_one.sh 3912 [extra args]
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=. HF_HOME=/projects/e32706/omb8654/hf_cache OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
id=$1; shift
exec /projects/e32706/omb8654/conda/envs/jukebox/bin/python -W ignore -m jukebox.render.arrange outputs/$id "$@"
