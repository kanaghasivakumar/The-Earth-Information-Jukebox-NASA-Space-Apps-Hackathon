#!/bin/bash
# Build the 5046 data track on a compute node:  srun ... scripts/build_5046.sh [args]
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=. HF_HOME=/projects/e32706/omb8654/hf_cache OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PATH=/projects/e32706/omb8654/conda/envs/jukebox/bin:$PATH
exec python -W ignore -m jukebox.datasets.seaice_5046 "$@"
