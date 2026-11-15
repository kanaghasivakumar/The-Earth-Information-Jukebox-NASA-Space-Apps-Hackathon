#!/bin/bash
#SBATCH -A e32706
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -J mood-base
#SBATCH -o /projects/e32706/omb8654/jukebox-wt/magenta/outputs/logs/gpu_base_%j.log
# Magenta RT 2 "base" (2.4B) mood stems for the pilot tracks, then mix + fidelity (CPU part).
# Weights (SigLIP) are made on CPU beforehand; they are recomputed here only if missing.
module purge 2>/dev/null
unset LD_LIBRARY_PATH          # use the pip CUDA 12 libraries bundled with jax, not a site module
source /projects/e32706/omb8654/jukebox-wt/magenta/scripts/mood/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
python -c "import jax; print(jax.devices())"
for id in 5313 3912 5209; do
  [ -f outputs/${id}_mood/weights.json ] || JAX_PLATFORMS=cpu python -m jukebox.learned.mood $id
done
python -m jukebox.learned.magenta_mood 5313 3912 5209 --size mrt2_base
