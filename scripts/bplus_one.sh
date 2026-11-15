#!/bin/bash
# MuseControlLite B+ for one track (drum-free accompaniment under the data lead, bands and Earth voices).
# Driven by the arranged track in $MCL_SRC/<id> (default ../integrate/outputs/<id>); writes outputs/<id>_mclbplus.
#
#   bash scripts/bplus_one.sh <id> submit   # sbatch the GPU stage (one job per track; prepare runs inside)
#   bash scripts/bplus_one.sh <id> gpu      # prepare + generate every variant (mcl env, needs a GPU)
#   bash scripts/bplus_one.sh <id> cpu      # evaluate, mix, export to demo/<id>_mclbplus (jukebox env, CPU)
#
# New track: add its prompt wording to jukebox/learned/bplus.py DESC and its fidelity window / rhythm plan
# to jukebox/learned/musecontrol.py TRACKS, then run `submit`, then `cpu` (srun -A p32576 -p short).
set -euo pipefail
ID=$1; STAGE=${2:-cpu}
ROOT=/projects/e32706/omb8654/jukebox-wt/mcl-bplus
cd $ROOT
export PYTHONPATH=. HF_HOME=/projects/e32706/omb8654/hf_cache OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1
mkdir -p logs_mcl
case $STAGE in
  submit)
    sbatch -A ${ACCOUNT:-e32706} -J bplus-$ID scripts/mcl_bplus_gpu.sbatch $ID ;;
  gpu)
    PY=/projects/e32706/omb8654/conda/envs/mcl/bin/python
    # duplicate submissions (e32706 and p32576 fallback): the first job to start takes the lock, the rest exit
    mkdir -p outputs/${ID}_mclbplus
    mkdir outputs/${ID}_mclbplus/.gpu_lock${SEED:+_$SEED}${LOCK_TAG:-} 2>/dev/null || { echo "$ID: another job has the lock"; exit 0; }
    echo "start gpu $ID $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader || true
    [ -n "${SKIP_PREPARE:-}" ] || $PY -W ignore -m jukebox.learned.bplus prepare $ID
    $PY -W ignore -m jukebox.learned.bplus generate $ID ${VARIANTS:+--variants $VARIANTS} ${SEED:+--seed $SEED}
    echo "done gpu $ID $(date)" ;;
  cpu)
    PY=/projects/e32706/omb8654/conda/envs/jukebox/bin/python
    for c in evaluate mix export; do $PY -W ignore -m jukebox.learned.bplus $c $ID; done ;;
esac
