#!/bin/bash
# Render + evaluate the MuseControlLite pilots (5209, 5313, 3912). GPU: called by scripts/mcl_gpu.sbatch.
# Full-condition variant (lead + arranged harmony/bands in the melody control): COND=full GUIDANCE=1.5
# CPU smoke test:  DEVICE=cpu STEPS=4 MAXSEC=12 IDS=5209 GUIDANCE=1.5 bash scripts/mcl_render_all.sh
set -euo pipefail
cd /projects/e32706/omb8654/jukebox-wt/musecontrol
export PYTHONPATH=. HF_HOME=/projects/e32706/omb8654/hf_cache OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/projects/e32706/omb8654/conda/envs/mcl/bin/python
IDS=${IDS:-"5209 5313 3912"}
DEVICE=${DEVICE:-cuda}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-"1.0 1.5 2.5"}
ADAPTERS=${ADAPTERS:-"hf-melody wosdd-all"}
COND=${COND:-lead}
SEEDS=${SEEDS:-42}
export HF_HUB_OFFLINE=1
EXTRA=""
[ -n "${MAXSEC:-}" ] && EXTRA="--max-seconds $MAXSEC"
[ "$DEVICE" = cuda ] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "start $(date)"
for id in $IDS; do [ -f outputs/${id}_mcl/controls_mcl.npz ] && [ -z "${REPREPARE:-}" ] || $PY -W ignore -m jukebox.learned.musecontrol prepare $id; done
for ad in $ADAPTERS; do
  if [ "$ad" = wosdd-all ] && [ ! -f /projects/e32706/omb8654/assets/MuseControlLite/checkpoints/woSDD-all/model_3.safetensors ]; then
    echo "skip wosdd-all: no checkpoints"; continue; fi
  for seed in $SEEDS; do
    $PY -W ignore -m jukebox.learned.musecontrol generate $IDS --adapter $ad --device $DEVICE --steps $STEPS \
        --guidance $GUIDANCE --cond $COND --seed $seed $EXTRA
  done
done
for id in $IDS; do $PY -W ignore -m jukebox.learned.musecontrol evaluate $id > logs_mcl/eval_$id.log 2>&1 & done; wait
grep -h " A: " logs_mcl/eval_*.log || true
echo "done $(date)"
