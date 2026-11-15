#!/bin/bash
# CPU smoke test on a short node: prepare + check the melody control against upstream + 4-step generation
# of the first MAXSEC seconds with both adapters + evaluation.
#   srun -A p32576 -p short -t 04:00:00 --mem=48G -c 8 bash scripts/mcl_cpu_smoke.sh 5209
set -euo pipefail
cd /projects/e32706/omb8654/jukebox-wt/musecontrol
export PYTHONPATH=. HF_HOME=/projects/e32706/omb8654/hf_cache OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MCL_OUTROOT=${MCL_OUTROOT:-/projects/e32706/omb8654/jukebox-wt/musecontrol/outputs/_cpu_smoke} HF_HUB_OFFLINE=1
export HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/projects/e32706/omb8654/conda/envs/mcl/bin/python
id=${1:-5209}
[ -f $MCL_OUTROOT/${id}_mcl/controls_mcl.npz ] || $PY -W ignore -m jukebox.learned.musecontrol prepare $id
$PY -W ignore - <<PYEOF
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "/projects/e32706/omb8654/assets/mcl_hf_pkg")
from jukebox.learned import musecontrol as M
M._import_hf_package()
from mcl_hf.melody_processing import compute_melody
out = M.out_dir("$id")
a = M.fit_window(M.read_wav(out / "cond_real.wav"), 0.0)
sf.write("/projects/e32706/omb8654/jukebox-wt/musecontrol/logs_mcl/_cond_check.wav", a, 44100)
ref = compute_melody("/projects/e32706/omb8654/jukebox-wt/musecontrol/logs_mcl/_cond_check.wav")
mine = np.load(out / "controls_mcl.npz")["melody_real_0"]
n = min(ref.shape[1], mine.shape[1])
print("melody control vs upstream compute_melody: shapes", ref.shape, mine.shape, "top-1 agree", np.mean(ref[0,:n]==mine[0,:n]), "all agree", np.mean(ref[:,:n]==mine[:,:n]))
PYEOF
for ad in hf-melody wosdd-all; do
  /usr/bin/time -v $PY -W ignore -m jukebox.learned.musecontrol generate $id --adapter $ad --device cpu --steps ${STEPS:-4} \
     --guidance 1.5 --max-seconds ${MAXSEC:-12} 2>&1 | grep -v "^\s*\(Page\|Socket\|Signals\|Swaps\|File system\|Average\|Voluntary\|Involuntary\|Minor\|Major\|Exit\|Percent\|Command\)" 
done
$PY -W ignore -m jukebox.learned.musecontrol evaluate $id
