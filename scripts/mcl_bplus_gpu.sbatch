#!/bin/bash
#SBATCH -A e32706
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -J bplus
#SBATCH -o /projects/e32706/omb8654/jukebox-wt/mcl-bplus/logs_mcl/bplus_gpu_%x_%j.log
# One track per job:  sbatch -J bplus-5209 scripts/mcl_bplus_gpu.sbatch 5209
# (falls back to -A p32576 if e32706 is pending). The steps live in scripts/bplus_one.sh.
exec bash /projects/e32706/omb8654/jukebox-wt/mcl-bplus/scripts/bplus_one.sh "$1" gpu
