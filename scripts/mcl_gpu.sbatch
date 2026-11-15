#!/bin/bash
#SBATCH -A e32706
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -J mcl-pilot
#SBATCH -o /projects/e32706/omb8654/jukebox-wt/musecontrol/logs_mcl/gpu_%j.log
# MuseControlLite pilot renders on one GPU. Thin wrapper: the real steps live in scripts/mcl_render_all.sh,
# which is read when the job starts (so it can still be fixed while the job waits in the queue).
exec bash /projects/e32706/omb8654/jukebox-wt/musecontrol/scripts/mcl_render_all.sh "$@"
