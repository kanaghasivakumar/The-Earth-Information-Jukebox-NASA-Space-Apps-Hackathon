#!/bin/bash
#SBATCH -A p32576
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -J build5046
#SBATCH -o logs_5046/%x-%j.out
# sbatch scripts/build_5046.sbatch   (run from the worktree root)
cd /projects/e32706/omb8654/jukebox-wt/track-5046
scripts/build_5046.sh "$@"
