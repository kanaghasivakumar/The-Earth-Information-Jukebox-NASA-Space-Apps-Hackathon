#!/bin/bash
#SBATCH -A p32576
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -J arr5046
#SBATCH -o logs_5046/%x-%j.out
# Keep the builder's lead and control stems as orig_* (the "before" numbers), then arrange.
set -e
cd /projects/e32706/omb8654/jukebox-wt/track-5046
for s in lead_extent control_shuffled; do
  [ -f outputs/5046/orig_$s.wav ] || cp outputs/5046/$s.wav outputs/5046/orig_$s.wav
done
scripts/arrange_one.sh 5046 "$@"
