# MuseControlLite B+ build: status

Branch `feat/mcl-bplus` (worktree `/projects/e32706/omb8654/jukebox-wt/mcl-bplus`), not pushed.
Deadline triage (coordinator, 11:14): ONE variant "B+" = bass kept, no cymbals / hi-hats, all Earth-voice beds
kept. GPU queue was ~7 h deep, so every GPU job was cancelled; renders run on CPU (`short`).

## Per track

| track | MCL take | render jobs (CPU, one per 47 s window) | post job (stitch, decymbal, mix, fidelity, export) |
|---|---|---|---|
| 5209 | pilot take reused (not regenerated) | – | 8838490 |
| 5313 | pilot take reused | – | 8838491 |
| 3912 | pilot take reused (seam at 32–36 s unchanged) | – | 8838492 |
| 4950 | new render, 2 windows | 8838336–8838339 | 8838494 (afterany renders) |
| 5176 | new render, 1 window | 8838340–8838341 | 8838495 |
| 5011 | new render, 3 windows | 8838342–8838347 | 8838496 |
| 5058 | new render, coarse melody | 8838351–8838352 | 8838497 |
| 5669 | new render, dominant-voice melody, real only | 8838353 | 8838498 |
| 5046 | not started (data track lands from another worktree) | – | – |

Final check job 8838549 (afterany on all post jobs): `scripts/export_next_app.py`, `npm run lint`, `npm test`,
`npx next build --webpack`; exit codes in `logs/final_status.txt`, full logs in `logs/`.
Job list files: `logs_mcl/render_jobs.txt`, `logs_mcl/post_jobs.txt`, `logs_mcl/final_job.txt`.

## Resume (if interrupted)

```bash
cd /projects/e32706/omb8654/jukebox-wt/mcl-bplus
squeue -u $USER                                         # what is still running
cat logs/final_status.txt                               # final checks, once job 8838549 ran
ls demo/*_mclbplus/mix.mp3                              # tracks that finished
# a track whose post job failed: re-run it (it stitches, mixes and exports)
sbatch -J bpp-<id> scripts/bplus_post.sbatch <id>
# a missing render window: python -m jukebox.learned.bplus units <id>  (mcl env), then
sbatch -J bpc-<id>-<which>-<k> scripts/bplus_render.sbatch <id> <variant> <which> <k>
# then: sbatch scripts/bplus_final.sbatch ; commit demo/*_mclbplus, the frontend public/ + lib/data changes
```
A new track (e.g. 5046): add its prompt wording to `DESC` in `jukebox/learned/bplus.py` and its window/rhythm to
`TRACKS` in `jukebox/learned/musecontrol.py`, run `prepare` (mcl env), submit one render per unit, then the post job.

## Committed

- (see `git log`) pipeline code, scripts, frontend export and test changes.

## Remaining

- commit demo/<id>_mclbplus per track as post jobs finish; update `demo/learned/README.md` B+ section
- run/verify the final checks, commit the exported frontend data

Track 5046 (Arctic sea ice) build notes, merged from three-voices-m1: `git show 74846c6:STATUS.md`.
