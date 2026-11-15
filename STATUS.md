# MuseControlLite B+ build: status (11:46, complete)

Branch `feat/mcl-bplus` (worktree `/projects/e32706/omb8654/jukebox-wt/mcl-bplus`), merged with three-voices-m1
(track 5046), NOT pushed. Deadline triage: ONE variant "B+" = bass kept, no cymbals / hi-hats, every Earth-voice
bed kept. The GPU queue was ~7 h deep; all GPU jobs were cancelled and the new takes rendered on CPU (`short`,
one job per 47 s window, 16 cores, 50 steps, ~4.5 min per window).

## Done and committed

- All 9 tracks have `demo/<id>_mclbplus/` (mix.mp3, control.mp3 except 5669, accompaniment.mp3, spectrogram.png,
  bplus_report.json, README.md) and a Compare entry "MuseControlLite B+ (no cymbals, with Earth voices)".
  5209, 5313, 3912: the pilot takes were reused with the cymbal post-process. 4950, 5176, 5011, 5058, 5669, 5046:
  new CPU renders with a cymbal-free prompt and negative prompt, plus the same post-process.
- `demo/learned/README.md`: B+ section and table (`python scripts/bplus_summary.py`).
- `scripts/export_next_app.py` was run, then on the login node (compute nodes have no npm): `npm run lint` exit 0,
  `npm test` 51/51, `npx next build --webpack` exit 0. Logs are in `logs/` (git-ignored).

## Not done / open

- The soft-rhythm / no-percussion pair was dropped (triage). The 5669 dynamics-only alternative was not rendered.
- The 3912 seam (pilot take, 32–36 s) is not fixed. The new crossfade method is unproven (see the learned README).
- Nothing has been listened to.

## Re-run one track

```bash
# new track: DESC in jukebox/learned/bplus.py, TRACKS in jukebox/learned/musecontrol.py, then (mcl env)
PYTHONPATH=. MCL_SRC=<dir containing <id>/> python -m jukebox.learned.bplus prepare <id>
python -m jukebox.learned.bplus units <id>      # one line per window
sbatch -J bpc-<id>-<which>-<k> scripts/bplus_render.sbatch <id> <variant> <which> <k>
sbatch -J bpp-<id> --dependency=afterany:<render ids> scripts/bplus_post.sbatch <id>
# per-track trims: BPLUS_GEN_OFFSET_DB (4950 used -10), BPLUS_VARIANT (5058 uses bplus_nomel via outputs/5058_mclbplus/variant.txt)
python scripts/export_next_app.py && cd eic-vae/eic-sonification/frontend && npm run lint && npm test && npx next build --webpack
```

Track 5046 (Arctic sea ice) build notes, merged from three-voices-m1: `git show 74846c6:STATUS.md`.
