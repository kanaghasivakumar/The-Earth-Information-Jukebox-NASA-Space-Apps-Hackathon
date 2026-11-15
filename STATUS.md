# Track 5046 (Daily Arctic Sea Ice, By Year): status

Worktree `/projects/e32706/omb8654/jukebox-wt/track-5046`, branch `feat/track-5046`. Do not push.
Logs are in `logs_5046/` (gitignored). Inputs are in `data/raw/5046/` (gitignored): the movie webm was copied from
`../form/data/raw/5046/`, and the NSIDC CSVs are fetched by the builder.

## Facts so far
- Movie: 730 frames at 30p (24.33 s). Burned-in dates 2021-04-16 .. 2023-04-15, one day per frame (checked by eye at 4 times).
  The SVS frames folder lists 2024304..2026303, a newer render, so it is NOT used for dates.
- Extent: classify ice/water/land pixels. Land = land-coloured on at least 2 % of frames, plus the Greenland ice-sheet component.
  Pixels are area-weighted with a perspective globe; the camera distance is fitted so Greenland's ice sheet = 1.71 M km².
- Ground truth: NSIDC G02135 v4 daily CSV (public https). The first run (orthographic, wrong disc) gave r = 0.9965 but the scale was off.
- Audio is 1.5x slower than the movie (36.5 s). `track.json` has `video_rate: 0.6667`, which the players must honour (TODO).

## Steps
1. [done] Builder (job 8838139): NSIDC r 0.998, MAE 0.35 M km2; builder lead R2 0.992, control -1.28. Rerun: `sbatch scripts/build_5046.sbatch`.
2. [done] Arrangement (job 8838203): lead stem R2 0.990, mix R2 0.990, control -1.26, -16.0 LUFS. `_5046` added. Job 8838203: `sbatch scripts/arrange_5046.sbatch` (copies orig_* stems, then runs arrange_one.sh 5046).
3. [done] Export (job 8838204), check_web OK. `sbatch scripts/export_5046.sbatch` runs export_track, demo copies, check_web and export_next_app (--integrate outputs). Still todo: add 5046 to `export_all.sh`. Old note: `python scripts/export_track.py outputs/5046 --meta scripts/track_meta/5046.json`; add 5046 to `export_all.sh`;
   copy `mix.mp3`/`arrange_report.json` to `demo/5046`; run `python scripts/check_web.py`; run `scripts/export_next_app.py`.
4. [done] `video_rate` support in web/js/app.js and DataPlayer.tsx, a 5046 entry in copy.ts, and 5046 in export_all.sh plus the demo/README listening guide.
5. [done 11:20, login node] lint exit 0; vitest 50/50 passed; `npx next build --webpack` exit 0 (SWC native load warnings only). Compute nodes have no npm, so job 8838237 failed with exit 127. Run these three on the login node: `cd eic-vae/eic-sonification/frontend && npm run lint && npm test && npx next build --webpack`.
   node_modules can be symlinked from `../ui/eic-vae/eic-sonification/frontend/node_modules`.

All steps are done and committed on feat/track-5046 (not pushed). outputs/5046 is a complete arranged dir for a later MuseControlLite B+ pass.
