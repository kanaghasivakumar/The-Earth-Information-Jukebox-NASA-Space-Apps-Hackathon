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
1. [running] Builder: `sbatch scripts/build_5046.sbatch` (job 8837978). Writes `outputs/5046/` and `demo/5046/`.
2. [todo] Arrangement `_5046` in `jukebox/render/arrangements.py`, then
   `sbatch -A p32576 -p short -t 01:00:00 --mem=16G -c 4 -o logs_5046/arr-%j.out --wrap "scripts/arrange_one.sh 5046"`
3. [todo] Export: `python scripts/export_track.py outputs/5046 --meta scripts/track_meta/5046.json`; add 5046 to `export_all.sh`;
   copy `mix.mp3`/`arrange_report.json` to `demo/5046`; run `python scripts/check_web.py`; run `scripts/export_next_app.py`.
4. [todo] Support `video_rate` in `web/js/app.js` and in the Next player. Add a line to `demo/README.md`.
5. [todo] In `eic-vae/eic-sonification/frontend`: `npm run lint`, `npm test`, `npx next build --webpack`.
   node_modules can be symlinked from `../ui/eic-vae/eic-sonification/frontend/node_modules`.
