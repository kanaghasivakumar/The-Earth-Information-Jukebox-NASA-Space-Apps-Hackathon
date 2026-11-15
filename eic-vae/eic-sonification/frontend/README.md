# Earth Jukebox frontend

The Next.js app for Earth Jukebox. See the [root README](../../README.md) for what it is and how to run everything.

```sh
npm install
npm run dev     # http://localhost:3000 (the Upload page also needs the API on :8000)
```

Routes live in `app/(site)/`: `/`, `/how-it-works`, `/upload`, `/accessibility`, `/compare`, `/credits`. Styling is in `app/jukebox.css`; design notes are in `DESIGN.md`.

The Listen page plays the hurricane season (an artistic sonification by the VAE) or one of eight data-faithful tracks, chosen with `?track=<SVS id>`. The data tracks live in `public/jukebox/tracks/<id>/` (contract: `web/README.md` at the repo root) and are refreshed with `python scripts/export_next_app.py` from the repo root. Their code is in `lib/data/` and `components/data/`. `/compare?track=<id>` switches between versions of a track at the same moment and keeps your 1–5 ratings in this browser until you export them.

```sh
npm run lint && npm test && npm run build
```

On a machine whose glibc is older than 2.29 (Quest), Turbopack has no native binding: build with `npx next build --webpack`.
