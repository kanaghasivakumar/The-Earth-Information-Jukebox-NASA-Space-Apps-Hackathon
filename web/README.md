# Earth Information Jukebox: web player

A static site that plays NASA SVS Earth visualizations as music. Every lead note is a data value.
The site is plain HTML, CSS and JavaScript ES modules, with no build step. Video streams from
svs.gsfc.nasa.gov and is never copied into the repo.

## Run locally

```bash
cd web
python -m http.server 8000
# open http://localhost:8000/  (deep link to a track: http://localhost:8000/#3912)
```

The page must be served over HTTP. Opening `index.html` from disk blocks `fetch()`. For GitHub
Pages, publish the `web/` folder; `.nojekyll` is already there.

## Using it

Every feature has an on-screen button, and most also have a key. Press `?` in the page for help.

| Key | Action |
|---|---|
| Space | play / pause |
| ← / → | scrub 5 s back or forward |
| Shift+← / → | step to the previous or next lead note, hear it, and hear its value |
| Home | back to the start |
| L | spoken legend, then calibration tones at the lowest, middle and highest value (lead, then bands) |
| V | say the current time, value, trend (and the voice balance, if real controls exist) |
| S | sweep: every latitude band at this moment, north to south, panned left to right |
| Shift+S | overview: the whole lead in 8 s, panned left to right over time |
| P | flip polarity (higher value plays lower). The lead is then re-synthesized live from `timeline.json` |
| A | A/B: real data or the shuffled control |
| H | headphone 3D sound (HRTF `PannerNode`) |
| 1–7 | mute a group: melody, bands, events, harmony, geophony, anthrophony, biophony |
| M | mute everything |
| Esc | stop speech, sweep or overview |

### Accessibility

- **Screen readers.** A polite ARIA live region announces the time (or the date, when `time_axis` is set), the value and the units while the track plays. You can set it to every 5 s, every 10 s, each note (at most every 1.5 s) or off. Toggle confirmations go to a separate `role=status` region. A "browser voice" option also speaks the announcements with `speechSynthesis`.
- **Shortcuts and screen-reader modes.** Shortcuts are ignored inside form fields. Space and Enter still activate the focused button. Range sliders keep their native arrow keys. Screen-reader users in browse mode should switch to focus mode, or use the buttons.
- **Spoken legend.** The legend uses `speechSynthesis` so the words and the WebAudio calibration tones play in order. Without speech support it falls back to the live region.
- **Text alternatives.** The canvas piano roll is `aria-hidden`. It has a written summary (start, end, min, max, trend) and a full data table of the lead notes.
- **Fidelity.** The fidelity panel states R² in plain language ("the melody follows the data closely; shuffled data does not").
- **Layout and display.** Works at phone width. All targets are at least 44 px. Focus outlines are visible. The page respects `prefers-color-scheme`, `prefers-contrast: more`, `forced-colors` and `prefers-reduced-motion` (the video starts paused, and the checkbox "Play the video" is off).
- **No Web Audio.** If Web Audio is missing or decoding fails, a native `<audio controls>` element plays `mix.mp3`.

### Headphone 3D

Each stem group goes through its own HRTF `PannerNode`. The group starts at a home direction:

- melody: in front
- geophony: left (about 55°)
- anthrophony: right (about 55°)
- biophony: above
- bands: slightly above

The group then moves up to ±75° with the average `pan` of its sounding notes (−1 is west/left, +1 is east/right). Sweep and overview tones are placed with HRTF too while 3D is on.

## Track contract

`tracks/index.json`:

```json
{"tracks": [{"id": "3912", "title": "…", "voice": "geophony", "summary": "…", "duration_s": 66.5}]}
```

A plain list of ids or objects also works.

`tracks/<id>/track.json` has these required fields:

| Field | Meaning |
|---|---|
| `id`, `title`, `svs_url`, `video_url`, `duration_s`, `units`, `description` | basic metadata |
| `legend` | `[{voice, meaning}]` |
| `stems` | `[{name, file, group, label?}]`, where `group` is one of melody, bands, events, harmony, geophony, anthrophony, biophony |
| `timeline` | `"timeline.json"`: `[{start, end, pitch, velocity, voice, value, pan}]` (`value` is in `units`) |
| `controls` | `"controls.json"`: `{t: [...], curves: {geophony, anthrophony, biophony, brightness}}`, each 0..1. Add `"synthetic": true` if it is a placeholder |
| `fidelity` | `{r2, pearson_r, mae, control_r2}` |
| `data_source` | string |

Optional fields that the player uses when they are present:

| Field | Used for |
|---|---|
| `control: {file, replaces: "melody", label}` | the A/B shuffled control (key A) |
| `mix` | a full-mix MP3, for the download link and the no-Web-Audio fallback |
| `pitch_map: {vmin, vmax, low, span, root, scale, polarity}` | the lead mapping: calibration tones and polarity flip |
| `band_pitch_map` and `sweep: {label, voices: [{voice, label, pan, pitch_offset}]}` | sweep mode (S). Without them, sweep falls back to all `band*` voices |
| `lead_voice` (default `"lead"`) and `voice_groups: {voice: group}` | assigning timeline voices to groups. Default: `lead`→melody, `band*`→bands, otherwise a name prefix |
| `time_axis: {start, end, format: "year" \| "month" \| "day"}` | spoken dates instead of movie time (ISO dates, linear over the track) |
| `video_alt_url` | a second `<source>`, such as WebM |

All stems play from t = 0 in sample sync. Keep them the same length and render them with the same encoder.

## Adding a track

1. Run the pipeline. It writes `outputs/<id>/` with `report.json`, `timeline.json`, `lead.wav`, `mix.wav` and `control_shuffled.wav`. Extra stems named `<group>*.wav` (for example `geophony_hydrophone.wav`) are picked up automatically.
2. Optionally, write `scripts/track_meta/<id>.json` to override or extend `track.json`: description, legend, `time_axis`, `sweep` and so on.
3. Export, using the jukebox conda env (it has ffmpeg):

   ```bash
   python scripts/export_track.py outputs/<id> --meta scripts/track_meta/<id>.json
   ```

   The script writes `web/tracks/<id>/` and adds the track to `web/tracks/index.json`.

   - WAV stems are encoded to MP3, and MP3 stems are copied.
   - If there is no bands stem, one is re-rendered from the timeline at the level it has in the mix, so lead and bands can be muted separately.
   - If `controls.json` is missing, a synthetic one is written and marked `"synthetic": true`.
   - The video is linked by URL from the SVS catalog (the smallest MP4/M4V at least 600 px wide), never copied.

The 3912 demo was exported with:

```bash
python scripts/export_track.py demo/3912 --meta scripts/track_meta/3912.json
```

## Files

- `index.html`: page structure, help dialog and live regions
- `css/style.css`: theme tokens, layout, focus, contrast and motion preferences
- `js/app.js`: track loading, UI, keyboard, announcements, legend, sweep, overview, A/B, polarity
- `js/audio.js`: the Web Audio engine (stems, group gains, HRTF, live synth, look-ahead scheduler)
- `js/speech.js`: the live-region announcer and the `speechSynthesis` helpers
- `js/roll.js`: the decorative canvas piano roll
- `js/music.js`: the pitch map (mirrors `jukebox/score/mapping.py`) and number/time formatting

## Known limits

- iOS mutes Web Audio when the ring/silent switch is on.
- Polarity flip and the live lead use a simple oscillator timbre, not the recorded one.
- The 3912 `controls.json` is a placeholder: the geophony level and brightness follow the lead.
