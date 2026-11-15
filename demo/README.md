# Listening guide: the eight integrated tracks

Every `demo/<id>/mix.mp3` is the **integrated arrangement** made by `jukebox.render.arrange`. It has the data
melody and bands (FluidSynth, FluidR3_GM), model-written harmony (Anticipatory Music Transformer, medium) and the
three recorded voices driven by `controls.json`. The mix is normalised to about −16 LUFS. The same stems play in
the web jukebox (`web/tracks/<id>/`), in this story order:

| # | Track | What you hear | What it means | Key fidelity number |
|---|---|---|---|---|
| 1 | **5209** Zonal temperature 1880–2023 | A flute plays one note per year. Eight harp voices are strummed from the Arctic down to Antarctica. Strings, bass and wind sit underneath. | Higher pitch is a warmer year. By the end every zone has climbed, and the Arctic harp most of all. | Video vs GISS CSV r 0.9997. The pitch of the flute in the full mix gives back the global anomaly with R² 0.966 (shuffled: −0.66). |
| 2 | **5313** Night lights 2012–2023 | A flute plays the yearly light total, a harp arpeggio plays six longitude sectors, and clarinet runs follow the change map. A wood block ticks each year. City hum rises while birds and insects fade. | More light gives a higher note. Rising runs mean places got brighter and falling runs mean they got dimmer. The human voice grows as the living voice fades. | Years melody in the full mix: R² 0.977 (shuffled −0.92). |
| 3 | **3912** Sea surface currents and temperature | A flute melody plays over six soft pads, with strings and bass and a Pacific hydrophone. Cymbal swells mark section changes. | The flute is the global-mean SST, frame by frame. The ocean recording swells when the currents move more. The sections come from DINOv2 novelty in the video frames and are cues only. | R² 0.987 on the full mix (shuffled −1.11). |
| 4 | **5011** Lightning from the ISS 2017–2023 | Tom hits are panned to where the storms are, with a thunderclap behind each hit, rain and wind, and a flute line. There is no model harmony because the track is percussive. | Hit pan is longitude and drum pitch is latitude. The flute is the storm belt's mean latitude: it rises every northern summer. | Flute in the full mix: R² 0.983 (shuffled −1.03). Hits vs flash density r 0.83 (builder). |
| 5 | **4950** GEDI forest height | A flute with three soft band pads, strings and bass, an ocean bed, and a forest chorus. | Pitch is the mean canopy height of the forest facing you, and loudness is how much of the view is vegetated. The chorus swells over tall forest and the ocean grows over open sea. | Full mix R² 0.983 (shuffled −1.26). |
| 6 | **5669** GEOS aerosols (still image) | A west-to-east sweep with one instrument per aerosol type: double bass for sea salt, clarinet for dust, oboe for sulfate and harp for smoke. Wind and surf, city hum and birds sit underneath. There is no lead and no harmony. | Loudness is how much of a longitude column holds that aerosol, and pitch is its latitude. Listen for Saharan dust and African smoke. | On each type's own stem, 100 % of solo notes come back on the written pitch (dust R² 0.974, sulfate R² 0.79). R² for latitude is limited by the 9° pitch steps. |
| 7 | **5058** Florida lights after Hurricane Ian | Trumpet runs fall as the "after" picture wipes across the coast, over a pad, sparse strings and bass, traffic, frogs and insects. | Each run is one strip of the map, and a bigger fall means more light was lost (about 63 % of the light is lost overall). City hum fades and frogs take over. | Trumpet stem: R² 0.988 for the light value (the old synth stem scored −0.35). Full mix: R² 0.69, with 90 % of notes on the right pitch. This is the weakest mix, see below. |
| 8 | **5176** SST anomaly, 2024–2026 (daily) | A flute over six band pads, strings and bass, and ocean recordings. | Pitch is the global ocean's departure from normal (higher means warmer than normal). | Full mix R² 0.979 (shuffled −1.26). |

## How the fidelity numbers are measured

`arrange_report.json` in each demo folder holds the full numbers. pyin runs on the final rendered audio,
bounded to the lead's own sounding range ±1 semitone. Each lead note's onset window (`win`, after skipping
`offset` of the attack) gives a median pitch. That pitch goes back through the builder's PitchMap and is
scored against the note's data value (R²), and against the written pitch (share within ½ semitone). The
"before" number is the same measurement on the builder's original synth stem. The shuffled control is the
same arrangement with the lead's pitches permuted in time, scored against the real values.

| Track | Builder R² (own method) | Before: synth lead stem | After: final lead stem | After: full mix | Shuffled control |
|---|---|---|---|---|---|
| 5209 | 0.971 | 0.965 | 0.966 | 0.966 | −0.66 |
| 5313 (years) | 0.975 | 0.975 | 0.976 | 0.977 | −0.92 |
| 3912 | 0.986 | 0.985 | 0.987 | 0.987 | −1.11 |
| 5011 | 0.988 | 0.984 | 0.984 | 0.983 | −1.03 |
| 4950 | 0.982 | 0.981 | 0.983 | 0.983 | −1.26 |
| 5669 dust / sulfate / sea salt / smoke (pitch hit rate) | 0.97 / 0.89 / n=4 / −1.32 | 87 / 36 / 50 / 100 % | 100 / 100 / 100 / 100 % | not measured (4 voices at once) | none |
| 5058 | interval r 0.63 | −0.35 | 0.988 | 0.688 | −0.29 |
| 5176 | 0.978 | 0.979 | 0.979 | 0.979 | −1.26 |

Mix decisions that the numbers forced:

- The **flute** is the lead (GM 73). In the full mix it survives under strings and bass, while a vibraphone does not.
- **5058** uses a trumpet because its 0.12 s notes are shorter than the flute's attack (flute 60 % of notes, trumpet 100 %).
- **Harmony** is AMT strings plus a bass line. It stays under the lowest lead note, uses the lead's own key (`infer_key` would read a D-major-pentatonic lead as A major), and sits 13 dB down.
- **Bands** sit 15 dB under the melody's active RMS, and the voices 10–11 dB under (scaled by each curve's peak).
- A **sidechain** keeps the accompaniment 8 dB (5058: 10 dB) under the sounding lead and ducks it by 9 dB at most. One gain curve is used for all of these groups, so anthrophony against biophony keeps its balance. Event hits (data) are never ducked. Before the sidechain, the 4950 mix scored R² −0.10 and the 5058 mix −0.71.

## Caveats

- **Nothing here has been listened to.** The balance was judged from per-group RMS, the round trip and
  spectrograms (`outputs/<id>/arrange.png`), not by ear.
- **5058 full mix** is the weakest result. 2 of its 20 short notes are lost under the drone and city hum (R² 0.69).
  The trumpet stem alone recovers all 20.
- **5669 R²** values are small or negative even when every note is on pitch. Each pitch step covers about 9° of
  latitude, and the solo notes of a type often span only a few steps. Sea salt has only 4 solo notes.
- **5011** dates: flashes keep appearing for about 11.6 s after the movie time that maps to the page's end date
  (31 July 2023). Spoken dates there are held at that end date.
- **3912** absolute SST is uncertain by ±1–2 °C (see `demo/3912/README.md`). Its geophony curve is derived from
  frame-to-frame movement, not measured ocean sound.
- **5313/5058** night-light values are relative because the maps have no numeric scale.

## Removed demos

`demo/3912_harmony` and `demo/3912_form` were built on an older, biased 3912 timeline (unrecorded
`--min-chroma 15` inversion, about 2–3 °C warm). Both are folded into the integrated 3912 track. The harmony is
regenerated on the corrected v2 timeline. The form sections (`demo/3912/form.json`, DINOv2 frame novelty) do not
depend on the timeline and are reused as cymbal cues. `demo/3912/timeline_arranged.json` is the v2 data timeline
plus harmony, bass and form cues, and its lead notes are identical to `demo/3912/timeline.json`.

Reproduce (Quest `short` partition):

```bash
for id in 3912 5209 5176 4950 5313 5011 5669 5058; do
  srun -A p32576 -p short -t 01:00:00 --mem=16G -c 4 scripts/arrange_one.sh $id &
done; wait
srun -A p32576 -p short -t 00:20:00 --mem=8G -c 2 scripts/export_all.sh
python scripts/check_web.py
```
