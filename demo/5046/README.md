# 5046 Daily Arctic Sea Ice, By Year: two winters

`mix.mp3` is the integrated arrangement (36.5 s). Built by `python -m jukebox.datasets.seaice_5046`
(`scripts/build_5046.sbatch`), then arranged by `scripts/arrange_5046.sbatch` (`jukebox.render.arrange`, config `_5046`).

**What you hear.** A flute plays the Arctic sea-ice extent every 5 days from 16 Apr 2021 to 15 Apr 2023. More ice plays
higher (F4 = 4.9, F6 = 15.9 million km², F major). It falls to a low each September and climbs back each winter, twice,
so the two years can be compared directly. A bowed-glass pad two octaves lower plays the NSIDC 1981–2010 median extent
for the same day of year, so the gap between the two lines is the ice that is missing. The Anticipatory Music Transformer
writes strings and bass under the flute. Ocean and wind (geophony) grow as water opens. Birds (biophony) stay low
and appear only in the summer open-water months, for seabirds feeding at the ice edge. There is no anthrophony.
Cymbal swells mark the A B C A B sections from `demo/5046_form` (cues only, times ×1.5).

**Time.** The movie is 730 frames (one day each) at 30 fps, which gives 24.3 s. The audio runs 1.5× slower (36.5 s, about 20 days a second).
`track.json` sets `video_rate: 0.6667`, and both players slow the video to match.

## How extent is measured

- **Pixels.** Each frame is decoded at 960 px. A pixel is water (blue), white (ice or snow) or land-coloured.
- **Land.** Land is any pixel that is land-coloured on at least 2 % of frames, which catches snow-free summer land. The
  Greenland ice sheet is added as land: it is the never-water, always-white component at Greenland's centre, split from
  the perennial pack by a 3 px erosion.
- **Ice.** Sea ice is a white pixel that is not land, within 0.88 of the globe radius. This cuts the hazy limb.
- **Area.** The globe is modelled as a pinhole camera at distance D. D = 1.81 Earth radii is fitted so that the
  Greenland ice sheet has its known area of 1.71 million km². No sea-ice data is used for this. An orthographic globe
  would make Greenland 3× too big.
- **Dates.** The burned-in labels show 16 Apr 2021 + i days. They were read by eye at 0.5, 5.0, 10.5 and 24.3 s, and the
  label changes on every frame. The SVS frames folder lists 2024304–2026303, which belongs to a newer render, so it is not used.

**Ground truth.** The movie extent was compared with the NSIDC Sea Ice Index v4 daily Arctic extent (G02135, public https,
no login) over all 730 days. It gives r = 0.998, MAE = 0.35 million km² and R² = 0.986. The movie reads about 0.6 low near the winter maxima and 0.6 high near the September minima (the 60 highest and lowest days), so the seasonal swing is about 10 % too small. Likely causes are the 3-day running minimum and the 15 % edge pixels at 960 px, plus seas near the limb (Baltic, Okhotsk) at the maxima.

## Fidelity (pyin on the rendered audio, `arrange_report.json`)

| | R² | notes within ½ semitone |
|---|---|---|
| builder synth lead stem (before) | 0.988 | 100 % |
| final flute lead stem | 0.990 | 100 % |
| **full mix** | **0.990** | 100 % |
| shuffled control (lead permuted in time) | −1.26 | — |

The builder's own round trip (one value per 5-day note) gives lead R² 0.992 and control R² −1.28. The climatology band
alone gives R² 0.994.

## Caveats

- The video is the hosted 2021–2023 **WebM**. The SVS **MP4** at the same address is now re-rendered daily (it showed
  2024–2026 on 15 Nov 2026), so it would not match this audio. Safari before version 17 may not play the WebM. If the
  video fails, the music still plays.
- Extent comes from the movie's colours, not from the AMSR2 files. The 3-day running minimum in the movie smooths day-to-day changes.
- Nothing has been listened to. The balance was judged from per-group RMS and `arrange.png`.
