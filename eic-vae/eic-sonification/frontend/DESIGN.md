---
name: Earth Jukebox
description: A synchronized hurricane-season player styled as a chart recorder, with an ink-blue ground, one vermilion pen colour and four lettered pattern-group hues.
colors:
  floor: "#0c1116"
  deep: "#0b1520"
  ground-2: "#12202f"
  footer-floor: "#080b0e"
  ink: "#e8eef2"
  ink-2: "rgb(232 238 242 / 0.8)"
  ink-3: "rgb(232 238 242 / 0.64)"
  rule: "rgb(200 220 235 / 0.16)"
  panel: "rgb(200 220 235 / 0.05)"
  well: "rgb(2 6 10 / 0.5)"
  pen-vermilion: "#ff6b4a"
  pen-vermilion-hi: "#ff9275"
  on-pen: "#1a0905"
  group-a: "#7cc4ff"
  group-b: "#ffb454"
  group-c: "#b79cff"
  group-d: "#7fd1ae"
typography:
  display:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "clamp(2.2rem, 3.8vw, 3.5rem)"
    fontWeight: 800
    lineHeight: 0.98
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "clamp(1.9rem, 3.4vw, 2.75rem)"
    fontWeight: 800
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  title:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "1.2rem"
    fontWeight: 800
    lineHeight: 1.2
  body:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 800
    letterSpacing: "0.02em"
  numeral:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "0.78rem"
    fontWeight: 400
    fontFeature: "tnum"
  clock:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "1.6rem"
    fontWeight: 500
    letterSpacing: "-0.02em"
    fontFeature: "tnum"
rounded:
  xs: "2px"
  sm: "4px"
  md: "6px"
  lg: "8px"
spacing:
  xs: "4px"
  sm: "10px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  gutter: "56px"
components:
  button-play:
    backgroundColor: "{colors.pen-vermilion}"
    textColor: "{colors.on-pen}"
    rounded: "{rounded.md}"
    padding: "0 28px"
    height: "56px"
    typography: "{typography.label}"
  button-play-hover:
    backgroundColor: "{colors.pen-vermilion-hi}"
  button-skip:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "0 14px"
    height: "56px"
  button-ghost:
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "0 18px"
    height: "44px"
  style-row:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "10px 14px 10px 12px"
    height: "62px"
  style-row-selected:
    backgroundColor: "rgb(200 220 235 / 0.12)"
    textColor: "{colors.ink}"
  style-icon-tile:
    backgroundColor: "rgb(255 255 255 / 0.12)"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    size: "44px"
  style-icon-tile-selected:
    backgroundColor: "{colors.pen-vermilion}"
    textColor: "{colors.on-pen}"
  panel:
    backgroundColor: "{colors.panel}"
    rounded: "{rounded.lg}"
    padding: "20px 22px"
  group-chip:
    textColor: "{colors.deep}"
    rounded: "{rounded.sm}"
    size: "32px"
  chip-button:
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "0 14px"
    height: "44px"
  playhead-tag:
    backgroundColor: "{colors.pen-vermilion}"
    textColor: "{colors.deep}"
    rounded: "{rounded.xs}"
    padding: "1px 7px"
    typography: "{typography.numeral}"
---

# Design System: Earth Jukebox

Scope: the home route only (Listen, Upload, How it works, Accessibility, plus Compare and Credits). The `/lab` route keeps its own system and is not governed here. All rules are scoped under the `.jb` root in `app/jukebox.css`. Token names above map to `--ink`, `--ink-2`, `--ink-3`, `--rule`, `--deep`, `--ground-2`, `--panel`, `--well`, `--accent`, `--accent-hi`, `--on-accent`, `--r0` to `--r3`.

## Overview

**Creative North Star: "The Chart Recorder"**

One synchronized instrument: footage, score and one shared timeline move together like a strip chart being drawn. The ground is ink-blue, near black at the page edge and a little lifted at the top. A single vermilion is the pen: it draws the played part of the loudness curve, the playhead and the Play button, and nothing else carries that colour. Four cool-to-warm pattern-group hues colour the data, always with a letter.

The recorder's ruled chart paper appears in exactly one place, behind the timeline lanes. Everywhere else the surface is quiet: translucent blue-grey panels, hairline rules, and large plain type. No glass, no starfield, no glow.

Density is moderate and predictable: large hit targets (56px transport, 44px minimum elsewhere), copy capped near 52-72ch, a two-column stage (480px footage column, fluid control column) that collapses to a single ordered column under 980px. Accessibility is the stated purpose: 3px vermilion focus rings, a letter on every group colour, reduced-motion support.

**Key Characteristics:**
- Ink-blue gradient ground (#12202f to #0b1520 over 760px), pale blue-grey ink, one vermilion pen.
- Panels 8px, buttons and rows 6px, chips 4px; soft tonal panels, hairline rules.
- Figtree for everything spoken, JetBrains Mono for every numeral.
- Four group hues, each paired with A-D; chart-paper grid only under timeline lanes.

## Colors

An ink-blue ground, pale ink, one vermilion for action and the pen, four lettered pastel group hues reserved for data.

### Primary
- **Pen Vermilion** (`{colors.pen-vermilion}`): Play button, playhead and its time tag, played portion of the loudness curve, focus rings, progress fill, text selection, selected style icon tile, step numeral chips, pipeline frame highlight, data-flow dashes, drag-over drop zone, checkbox accent, nav-hover underline.
- **Pen Vermilion Hi** (`{colors.pen-vermilion-hi}`): hover of the Play button, link colour and hover.
- **On Pen** (`{colors.on-pen}`): text on vermilion fills.

### Tertiary (pattern groups)
- **Group A Sky** (`{colors.group-a}`), **Group B Amber** (`{colors.group-b}`), **Group C Lilac** (`{colors.group-c}`), **Group D Sage** (`{colors.group-d}`): timeline blocks, group chips, the sound-only letter, pipeline map and piano roll. Also: group A's sky blue tints the chart-paper grid, the loudness stroke and fill.

### Neutral
- **Floor** (`{colors.floor}`): body and top bar (at 92%); **Footer Floor** for the footer band; **Deep** and **Ground 2** form the page gradient and the dark ink used on bright fills.
- **Ink**, **Ink 2** (80%), **Ink 3** (64%): primary, reading and secondary text.
- **Rule** (blue-grey at 16%): every divider and panel border. **Panel** (blue-grey at 5%) and **Well** (near-black at 50%) are the two surface tints.

### Named Rules
**The One Pen Rule.** Vermilion marks what plays, what has been played, what has focus and what progresses. Selection of a choice is not action: it is a quiet tint.
**The Letter Beside Color Rule.** A group hue never appears without its letter (A-D); colour is never the only cue.

## Typography

**Display Font:** Figtree (system-ui fallback)
**Body Font:** Figtree (400, 500, 600, 700, 800)
**Label/Mono Font:** JetBrains Mono, numerals only

**Character:** A friendly geometric sans at heavy weight for statements and regular for reading, with a mono that keeps every number from jittering.

### Hierarchy
- **Display** (800, clamp(2.2rem, 3.8vw, 3.5rem), 0.98, -0.03em): the Listen heading.
- **Headline** (800, clamp(1.9rem, 3.4vw, 2.75rem)): How it works, Accessibility, Upload.
- **Title** (700-800, 1.1-1.35rem): style names, step titles, pipeline column heads.
- **Body** (400, 1rem/1.55; lede 1.125rem): Ink 2, capped 52-72ch.
- **Label** (800, 1.125rem, +0.02em, uppercase): the Play button.
- **Clock** (mono 500, 1.6rem): elapsed time leading the transport row.
- **Numeral** (mono 400-500, 0.7-0.8rem, tabular): playhead tag, axis, style duration, key hints, pipeline bar values.

### Named Rules
**The Mono Is Numbers Rule.** JetBrains Mono carries timecodes, durations, step numerals and key hints; never prose.

## Layout

Centered column, max-width 1240px, 24px side gutter (16px under 720px). The stage is a grid: 480px footage column plus fluid control column, 56px column gap and 32px row gap from 980px; below that the pieces stack in a set order (intro, transport, video, styles, timeline, on-screen, now). Play, clock and the style list sit within the first screen. The timeline spans both columns; lanes are a 220px label column plus plotting column from 720px. Bands (Listen, Upload, How it works, Accessibility) are separated by a hairline top border with 56-72px vertical padding. The pipeline walkthrough is a five-part grid from 900px (frames 150px, arrow 44px, bars, arrow 44px, map and roll) and a single column below. Top bar min 72px, nav wraps below 720px.

## Elevation & Depth

Mostly tonal. Panels are translucent blue-grey (Panel) with a hairline Rule border; lanes and notes sit in Wells. Soft, wide, low-opacity drop shadows exist only on the footage frame, the timeline panel and the selected style row. The playhead breaks out of the lane stack by 6px instead of casting anything. Chart-paper ruling (50px major, 10px by 13px minor lines in group-A blue at 10% and 4%) exists only inside timeline lane bodies, not on the page ground.

### Named Rules
**The Paper Under The Lanes Rule.** Ruled grid belongs to the timeline lanes and nowhere else; the group-blocks lane stays clear of it.
**The Hairline Before Shadow Rule.** Separate surfaces with a Rule border or a Well first; a shadow is allowed only on the three surfaces named above.

## Shapes

Three radii by role: 8px for panels, the footage frame, the timeline panel and the pipeline; 6px for buttons, style rows, icon tiles, chip buttons and pipeline thumbnails; 4px for group chips, lanes, checks, step numerals and the error note. kbd uses 3px, the playhead tag and focus ring 2px. Group blocks in the lane are sharp rectangles. The drop zone is a 2px dashed border at 40% white that turns solid when a file is held and vermilion on drag-over.

## Components

### Buttons
- **Shape:** 6px corners.
- **Play:** vermilion fill, On Pen text, 56px tall, 148px min width, 800 uppercase. Hover Pen Vermilion Hi, active 1px press, disabled blue-grey at 14% with Ink 2 text. Transitions 0.15s ease-out.
- **Skip:** transparent, 2px Rule border, 56px minimum; hover brightens border to Ink with an 8% white wash.
- **Ghost (file choose):** 2px Ink border, 44px tall, hover 12% white wash.

### Style list
One column of separate rows with 8px gap. Each row is a 6px bordered tile, 62px minimum, with a 44px icon tile at left, name and duration over description and key hint. Selected is a quiet tint (blue-grey at 12%) with an Ink border and the icon tile switching to vermilion; text stays Ink. Hover 12% white wash. Focus ring inset.

### Timeline (signature)
Three stacked 52px lanes in one 8px panel sharing one x-axis, each with a 4px-radius Well over chart paper. Pattern-group blocks carry their letters; the current block is full strength and the others dim only their background to 45% so letters stay legible. Loudness is a sky-blue area with the played part in vermilion; visual change is a 2px sky-blue line. One slider layer carries the 2px vermilion playhead with a vermilion mono time tag above; axis ticks below.

### Group chip
32px square, 4px corners, group hue fill, letter in 800 Deep, used in the live "now" description.

### Panels
8px, Panel tint, Rule border, 20-24px padding: on-screen, now, upload form, step cards, pipeline.

### Pipeline walkthrough
A panel with a control bar (sound checkbox, a row of chip buttons for style choice) above a stage: frame thumbnails (96px, 6px radius, 2px border that turns vermilion when current, 50% to 100% opacity by progress), dashed flow arrows that animate in vermilion while running, eight bars with vermilion fill and mono values, a map of dots, and a piano roll in a Well.

### Sound-only view
A toggle hides the footage frame and shows the current group's letter at display size (clamp 7rem to 11rem, 800) over a "sound only" caption on the dark frame.

### Upload panel
Dashed drop zone in a Well, ghost "choose" button, fieldset of check chips (44px, 1px Rule border, 4px radius, native vermilion checkbox, checked adds Ink border and 10% wash), full-width Play-style go button, 8px progress well with vermilion fill, and a white error note with dark text.

### Navigation
Top bar on near-black with a hairline bottom border; wordmark with a line-drawn globe-waveform mark; links 500 weight, 44px minimum, 2px vermilion underline on hover. Skip link turns vermilion on focus.

## Data tracks (stem groups)

The eight data tracks (the "Three Voices" pipeline: colour-bar inversion, data notes, model harmony, recorded Earth voices, fidelity check) play in the same stage as the hurricane season. They add no new colours: their seven stem groups map onto the pen, the four group hues and ink. The tokens live on `.jb` as `--g-<group>`; the class `g-<group>` sets `--g` for a chip, a lane line or a roll note.

| Key | Group | Token | Colour | Why |
|---|---|---|---|---|
| 1 | Melody | `--g-melody: var(--accent)` | Pen Vermilion | The melody is the data line the recorder draws: every note is a value. |
| 2 | Bands | `--g-bands: var(--r2)` | Group C Lilac | Latitude bands and per-series voices, also data. |
| 3 | Events | `--g-events: var(--r1)` | Group B Amber | Hits and ticks for discrete events. |
| 4 | Harmony | `--g-harmony: var(--ink-3)` | Ink 3 | Written by a model and carries no data, so it gets no hue. |
| 5 | Geophony | `--g-geophony: var(--r0)` | Group A Sky | Wind, water, rain, thunder. |
| 6 | Anthrophony | `--g-anthrophony: var(--ink)` | Ink | Traffic, city hum, engines. |
| 7 | Biophony | `--g-biophony: var(--r3)` | Group D Sage | Birds, frogs, insects. |

### Named Rules
**The Number Beside Color Rule.** A stem-group colour never appears without its key number 1–7 (the mute shortcut) and its name, the way a pattern group carries its letter. Pattern groups keep A–D; stem groups use numbers, so the two never collide on one screen.
**The Pen Draws The Data.** On a data track the melody takes the pen colour. Everywhere else the One Pen Rule holds: selection of a track, a version, a group or a pressed tool is the quiet blue-grey tint with an Ink border. The roll keeps the played part at full strength and dims the rest, like the played loudness curve.

### Components
- **Track chooser.** One radiogroup above the stage: tiles in the style-row language (6px, Panel, Rule border, 56px), quiet tint when chosen, name in 800 over a small Ink 3 tag ("Artistic sonification" for the hurricane season, "Data-faithful" plus R² in mono for the data tracks). A single scrolling row under 720px.
- **Group rows (mixer).** The style list's row, with a 32px group chip carrying the key number in mono, a checkbox (vermilion accent), the group name, a volume slider and the key hint. A muted group's chip turns to an Ink 3 outline so the number stays legible.
- **Data timeline.** The hurricane timeline's parts: melody value as the sky area with its played part in vermilion, the Earth voices as 2px lines in their group colours with numbered mini chips in the lane label, sections as neutral blocks. Same slider layer and playhead.
- **Right now.** A Panel with the current value in mono (2.25rem), units, trend and spoken date, then listening tools as chip buttons with key hints; pressed tools use `aria-pressed` and the quiet tint.
- **Score, fidelity, legend.** Panels below the stage. The roll is a canvas in a Well (no chart paper: the paper stays under the lanes), with a text summary and a table of melody notes. Fidelity bars are 10px Wells with a vermilion fill for the real data, numbers in mono. The shuffled-control test is described in one sentence on How it works; the site does not play it.
- **Compare.** The style list again, one row per version, the icon tile carrying the version number. Ratings are native radios in check chips plus a textarea in a Well; Export is a ghost button.
- **Shortcut dialog.** A native dialog on Deep with a Rule border and 8px corners, holding the shortcut table.
- **Forced colours.** Chips keep a CanvasText outline and their number; chosen and pressed states get a Highlight outline; the playhead, sweep line and fidelity fills use Highlight.

## Do's and Don'ts

### Do:
- **Do** reserve vermilion for play, played, playhead, focus and progress.
- **Do** pair every group colour with its letter and give it dark ink on bright fills.
- **Do** mark selection with a quiet blue-grey tint and Ink border.
- **Do** keep hit targets at 44px or more (56px transport).
- **Do** use 3px vermilion focus rings with 3px offset (inset on the style list).
- **Do** set every number in JetBrains Mono, tabular.
- **Do** disable transitions and animation under `prefers-reduced-motion`.
- **Do** pair every stem-group colour with its key number and name.

### Don't:
- **Don't** add a second action colour, glass, glow or a starfield image.
- **Don't** put chart-paper ruling on the page ground or panels; it is for timeline lanes only.
- **Don't** fill a selected choice with white or vermilion; vermilion is not selection.
- **Don't** use radii beyond 8px or below the role-based set above.
- **Don't** set prose in JetBrains Mono.
- **Don't** port this look into `/lab`.

Not canonized (defects the build carries): the uppercase tracked line under the wordmark and the uppercase tracked `dt` labels in the roles list are kicker-style labels; the vermilion-tinted drop shadow under Play reads as glow; `!important` colour overrides on `.jb-fine`, `.jb-using`, `.jb-now-region`, `.jb-region`; chip buttons in the pipeline bar still select with a white fill, which contradicts the quiet-tint selection rule; the video frame's 1px white ring plus 24px shadow stacks heavier than other surfaces.
