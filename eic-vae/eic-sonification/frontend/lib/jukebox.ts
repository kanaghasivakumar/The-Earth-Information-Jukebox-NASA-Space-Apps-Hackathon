/** Shared types and helpers for the hurricane-season player (the VAE track and uploads). */

/** [start seconds, end seconds, pattern group id] */
export type Region = [number, number, number];

export interface Instrument {
  role: string;
  sound: string;
}

export interface Track {
  id: string;
  tempo: number;
  scale: string;
  duration: number;
  audio: string;
  midi: string;
  instruments: Instrument[];
  /** loudness of the rendered audio, 0..1, evenly spaced over the track */
  loudness: number[];
  regions: Region[];
}

export interface Signals {
  sourceDuration: number;
  firstTimestamp: number;
  lastTimestamp: number;
  frames: number;
  /** how fast the learned numbers change, 0..1, evenly spaced over the source */
  movement: number[];
}

export interface Scene {
  from: number;
  text: string;
}

/** Pattern-group hues, in the order of the letters A to D (DESIGN.md group-a..group-d). */
export const REGION_COLORS = ["#7cc4ff", "#ffb454", "#b79cff", "#7fd1ae"];

export const STYLE_COPY: Record<string, { name: string; line: string; hear: string }> = {
  ambient: {
    name: "Ambient",
    line: "Soft, spacious tones leave room to follow the picture.",
    hear: "Listen for the flute: it carries the motif of each pattern group, over a warm pad and a music box.",
  },
  cinematic: {
    name: "Cinematic",
    line: "A sweeping arrangement follows the changing picture.",
    hear: "Strings swell when the picture moves fast, and taiko-style drums push the busiest stretches.",
  },
  classical: {
    name: "Classical",
    line: "A familiar ensemble turns the same signals into a measured score.",
    hear: "The violin takes the motifs, with piano and cello underneath and timpani at the turns.",
  },
  jazz: {
    name: "Jazz",
    line: "A nimble, syncopated reading gives the motion a playful pulse.",
    hear: "The tenor sax states each motif, and the brush kit gets busier as the storm season picks up.",
  },
  electronic: {
    name: "Electronic",
    line: "Bright synthesized textures trace shifts in the learned pattern.",
    hear: "A saw lead carries the motifs over a pulsing synth bass, the one style here in a minor key.",
  },
};

/** Written by the team from the footage itself. The model never sees these words. */
export const SCENES: Scene[] = [
  {
    from: 0,
    text: "A globe turned to the Atlantic in early summer 2022. The ocean is coloured by surface temperature: blue is cooler, orange and red are warmer. White is cloud and green is rain.",
  },
  { from: 40, text: "Through July the warm water spreads north across the Atlantic, and the Gulf of Mexico deepens to dark red." },
  { from: 85, text: "August: clouds and green rain bands drift west off Africa across the tropical Atlantic." },
  { from: 130, text: "September: thin lines start to trace the paths the season's storms have taken across the ocean." },
  { from: 175, text: "Late September and October: more storm tracks curve north over the Caribbean and the Gulf of Mexico." },
  { from: 220, text: "October into November: the northern Atlantic cools to blue as the season winds down, its storm tracks still drawn on the globe." },
];

export const regionLetter = (id: number) => String.fromCharCode(65 + (((id % 26) + 26) % 26));

export function fmt(t: number): string {
  const s = Math.max(0, Math.floor(t || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** The region under time t (the last region that has started, so gaps keep the previous group). */
export function regionAt(regions: Region[], t: number): Region | null {
  let found: Region | null = null;
  for (const r of regions) {
    if (r[0] <= t) found = r;
    else break;
  }
  return found;
}

/** Start of the most recent earlier visit to the same pattern group, or null on its first appearance. */
export function earlierVisit(regions: Region[], region: Region): number | null {
  let back: number | null = null;
  for (const r of regions) {
    if (r[0] >= region[0]) break;
    if (r[2] === region[2]) back = r[0];
  }
  return back;
}

export function sceneAt(t: number): Scene {
  let s = SCENES[0];
  for (const x of SCENES) if (x.from <= t) s = x;
  return s;
}
