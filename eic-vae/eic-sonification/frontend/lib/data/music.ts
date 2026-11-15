/** Pitch map (mirrors jukebox/score/mapping.py) and number, time and unit formatting for speech and display. */
import type { Group, PitchMapSpec } from "./types";

export const GROUP_LABELS: Record<Group, string> = {
  melody: "Melody",
  bands: "Bands",
  events: "Events",
  harmony: "Harmony",
  geophony: "Geophony",
  anthrophony: "Anthrophony",
  biophony: "Biophony",
};

/** What each group is, in plain words. */
export const GROUP_LINES: Record<Group, string> = {
  melody: "The data line. Every note is a data value.",
  bands: "One voice per latitude band or series, also data.",
  events: "Hits and ticks for discrete events in the data.",
  harmony: "Strings and bass written by a model. No data.",
  geophony: "Recorded Earth sounds: wind, water, rain, thunder.",
  anthrophony: "Recorded human sounds: traffic, city hum, engines.",
  biophony: "Recorded living sounds: birds, frogs, insects.",
};

const SCALES: Record<string, number[]> = {
  major_pent: [0, 2, 4, 7, 9],
  minor_pent: [0, 3, 5, 7, 10],
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

export const midiHz = (p: number) => 440 * 2 ** ((p - 69) / 12);

export class PitchMap {
  vmin: number;
  vmax: number;
  low: number;
  span: number;
  polarity: number;
  allowed: number[];

  constructor({ vmin, vmax, low = 48, span = 36, root = 0, scale = "major_pent", polarity = 1 }: PitchMapSpec) {
    this.vmin = vmin;
    this.vmax = vmax;
    this.low = low;
    this.span = span;
    this.polarity = polarity;
    const pcs = (SCALES[scale] ?? SCALES.chromatic).map((s) => (root + s) % 12);
    this.allowed = [];
    for (let p = low; p <= low + span; p++) if (pcs.includes(((p % 12) + 12) % 12)) this.allowed.push(p);
  }

  /** value -> MIDI pitch, snapped to the scale. polarity +1: higher value plays higher; -1 inverted. */
  pitch(v: number, polarity = this.polarity): number {
    let x = (v - this.vmin) / (this.vmax - this.vmin || 1);
    x = Math.min(1, Math.max(0, polarity > 0 ? x : 1 - x));
    const c = this.low + x * this.span;
    return this.allowed.reduce((a, b) => (Math.abs(b - c) < Math.abs(a - c) ? b : a), this.allowed[0]);
  }

  get vmid() {
    return (this.vmin + this.vmax) / 2;
  }
}

const NAMES = ["C", "C sharp", "D", "D sharp", "E", "F", "F sharp", "G", "G sharp", "A", "A sharp", "B"];
const SHORT = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
export const noteName = (p: number, spoken = false) => (spoken ? NAMES : SHORT)[((p % 12) + 12) % 12] + (Math.floor(p / 12) - 1);

export function spokenTime(t: number): string {
  const r = Math.max(0, Math.round(t || 0));
  const m = Math.floor(r / 60);
  const s = r % 60;
  if (!m) return `${s} second${s === 1 ? "" : "s"}`;
  return `${m} minute${m === 1 ? "" : "s"}${s ? ` ${s} second${s === 1 ? "" : "s"}` : ""}`;
}

/** Decimals that carry about three significant digits of the data range. */
export function fmtValue(v: number | null | undefined, units = "", range = 10): string {
  if (v == null || !Number.isFinite(v)) return "no data";
  const d = range < 2 ? 2 : range < 50 ? 1 : 0;
  return units ? `${v.toFixed(d)} ${units}` : v.toFixed(d);
}

/** Screen readers read "°C" inconsistently; spell it out for speech. */
export function spokenUnits(u = ""): string {
  return u.replace("°C", "degrees Celsius").replace("°F", "degrees Fahrenheit").replace("°", " degrees").replace(/%/g, " percent");
}

/** Short units for display next to a big number (the part before any parenthesis or semicolon). */
export const shortUnits = (u = "") => u.split(/[;(]/)[0].trim();

/** R² and similar, with a true minus sign for display. */
export const fmtNum = (x: number, d = 2) => x.toFixed(d).replace("-", "−");
export const spokenNum = (x: number, d = 2) => x.toFixed(d).replace("-", "minus ");

export function spokenValue(v: number | null | undefined, units: string, range: number): string {
  const s = fmtValue(v, "", range);
  return s === "no data" ? s : `${s.replace("-", "minus ")} ${spokenUnits(shortUnits(units))}`.trim();
}
