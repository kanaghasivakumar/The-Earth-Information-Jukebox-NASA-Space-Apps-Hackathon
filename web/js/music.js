// Pitch-map helpers mirroring jukebox/score/mapping.py, and text formatting.

export const GROUPS = ['melody', 'bands', 'events', 'harmony', 'geophony', 'anthrophony', 'biophony'];
export const GROUP_LABELS = {
  melody: 'Melody', bands: 'Bands', events: 'Events', harmony: 'Harmony',
  geophony: 'Geophony (Earth sounds)', anthrophony: 'Anthrophony (human sounds)', biophony: 'Biophony (living sounds)',
};

const SCALES = {
  major_pent: [0, 2, 4, 7, 9], minor_pent: [0, 3, 5, 7, 10], major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10], dorian: [0, 2, 3, 5, 7, 9, 10], chromatic: [...Array(12).keys()],
};

export const midiHz = (p) => 440 * 2 ** ((p - 69) / 12);

export class PitchMap {
  constructor({ vmin, vmax, low = 48, span = 36, root = 0, scale = 'major_pent', polarity = 1 }) {
    Object.assign(this, { vmin, vmax, low, span, root, scale, polarity });
    const pcs = (SCALES[scale] || SCALES.chromatic).map((s) => (root + s) % 12);
    this.allowed = [];
    for (let p = low; p <= low + span; p++) if (pcs.includes(((p % 12) + 12) % 12)) this.allowed.push(p);
  }
  /** value -> MIDI pitch, snapped to the scale. polarity: +1 higher value = higher pitch, -1 inverted. */
  pitch(v, polarity = this.polarity) {
    let x = (v - this.vmin) / (this.vmax - this.vmin || 1);
    x = Math.min(1, Math.max(0, polarity > 0 ? x : 1 - x));
    const c = this.low + x * this.span;
    return this.allowed.reduce((a, b) => (Math.abs(b - c) < Math.abs(a - c) ? b : a), this.allowed[0]);
  }
  get vmid() { return (this.vmin + this.vmax) / 2; }
}

const NAMES = ['C', 'C sharp', 'D', 'D sharp', 'E', 'F', 'F sharp', 'G', 'G sharp', 'A', 'A sharp', 'B'];
const SHORT = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
export const noteName = (p, spoken = false) => (spoken ? NAMES : SHORT)[p % 12] + (Math.floor(p / 12) - 1);

export function fmtTime(t) {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}
export function spokenTime(t) {
  t = Math.max(0, Math.round(t || 0));
  const m = Math.floor(t / 60), s = t % 60;
  if (!m) return `${s} second${s === 1 ? '' : 's'}`;
  return `${m} minute${m === 1 ? '' : 's'}${s ? ` ${s} second${s === 1 ? '' : 's'}` : ''}`;
}

/** Choose decimals so the value carries about 3 significant digits of the data range. */
export function fmtValue(v, units = '', range = 10) {
  if (v == null || !Number.isFinite(v)) return 'no data';
  const d = range < 2 ? 2 : range < 50 ? 1 : 0;
  return units ? `${v.toFixed(d)} ${units}` : v.toFixed(d);
}
/** Screen readers read "°C" inconsistently; spell it out for speech. */
export function spokenUnits(u = '') {
  return u.replace('°C', 'degrees Celsius').replace('°F', 'degrees Fahrenheit').replace('°', ' degrees').replace(/%/g, ' percent');
}
/** R² etc. for display (true minus sign) and for speech ("minus"). */
export const fmtNum = (x, d = 2) => x.toFixed(d).replace('-', '\u2212');
export const spokenNum = (x, d = 2) => x.toFixed(d).replace('-', 'minus ');
export const spokenValue = (v, units, range) => {
  const s = fmtValue(v, '', range);
  return s === 'no data' ? s : `${s.replace('-', 'minus ')} ${spokenUnits(units)}`.trim();
};
