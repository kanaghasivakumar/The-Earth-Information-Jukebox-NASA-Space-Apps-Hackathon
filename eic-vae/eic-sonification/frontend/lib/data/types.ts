/**
 * The data-track contract (web/README.md, "Track contract"), as TypeScript.
 * Files live in public/jukebox/tracks/<id>/: track.json, timeline.json, controls.json and the stems.
 */

export const GROUPS = ["melody", "bands", "events", "harmony", "geophony", "anthrophony", "biophony"] as const;
export type Group = (typeof GROUPS)[number];

export interface Stem {
  name: string;
  file: string;
  group: Group;
  label?: string;
}

/** One note of the score. `value` is in the track's units, or null for model-written notes (harmony). */
export interface TimelineNote {
  start: number;
  end: number;
  pitch: number;
  velocity: number;
  voice: string;
  value: number | null;
  pan: number;
}

export type Timeline = TimelineNote[];

export interface Controls {
  t: number[];
  curves: Partial<Record<"geophony" | "anthrophony" | "biophony" | "brightness", number[]>>;
  synthetic?: boolean;
}

export interface Fidelity {
  r2: number | null;
  pearson_r?: number | null;
  mae?: number | null;
  n?: number | null;
  control_r2?: number | null;
  mix_r2?: number | null;
  pitch_within_half_semitone?: number | null;
  mix_pitch_within_half_semitone?: number | null;
  method?: string;
  /** per-series results when there is no single lead (5669) */
  detail?: Record<string, { final_stem?: { n?: number; r2?: number; pitch_within_half_semitone?: number } }>;
}

export interface PitchMapSpec {
  vmin: number;
  vmax: number;
  low?: number;
  span?: number;
  root?: number;
  scale?: string;
  polarity?: number;
}

export interface SweepVoice {
  voice: string;
  label?: string;
  pan?: number;
  pitch_offset?: number;
}

export interface Sweep {
  label?: string;
  voices?: (SweepVoice | string)[];
  duration_s?: number;
  lon_start?: number;
  lon_end?: number;
}

export interface Calendar {
  ref_date: string;
  t_ref: number;
  sec_per_year: number;
  clamp?: [string, string];
}

export interface Segment {
  name: string;
  start: number;
  end: number;
}

export interface LegendEntry {
  voice: string;
  meaning: string;
}

export interface DataTrack {
  id: string;
  title: string;
  svs_url: string;
  video_url: string | null;
  video_alt_url?: string;
  /** movie seconds per audio second (default 1); below 1 the video plays slowed down */
  video_rate?: number;
  image_url?: string;
  duration_s: number;
  units: string;
  description: string;
  legend: LegendEntry[];
  stems: Stem[];
  timeline: string;
  controls: string;
  fidelity: Fidelity;
  data_source: string | Record<string, unknown>;
  control?: { file: string; replaces?: Group; label?: string };
  mix?: string;
  pitch_map?: PitchMapSpec;
  band_pitch_map?: PitchMapSpec;
  sweep?: Sweep;
  lead_voice?: string;
  lead_voices?: string[];
  voice_groups?: Record<string, Group>;
  time_axis?: { start: string; end: string; format?: "year" | "month" | "day" };
  calendar?: Calendar;
  dates?: string[];
  dates_t?: number[];
  segments?: Segment[];
  caveat?: string;
  voice?: string;
  summary?: string;
}

export interface TrackIndexEntry {
  id: string;
  title: string;
  voice?: string;
  summary?: string;
  duration_s?: number;
  order?: number;
}

/** A version of a track for the Compare view (versions.json, written by scripts/export_next_app.py). */
export interface Version {
  id: string;
  label: string;
  kind: "deterministic" | "learned" | "control";
  file: string;
  line: string;
  fidelity: {
    r2: number | null;
    within_half_semitone: number | null;
    control_r2: number | null;
    control_kind: string | null;
    note: string | null;
  };
}

export interface VersionsFile {
  id: string;
  duration_s: number;
  versions: Version[];
}

/** Everything the player needs once a track is loaded. */
export interface LoadedTrack {
  track: DataTrack;
  timeline: Timeline;
  controls: Controls | null;
  base: string;
}
