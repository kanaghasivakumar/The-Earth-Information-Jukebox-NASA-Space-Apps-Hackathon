/** Loader for the data-track contract: fetch, check, and derive the lead line and groups. */
import { PitchMap } from "./music";
import { GROUPS, type Controls, type DataTrack, type Group, type LoadedTrack, type Timeline, type TimelineNote, type VersionsFile } from "./types";

export const TRACKS_BASE = "/jukebox/tracks";

const REQUIRED = ["id", "title", "svs_url", "duration_s", "units", "description", "legend", "stems", "timeline", "controls", "fidelity", "data_source"] as const;

export class TrackError extends Error {}

type Fetcher = (url: string) => Promise<{ ok: boolean; status: number; json: () => Promise<unknown> }>;

async function getJSON<T>(fetcher: Fetcher, url: string): Promise<T> {
  const r = await fetcher(url);
  if (!r.ok) throw new TrackError(`${url}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

/** Validate a parsed track.json. Throws TrackError naming the first problem. */
export function parseTrack(raw: unknown): DataTrack {
  if (!raw || typeof raw !== "object") throw new TrackError("track.json is not an object");
  const t = raw as Record<string, unknown>;
  for (const k of REQUIRED) if (!(k in t)) throw new TrackError(`track.json is missing "${k}"`);
  if (!t.video_url && !t.image_url) throw new TrackError("track.json has neither video_url nor image_url");
  if (!Array.isArray(t.stems) || !t.stems.length) throw new TrackError("track.json has no stems");
  for (const s of t.stems as { name?: string; group?: string }[]) {
    if (!GROUPS.includes(s.group as Group)) throw new TrackError(`stem ${s.name} has an unknown group "${s.group}"`);
  }
  if (typeof t.duration_s !== "number" || !(t.duration_s > 0)) throw new TrackError("duration_s must be a positive number");
  return { ...(t as unknown as DataTrack), id: String(t.id) };
}

/** Validate timeline.json: numbers everywhere, except `value`, which may be null (model-written notes). */
export function parseTimeline(raw: unknown): Timeline {
  if (!Array.isArray(raw)) throw new TrackError("timeline.json is not a list");
  const out: Timeline = raw.map((n, i) => {
    const x = n as Partial<TimelineNote>;
    if (typeof x.start !== "number" || typeof x.end !== "number" || typeof x.pitch !== "number") {
      throw new TrackError(`timeline note ${i} needs numeric start, end and pitch`);
    }
    const value = x.value == null || !Number.isFinite(x.value) ? null : x.value;
    return { start: x.start, end: x.end, pitch: x.pitch, velocity: x.velocity ?? 80, voice: String(x.voice ?? ""), value, pan: x.pan ?? 0 };
  });
  return out.sort((a, b) => a.start - b.start);
}

export function parseControls(raw: unknown): Controls | null {
  if (!raw || typeof raw !== "object") return null;
  const c = raw as Controls;
  if (!Array.isArray(c.t) || !c.curves || typeof c.curves !== "object") return null;
  return c;
}

export async function loadTrack(id: string, fetcher: Fetcher = fetch, base = TRACKS_BASE): Promise<LoadedTrack> {
  const dir = `${base}/${encodeURIComponent(id)}/`;
  const track = parseTrack(await getJSON(fetcher, `${dir}track.json`));
  const timeline = parseTimeline(await getJSON(fetcher, dir + (track.timeline || "timeline.json")));
  let controls: Controls | null = null;
  if (track.controls) {
    try {
      controls = parseControls(await getJSON(fetcher, dir + track.controls));
    } catch {
      controls = null;
    }
  }
  return { track, timeline, controls, base: dir };
}

export async function loadVersions(id: string, fetcher: Fetcher = fetch, base = TRACKS_BASE): Promise<VersionsFile> {
  const v = await getJSON<VersionsFile>(fetcher, `${base}/${encodeURIComponent(id)}/versions.json`);
  if (!Array.isArray(v.versions) || !v.versions.length) throw new TrackError("versions.json lists no versions");
  return v;
}

/** Which stem group a timeline voice belongs to (voice_groups, then lead_voice, then name prefix). */
export function groupOf(track: Pick<DataTrack, "voice_groups" | "lead_voice">, voice: string): Group {
  const g = track.voice_groups?.[voice];
  if (g && GROUPS.includes(g)) return g;
  if (voice === (track.lead_voice || "lead")) return "melody";
  if (voice.startsWith("band")) return "bands";
  return GROUPS.find((x) => voice.toLowerCase().startsWith(x)) ?? "harmony";
}

export interface Prepared {
  /** the notes the readout, table and polarity flip use; never a null value */
  lead: TimelineNote[];
  byGroup: Partial<Record<Group, TimelineNote[]>>;
  vmin: number;
  vmax: number;
  /** the value range used to pick decimals */
  range: number;
  pm: PitchMap | null;
  bandPm: PitchMap | null;
  bandPms: Record<string, PitchMap>;
  /** groups that have a stem (or the control) in this track, in GROUPS order */
  present: Group[];
}

export function prepare(track: DataTrack, timeline: Timeline): Prepared {
  const leadVoices = track.lead_voices?.length ? new Set(track.lead_voices) : null;
  const lead = timeline.filter(
    (n) => (leadVoices ? leadVoices.has(n.voice) : groupOf(track, n.voice) === "melody") && n.value != null && Number.isFinite(n.value),
  );
  const byGroup: Partial<Record<Group, TimelineNote[]>> = {};
  for (const n of timeline) (byGroup[groupOf(track, n.voice)] ??= []).push(n);
  const vals = lead.map((n) => n.value as number);
  const vmin = vals.length ? Math.min(...vals) : 0;
  const vmax = vals.length ? Math.max(...vals) : 1;
  const range = (track.pitch_map ? track.pitch_map.vmax - track.pitch_map.vmin : vmax - vmin) || 10;
  const pm = track.pitch_map ? new PitchMap(track.pitch_map) : null;
  const bandPm = track.band_pitch_map ? new PitchMap(track.band_pitch_map) : null;
  const bandPms: Record<string, PitchMap> = {};
  for (const v of track.sweep?.voices ?? []) {
    if (bandPm && track.band_pitch_map && typeof v === "object") {
      bandPms[v.voice] = new PitchMap({ ...track.band_pitch_map, low: bandPm.low + (v.pitch_offset || 0) });
    }
  }
  const has = new Set<Group>(track.stems.map((s) => s.group));
  if (track.control) has.add(track.control.replaces ?? "melody");
  const present = GROUPS.filter((g) => has.has(g));
  return { lead, byGroup, vmin, vmax, range, pm, bandPm, bandPms, present };
}

/** Voices the sweep (S) plays, north to south, with their pan. */
export function sweepVoices(track: DataTrack, timeline: Timeline): { voice: string; label: string; pan: number }[] {
  if (track.sweep?.voices) {
    return track.sweep.voices.map((v) => (typeof v === "string" ? { voice: v, label: v, pan: 0 } : { voice: v.voice, label: v.label ?? v.voice, pan: v.pan ?? 0 }));
  }
  const vs = [...new Set(timeline.filter((n) => groupOf(track, n.voice) === "bands").map((n) => n.voice))];
  return vs.length > 1 ? vs.map((v, i) => ({ voice: v, label: v, pan: -1 + (2 * i) / (vs.length - 1) })) : [];
}
