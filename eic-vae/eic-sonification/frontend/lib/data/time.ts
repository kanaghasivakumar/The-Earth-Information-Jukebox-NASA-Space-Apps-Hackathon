/** Where we are in a data track, in words: spoken dates, the lead value and its trend. */
import { fmtValue, spokenTime, spokenValue } from "./music";
import type { Controls, DataTrack, TimelineNote } from "./types";
import type { Prepared } from "./load";

const LOCALE = "en-US";

export function fmtTime(t: number): string {
  const s = Math.max(0, Math.floor(t || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Index of the last lead note that has started by time t, or -1. Binary search; notes are sorted. */
export function leadIndexAt(lead: TimelineNote[], t: number): number {
  let lo = 0;
  let hi = lead.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (lead[m].start <= t + 1e-6) {
      ans = m;
      lo = m + 1;
    } else hi = m - 1;
  }
  return ans;
}

/** The note sounding at t, or else the last one that started. */
export function noteAt(notes: TimelineNote[], t: number): TimelineNote | null {
  let best: TimelineNote | null = null;
  for (const n of notes) {
    if (n.start <= t + 1e-6) {
      if (t < n.end) return n;
      best = n;
    } else break;
  }
  return best;
}

const ymd = (s: string) => (/^\d{8}$/.test(s) ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}` : s);

export function dateLabel(iso: string): string {
  if (/^\d{4}$/.test(iso)) return iso;
  const d = new Date(`${ymd(iso)}T12:00:00Z`);
  return Number.isNaN(+d) ? String(iso) : d.toLocaleDateString(LOCALE, { dateStyle: "medium", timeZone: "UTC" });
}

export function segmentAt(track: DataTrack, t: number): string | null {
  const seg = (track.segments ?? []).find((s) => t >= s.start && t < s.end);
  return seg ? seg.name : null;
}

/**
 * The date (or movie time) at t, for speech: `calendar` (5011), `dates` with optional `dates_t`
 * (5209, 5313, 5176), `time_axis`, else the time itself. A segment name goes first (5313).
 */
export function timeLabel(track: DataTrack, t: number, duration = track.duration_s): string {
  const seg = segmentAt(track, t);
  const pre = seg ? `${seg}, ` : "";
  const cal = track.calendar;
  if (cal?.ref_date && cal.sec_per_year) {
    let ms = Date.parse(`${cal.ref_date}T00:00:00Z`) + ((t - cal.t_ref) / cal.sec_per_year) * 365.2425 * 864e5;
    if (cal.clamp) ms = Math.min(Math.max(ms, Date.parse(`${cal.clamp[0]}T00:00:00Z`)), Date.parse(`${cal.clamp[1]}T00:00:00Z`));
    return pre + new Date(ms).toLocaleDateString(LOCALE, { year: "numeric", month: "long", timeZone: "UTC" });
  }
  if (track.dates?.length) {
    let i: number;
    if (track.dates_t?.length === track.dates.length) {
      i = -1;
      for (let k = 0; k < track.dates_t.length && track.dates_t[k] <= t + 1e-6; k++) i = k;
    } else i = Math.floor((t / (duration || 1)) * track.dates.length);
    if (i >= 0) return pre + dateLabel(track.dates[Math.min(i, track.dates.length - 1)]);
  }
  const ax = track.time_axis;
  if (ax?.start && ax?.end) {
    const a = Date.parse(ax.start);
    const b = Date.parse(ax.end);
    const d = new Date(a + (b - a) * (t / (duration || 1)));
    const opts: Intl.DateTimeFormatOptions =
      ax.format === "year" ? { year: "numeric" } : ax.format === "day" ? { dateStyle: "medium" } : { year: "numeric", month: "long" };
    return pre + d.toLocaleDateString(LOCALE, { ...opts, timeZone: "UTC" });
  }
  return pre + spokenTime(t);
}

/** True when the track can speak a date (or a segment) rather than only the movie time. */
export const hasDates = (track: DataTrack) => !!(track.calendar || track.dates?.length || track.time_axis || track.segments?.length);

/** How far the sweep line has crossed a still image (5669), 0..1. */
export function sweepFraction(track: DataTrack, t: number, duration = track.duration_s): number {
  return Math.min(1, Math.max(0, t / (track.sweep?.duration_s || duration || 1)));
}

export function describeNow(
  track: DataTrack,
  prep: Prepared,
  t: number,
  { full = false, controls = null as Controls | null } = {},
): string {
  const u = track.units || "";
  const i = leadIndexAt(prep.lead, t);
  const sw = track.sweep;
  if (!prep.lead.length && sw?.lon_start != null && sw.lon_end != null) {
    const lon = sw.lon_start + sweepFraction(track, t) * (sw.lon_end - sw.lon_start);
    return `${spokenTime(t)}: sweep at longitude ${Math.abs(Math.round(lon))} degrees ${lon < 0 ? "west" : "east"}.`;
  }
  if (i < 0) return `${timeLabel(track, t)}. No data yet.`;
  const n = prep.lead[i];
  const prev = prep.lead[i - 1];
  let msg = `${timeLabel(track, t)}: ${spokenValue(n.value, u, prep.range)}`;
  if (prev && full && n.value != null && prev.value != null) {
    msg += n.value > prev.value + 1e-9 ? ", rising" : n.value < prev.value - 1e-9 ? ", falling" : ", steady";
  }
  if (full && controls && !controls.synthetic) {
    const k = controls.t.findIndex((x) => x >= t);
    const j = k < 0 ? controls.t.length - 1 : k;
    const parts = (["geophony", "anthrophony", "biophony"] as const)
      .map((v) => [v, controls.curves[v]?.[j]] as const)
      .filter(([, x]) => x != null && x > 0.05);
    if (parts.length) msg += `. Voices: ${parts.map(([v, x]) => `${v} ${Math.round((x as number) * 100)} percent`).join(", ")}`;
  }
  return `${msg}.`;
}

/** The text alternative to the score picture: start, end, lowest, highest and trend of the lead. */
export function scoreSummary(track: DataTrack, prep: Prepared): string {
  const { lead } = prep;
  if (!lead.length) {
    return `This track has no single lead melody. ${track.stems.length} stems play together; the legend says what each one follows.`;
  }
  const u = track.units || "";
  const val = (n: TimelineNote) => n.value as number;
  const lo = lead.reduce((a, b) => (val(b) < val(a) ? b : a));
  const hi = lead.reduce((a, b) => (val(b) > val(a) ? b : a));
  const first = lead[0];
  const last = lead[lead.length - 1];
  const mid = lead.length >> 1;
  const avg = (a: TimelineNote[]) => a.reduce((s, n) => s + val(n), 0) / Math.max(1, a.length);
  const d = avg(lead.slice(mid)) - avg(lead.slice(0, mid));
  const trend =
    Math.abs(d) < 0.1 * (prep.vmax - prep.vmin) ? "no clear overall trend" : d > 0 ? "higher on average in the second half" : "lower on average in the second half";
  const f = (v: number) => fmtValue(v, shortU(u), prep.range);
  return (
    `${lead.length} lead notes over ${spokenTime(track.duration_s)}. ` +
    `The value starts at ${f(val(first))} and ends at ${f(val(last))}; ` +
    `lowest ${f(val(lo))} at ${fmtTime(lo.start)}, highest ${f(val(hi))} at ${fmtTime(hi.start)}; ${trend}.`
  );
}

const shortU = (u: string) => u.split(/[;(]/)[0].trim();
