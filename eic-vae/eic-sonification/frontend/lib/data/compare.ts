/** Compare view: version switching (pure) and ratings stored in localStorage. */
import type { Version } from "./types";
import { fmtNum, spokenNum } from "./music";

export interface SwitchResult {
  selected: number;
  /** the position does not move when the version changes */
  position: number;
  gains: number[];
  announcement: string;
}

/** Choose version `index` at `position` seconds. Out-of-range choices keep the current version. */
export function switchVersion(versions: Version[], current: number, index: number, position: number): SwitchResult {
  const selected = index >= 0 && index < versions.length ? index : current;
  const v = versions[selected];
  const r2 = v.fidelity.r2;
  return {
    selected,
    position,
    gains: versions.map((_, k) => (k === selected ? 1 : 0)),
    announcement: `${v.label}${r2 != null ? `, R squared ${spokenNum(r2)}` : ""}. Same moment.`,
  };
}

/** One-line fidelity for a version, for display. */
export function fidelityLine(v: Version): string {
  const f = v.fidelity;
  if (v.kind === "control") return f.r2 != null ? `R² ${fmtNum(f.r2)} against the real data` : "No fidelity number";
  const parts: string[] = [];
  if (f.r2 != null) parts.push(`R² ${fmtNum(f.r2, 3)}`);
  if (f.within_half_semitone != null) parts.push(`${Math.round(f.within_half_semitone * 100)}% of notes within ½ semitone`);
  if (f.control_r2 != null) parts.push(`shuffled ${fmtNum(f.control_r2)}`);
  return parts.join(" · ") || "No fidelity number";
}

// ---------------------------------------------------------------- ratings

export const RATINGS_KEY = "earth-jukebox:ratings:v1";

export interface Rating {
  rating: number | null;
  notes: string;
  updated: string;
}

/** trackId -> versionId -> rating */
export type Ratings = Record<string, Record<string, Rating>>;

type StorageLike = Pick<Storage, "getItem" | "setItem">;

function storage(): StorageLike | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null; // blocked storage throws on access
  }
}

export function loadRatings(store: StorageLike | null = storage()): Ratings {
  if (!store) return {};
  try {
    const raw = store.getItem(RATINGS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as Ratings) : {};
  } catch {
    return {};
  }
}

/** Returns the new ratings. Saving is best effort: a full or blocked storage keeps the in-memory copy. */
export function setRating(
  ratings: Ratings,
  trackId: string,
  versionId: string,
  patch: Partial<Pick<Rating, "rating" | "notes">>,
  store: StorageLike | null = storage(),
  now: () => Date = () => new Date(),
): { ratings: Ratings; saved: boolean } {
  const prev = ratings[trackId]?.[versionId] ?? { rating: null, notes: "", updated: "" };
  let rating = patch.rating !== undefined ? patch.rating : prev.rating;
  if (rating != null) rating = Math.min(5, Math.max(1, Math.round(rating)));
  const entry: Rating = { rating, notes: patch.notes !== undefined ? patch.notes : prev.notes, updated: now().toISOString() };
  const next: Ratings = { ...ratings, [trackId]: { ...(ratings[trackId] ?? {}), [versionId]: entry } };
  let saved = false;
  if (store) {
    try {
      store.setItem(RATINGS_KEY, JSON.stringify(next));
      saved = true;
    } catch {
      saved = false;
    }
  }
  return { ratings: next, saved };
}

export interface ExportedRatings {
  app: "Earth Jukebox";
  exported: string;
  ratings: {
    track: string;
    version: string;
    label: string;
    rating: number | null;
    notes: string;
    updated: string;
    r2: number | null;
    control_r2: number | null;
  }[];
}

/** Flatten ratings for download, with each version's label and fidelity next to the judgement. */
export function exportRatings(ratings: Ratings, catalogue: Record<string, Version[]>, now: () => Date = () => new Date()): ExportedRatings {
  const rows: ExportedRatings["ratings"] = [];
  for (const [track, byVersion] of Object.entries(ratings)) {
    for (const [version, r] of Object.entries(byVersion)) {
      if (r.rating == null && !r.notes.trim()) continue;
      const v = catalogue[track]?.find((x) => x.id === version);
      rows.push({
        track,
        version,
        label: v?.label ?? version,
        rating: r.rating,
        notes: r.notes,
        updated: r.updated,
        r2: v?.fidelity.r2 ?? null,
        control_r2: v?.fidelity.control_r2 ?? null,
      });
    }
  }
  rows.sort((a, b) => a.track.localeCompare(b.track) || a.version.localeCompare(b.version));
  return { app: "Earth Jukebox", exported: now().toISOString(), ratings: rows };
}
