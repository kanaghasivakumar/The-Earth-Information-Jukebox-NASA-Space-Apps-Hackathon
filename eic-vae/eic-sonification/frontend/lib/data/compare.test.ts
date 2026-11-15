import { describe, expect, test } from "vitest";
import { RATINGS_KEY, exportRatings, fidelityLine, loadRatings, setRating, switchVersion, type Ratings } from "./compare";
import { versionGains } from "./engine";
import type { Version } from "./types";

const v = (id: string, kind: Version["kind"], r2: number | null, control: number | null = null): Version => ({
  id,
  label: id.toUpperCase(),
  kind,
  file: `${id}.mp3`,
  line: "",
  fidelity: { r2, within_half_semitone: r2 == null ? null : 1, control_r2: control, control_kind: control == null ? null : "x", note: null },
});
const VERSIONS = [v("mix", "deterministic", 0.966, -0.55), v("mcl_a", "learned", 0.964, -0.92), v("odd", "learned", -0.55)];

class MemoryStorage {
  data = new Map<string, string>();
  getItem(k: string) {
    return this.data.get(k) ?? null;
  }
  setItem(k: string, val: string) {
    this.data.set(k, val);
  }
}
const blocked = {
  getItem: () => {
    throw new Error("SecurityError");
  },
  setItem: () => {
    throw new Error("QuotaExceededError");
  },
};
const fixed = () => new Date("2026-11-15T12:00:00Z");

describe("switching versions", () => {
  test("keeps the position and gives the chosen version all the gain", () => {
    const r = switchVersion(VERSIONS, 0, 1, 12.34);
    expect(r.selected).toBe(1);
    expect(r.position).toBe(12.34);
    expect(r.gains).toEqual([0, 1, 0]);
    expect(r.announcement).toBe("MCL_A, R squared 0.96. Same moment.");
  });
  test("an out-of-range choice keeps the current version", () => {
    expect(switchVersion(VERSIONS, 2, 7, 3).selected).toBe(2);
    expect(switchVersion(VERSIONS, 2, -1, 3).gains).toEqual([0, 0, 1]);
  });
  test("speaks a negative R² as minus", () => {
    expect(switchVersion(VERSIONS, 0, 2, 0).announcement).toBe("ODD, R squared minus 0.55. Same moment.");
  });
  test("versionGains is one-hot", () => {
    expect(versionGains(4, 3)).toEqual([0, 0, 0, 1]);
  });
  test("fidelity lines", () => {
    // the shuffled-control number is not shown on the site any more, only the real-data fidelity
    expect(fidelityLine(VERSIONS[0])).toBe("R² 0.966 · 100% of notes within ½ semitone");
    expect(fidelityLine(v("x", "learned", null))).toBe("No fidelity number");
  });
});

describe("ratings in localStorage", () => {
  test("round trip", () => {
    const store = new MemoryStorage();
    let r: Ratings = loadRatings(store);
    expect(r).toEqual({});
    ({ ratings: r } = setRating(r, "5209", "mix", { rating: 4 }, store, fixed));
    ({ ratings: r } = setRating(r, "5209", "mix", { notes: "flute clear" }, store, fixed));
    expect(loadRatings(store)).toEqual({ "5209": { mix: { rating: 4, notes: "flute clear", updated: "2026-11-15T12:00:00.000Z" } } });
    expect(JSON.parse(store.getItem(RATINGS_KEY)!)["5209"].mix.rating).toBe(4);
  });
  test("ratings are clamped to 1–5 and can be cleared", () => {
    const store = new MemoryStorage();
    let { ratings } = setRating({}, "3912", "mood", { rating: 9 }, store, fixed);
    expect(ratings["3912"].mood.rating).toBe(5);
    ({ ratings } = setRating(ratings, "3912", "mood", { rating: null }, store, fixed));
    expect(ratings["3912"].mood.rating).toBeNull();
  });
  test("blocked or full storage never throws; the in-memory copy still updates", () => {
    expect(loadRatings(blocked)).toEqual({});
    const r = setRating({}, "5313", "mcl_a", { rating: 2 }, blocked, fixed);
    expect(r.saved).toBe(false);
    expect(r.ratings["5313"].mcl_a.rating).toBe(2);
    expect(loadRatings(null)).toEqual({});
  });
  test("corrupt JSON reads as no ratings", () => {
    const store = new MemoryStorage();
    store.setItem(RATINGS_KEY, "{not json");
    expect(loadRatings(store)).toEqual({});
    store.setItem(RATINGS_KEY, "[1,2]");
    expect(loadRatings(store)).toEqual({});
  });
  test("export flattens ratings with labels and fidelity, skipping empty entries; an old control rating keeps its id", () => {
    const ratings: Ratings = {
      "5209": {
        mix: { rating: 4, notes: "", updated: "a" },
        mcl_a: { rating: null, notes: "  ", updated: "b" },
        control: { rating: null, notes: "sounds random", updated: "c" },
      },
    };
    const out = exportRatings(ratings, { "5209": VERSIONS }, fixed);
    expect(out.exported).toBe("2026-11-15T12:00:00.000Z");
    expect(out.ratings.map((r) => [r.version, r.label, r.rating, r.r2])).toEqual([
      ["control", "control", null, null],
      ["mix", "MIX", 4, 0.966],
    ]);
  });
});
