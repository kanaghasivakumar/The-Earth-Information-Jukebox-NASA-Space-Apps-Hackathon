import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { describe, expect, test } from "vitest";
import { groupOf, loadTrack, loadVersions, parseTimeline, parseTrack, prepare, sweepVoices, TrackError } from "./load";
import { describeNow, hasDates, leadIndexAt, scoreSummary, timeLabel } from "./time";
import catalogue from "./catalogue.json";

const PUBLIC = path.resolve(__dirname, "../../public");

/** fetch() stand-in that serves files from public/ */
async function fsFetch(url: string) {
  try {
    const text = await readFile(path.join(PUBLIC, decodeURIComponent(url)), "utf8");
    return { ok: true, status: 200, json: async () => JSON.parse(text) as unknown };
  } catch {
    return { ok: false, status: 404, json: async () => ({}) };
  }
}

const minimal = {
  id: 1,
  title: "t",
  svs_url: "u",
  video_url: "v.mp4",
  duration_s: 10,
  units: "°C",
  description: "d",
  legend: [],
  stems: [{ name: "melody", file: "melody.mp3", group: "melody" }],
  timeline: "timeline.json",
  controls: "controls.json",
  fidelity: { r2: 0.9 },
  data_source: "x",
};

describe("parseTrack", () => {
  test("accepts the contract and makes the id a string", () => {
    expect(parseTrack(minimal).id).toBe("1");
  });
  test("names the first missing field", () => {
    const { units: _units, ...rest } = minimal;
    void _units;
    expect(() => parseTrack(rest)).toThrow(/units/);
  });
  test("needs a video or a still image", () => {
    expect(() => parseTrack({ ...minimal, video_url: null })).toThrow(TrackError);
    expect(parseTrack({ ...minimal, video_url: null, image_url: "i.png" }).image_url).toBe("i.png");
  });
  test("rejects an unknown stem group", () => {
    expect(() => parseTrack({ ...minimal, stems: [{ name: "x", file: "x.mp3", group: "drums" }] })).toThrow(/unknown group/);
  });
});

describe("parseTimeline", () => {
  test("keeps null values (model harmony) and sorts by start", () => {
    const tl = parseTimeline([
      { start: 2, end: 3, pitch: 60, velocity: 80, voice: "lead", value: 1.5, pan: 0 },
      { start: 1, end: 2, pitch: 40, velocity: 60, voice: "harmony", value: null, pan: 0 },
    ]);
    expect(tl.map((n) => n.start)).toEqual([1, 2]);
    expect(tl[0].value).toBeNull();
  });
  test("rejects notes without numbers", () => {
    expect(() => parseTimeline([{ start: "a", end: 1, pitch: 60 }])).toThrow(TrackError);
  });
});

describe("loadTrack on the bundled tracks", () => {
  test.each(catalogue.map((c) => c.id))("%s loads and every stem file exists", async (id) => {
    const lt = await loadTrack(id, fsFetch);
    expect(lt.track.id).toBe(id);
    expect(lt.timeline.length).toBeGreaterThan(0);
    for (const s of lt.track.stems) expect(existsSync(path.join(PUBLIC, "jukebox/tracks", id, s.file))).toBe(true);
    const prep = prepare(lt.track, lt.timeline);
    expect(prep.lead.every((n) => n.value !== null)).toBe(true);
    expect(prep.present.length).toBeGreaterThan(0);
  });

  test("5209: the lead is the global anomaly, dates are years, harmony is not lead", async () => {
    const { track, timeline } = await loadTrack("5209", fsFetch);
    const prep = prepare(track, timeline);
    expect(prep.lead.every((n) => n.voice === "lead global")).toBe(true);
    expect(groupOf(track, "bass")).toBe("harmony");
    expect(hasDates(track)).toBe(true);
    expect(timeLabel(track, 0)).toBe("1880");
    expect(prep.pm?.pitch(prep.pm.vmax)).toBeGreaterThan(prep.pm!.pitch(prep.pm!.vmin));
    expect(describeNow(track, prep, 29.9)).toMatch(/^2023: .*degrees Celsius anomaly\.$/);
    expect(sweepVoices(track, timeline).length).toBe(8);
    expect(scoreSummary(track, prep)).toMatch(/higher on average in the second half/);
  });

  test("5313: the segment name is spoken before the year", async () => {
    const { track } = await loadTrack("5313", fsFetch);
    expect(timeLabel(track, 0.6)).toBe("annual series, 2013");
    expect(timeLabel(track, 12)).toMatch(/^change map, /);
  });

  test("5011: dates come from the fitted calendar and are clamped", async () => {
    const { track } = await loadTrack("5011", fsFetch);
    expect(timeLabel(track, track.calendar!.t_ref)).toBe("August 2017");
    expect(timeLabel(track, 100)).toBe("July 2023");
  });

  test("5669: a still image with no single melody", async () => {
    const { track, timeline } = await loadTrack("5669", fsFetch);
    const prep = prepare(track, timeline);
    expect(track.video_url).toBeNull();
    expect(prep.lead).toHaveLength(0);
    expect(describeNow(track, prep, 0)).toMatch(/longitude 180 degrees west/);
  });

  test("a missing track is a TrackError", async () => {
    await expect(loadTrack("9999", fsFetch)).rejects.toThrow(/HTTP 404/);
  });
});

/** MuseControlLite B+ is offered wherever its mix was exported (demo/<id>_mclbplus). */
const bplus = (id: string) => (existsSync(path.join(PUBLIC, "jukebox/tracks", id, "versions/mcl_bplus.mp3")) ? ["mcl_bplus"] : []);

describe("versions", () => {
  test("the three pilots offer the pilot versions, the others the mix and its control, B+ where rendered", async () => {
    for (const id of ["5209", "5313", "3912"]) {
      const v = await loadVersions(id, fsFetch);
      expect(v.versions.map((x) => x.id)).toEqual(["mix", "mcl_a", "mcl_b_augment", "mcl_b_replace", "mood", ...bplus(id), "control"]);
    }
    expect((await loadVersions("5176", fsFetch)).versions.map((x) => x.id)).toEqual(["mix", ...bplus("5176"), "control"]);
    expect((await loadVersions("5669", fsFetch)).versions.map((x) => x.id)).toEqual(["mix", ...bplus("5669")]);
  });

  test("every version file exists", async () => {
    for (const c of catalogue) {
      for (const v of (await loadVersions(c.id, fsFetch)).versions) {
        expect(existsSync(path.join(PUBLIC, "jukebox/tracks", c.id, v.file)), `${c.id}/${v.file}`).toBe(true);
      }
    }
  });
});

test("leadIndexAt finds the last note that has started", () => {
  const lead = [0, 1, 2].map((s) => ({ start: s, end: s + 1, pitch: 60, velocity: 80, voice: "lead", value: s, pan: 0 }));
  expect(leadIndexAt(lead, -1)).toBe(-1);
  expect(leadIndexAt(lead, 1.5)).toBe(1);
  expect(leadIndexAt(lead, 9)).toBe(2);
});
