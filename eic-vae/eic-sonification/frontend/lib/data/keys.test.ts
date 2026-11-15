import { describe, expect, test } from "vitest";
import { DATA_KEYS, compareKeyAction, describeTarget, keyAction, type KeyInput } from "./keys";

const k = (key: string, extra: Partial<KeyInput> = {}) => keyAction({ key, ...extra });
const on = (tagName: string, type?: string, role?: string) => ({ target: { tagName, type, role } });

describe("data player shortcuts", () => {
  test("transport", () => {
    expect(k(" ")).toEqual({ type: "toggle" });
    expect(k("ArrowLeft")).toEqual({ type: "seek", by: -5 });
    expect(k("ArrowRight")).toEqual({ type: "seek", by: 5 });
    expect(k("ArrowLeft", { shiftKey: true })).toEqual({ type: "step", dir: -1 });
    expect(k("ArrowRight", { shiftKey: true })).toEqual({ type: "step", dir: 1 });
    expect(k("Home")).toEqual({ type: "home" });
  });
  test("listening tools, upper or lower case", () => {
    expect(k("l")).toEqual({ type: "legend" });
    expect(k("V")).toEqual({ type: "value" });
    expect(k("s")).toEqual({ type: "sweep" });
    expect(k("S", { shiftKey: true })).toEqual({ type: "overview" });
    expect(k("p")).toEqual({ type: "polarity" });
    expect(k("h")).toEqual({ type: "hrtf" });
    expect(k("m")).toEqual({ type: "muteAll" });
    expect(k("Escape")).toEqual({ type: "stop" });
  });
  test("1 to 7 mute the groups in order", () => {
    expect(["1", "2", "3", "4", "5", "6", "7"].map((d) => (k(d) as { group: string }).group)).toEqual([
      "melody",
      "bands",
      "events",
      "harmony",
      "geophony",
      "anthrophony",
      "biophony",
    ]);
    expect(k("8")).toBeNull();
    expect(k("0")).toBeNull();
  });
  test("? opens help, even while help is open; other keys wait", () => {
    expect(k("?")).toEqual({ type: "help" });
    expect(keyAction({ key: "?" }, { helpOpen: true })).toEqual({ type: "help" });
    expect(keyAction({ key: " " }, { helpOpen: true })).toBeNull();
  });
  test("modifier chords belong to the browser", () => {
    expect(k("l", { ctrlKey: true })).toBeNull();
    expect(k("a", { metaKey: true })).toBeNull();
    expect(k("ArrowLeft", { altKey: true })).toBeNull();
  });
  test("text fields keep every key", () => {
    expect(k("a", on("INPUT", "text"))).toBeNull();
    expect(k(" ", on("TEXTAREA"))).toBeNull();
    expect(k("1", on("SELECT"))).toBeNull();
  });
  test("sliders and radio groups keep their arrows; shortcuts still work there", () => {
    expect(k("ArrowLeft", on("INPUT", "range"))).toBeNull();
    expect(k("ArrowRight", on("DIV", undefined, "slider"))).toBeNull();
    expect(k("ArrowDown", on("BUTTON", "button", "radio"))).toBeNull();
    expect(k("l", on("INPUT", "range"))).toEqual({ type: "legend" });
    expect(k("3", on("INPUT", "checkbox"))).toEqual({ type: "mute", group: "events" });
  });
  test("Space and Enter press the focused button instead of playing", () => {
    expect(k(" ", on("BUTTON", "button"))).toBeNull();
    expect(k("Enter", on("A"))).toBeNull();
    expect(k(" ", on("INPUT", "checkbox"))).toBeNull();
    expect(k(" ", on("DIV", undefined, "slider"))).toBeNull();
    expect(k("v", on("BUTTON", "button"))).toEqual({ type: "value" });
  });
  test("the help list documents every shortcut", () => {
    const keys = DATA_KEYS.map(([key]) => key).join(" ");
    for (const want of ["Space", "Shift + ← →", "L", "V", "S", "P", "H", "1 to 7", "?"]) expect(keys).toContain(want);
  });
  test("A no longer does anything: the shuffled control is gone from the player", () => {
    expect(k("a")).toBeNull();
    expect(k("A", { shiftKey: true })).toBeNull();
    expect(DATA_KEYS.map(([key]) => key)).not.toContain("A");
    expect(DATA_KEYS.map(([, v]) => v).join(" ")).not.toMatch(/shuffl/i);
  });
});

describe("compare view shortcuts", () => {
  test("digits choose a version, up to the number of versions", () => {
    expect(compareKeyAction({ key: "1" }, 6)).toEqual({ type: "select", index: 0 });
    expect(compareKeyAction({ key: "6" }, 6)).toEqual({ type: "select", index: 5 });
    expect(compareKeyAction({ key: "3" }, 2)).toBeNull();
  });
  test("transport", () => {
    expect(compareKeyAction({ key: " " }, 2)).toEqual({ type: "toggle" });
    expect(compareKeyAction({ key: "ArrowLeft" }, 2)).toEqual({ type: "seek", by: -5 });
    expect(compareKeyAction({ key: "Home" }, 2)).toEqual({ type: "home" });
  });
  test("the notes box and the rating radios keep their keys", () => {
    expect(compareKeyAction({ key: "2", ...on("TEXTAREA") }, 6)).toBeNull();
    expect(compareKeyAction({ key: "2", ...on("INPUT", "radio") }, 6)).toBeNull();
    expect(compareKeyAction({ key: " ", ...on("INPUT", "radio") }, 6)).toBeNull();
    // the version rows are role=radio buttons: digits still switch, arrows stay with the group
    expect(compareKeyAction({ key: "2", ...on("BUTTON", "button", "radio") }, 6)).toEqual({ type: "select", index: 1 });
    expect(compareKeyAction({ key: "ArrowLeft", ...on("BUTTON", "button", "radio") }, 6)).toBeNull();
  });
});

test("describeTarget copes with window as the target", () => {
  expect(describeTarget(null)).toBeNull();
  expect(describeTarget({} as EventTarget)).toBeNull();
  expect(describeTarget({ tagName: "INPUT", type: "range", getAttribute: () => null } as unknown as EventTarget)).toEqual({ tagName: "INPUT", type: "range", role: null });
});
