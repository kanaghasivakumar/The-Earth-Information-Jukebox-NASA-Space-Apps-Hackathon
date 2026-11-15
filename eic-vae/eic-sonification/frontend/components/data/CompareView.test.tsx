// @vitest-environment jsdom
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { CompareView } from "./CompareView";
import { RATINGS_KEY } from "@/lib/data/compare";

const PUBLIC = path.resolve(__dirname, "../../public");

beforeEach(() => {
  window.localStorage.clear();
  window.history.replaceState(null, "", "/compare?track=5209");
  // no Web Audio in jsdom: the view lists versions and keeps ratings, playback stays off
  vi.stubGlobal("fetch", async (url: string) => {
    try {
      const text = await readFile(path.join(PUBLIC, decodeURIComponent(url)), "utf8");
      return { ok: true, status: 200, json: async () => JSON.parse(text) };
    } catch {
      return { ok: false, status: 404, json: async () => ({}) };
    }
  });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const hasBplus = (id: string) => existsSync(path.join(PUBLIC, "jukebox/tracks", id, "versions/mcl_bplus.mp3"));
const N_5209 = hasBplus("5209") ? 7 : 6;
const versionRadios = () => screen.getAllByRole("radio").filter((r) => r.id.startsWith("ver-"));

test("lists every version of 5209 and switches with a click or a digit key", async () => {
  render(<CompareView />);
  await waitFor(() => expect(versionRadios()).toHaveLength(N_5209));
  expect(versionRadios()[0]).toHaveProperty("ariaChecked", "true");

  fireEvent.click(versionRadios()[3]);
  expect(versionRadios()[3].getAttribute("aria-checked")).toBe("true");
  expect(versionRadios()[0].getAttribute("aria-checked")).toBe("false");
  expect(screen.getByRole("heading", { name: /Your judgement: MuseControlLite, mode B \(replace\)/ })).toBeTruthy();

  act(() => {
    fireEvent.keyDown(window, { key: "5" });
  });
  expect(versionRadios()[4].getAttribute("aria-checked")).toBe("true");
  expect(screen.getByRole("heading", { name: /Mood layer/ })).toBeTruthy();
});

test("a rating and a note are stored per track and version", async () => {
  render(<CompareView />);
  await waitFor(() => expect(versionRadios()).toHaveLength(N_5209));
  fireEvent.click(versionRadios()[1]);
  const four = screen.getAllByRole("radio").find((r) => (r as HTMLInputElement).type === "radio" && r.parentElement?.textContent === "4")!;
  fireEvent.click(four);
  const notes = screen.getByRole("textbox", { name: /Notes/ });
  fireEvent.change(notes, { target: { value: "richer, flute still on top" } });
  fireEvent.blur(notes);
  const saved = JSON.parse(window.localStorage.getItem(RATINGS_KEY)!);
  expect(saved["5209"].mcl_a.rating).toBe(4);
  expect(saved["5209"].mcl_a.notes).toBe("richer, flute still on top");
  expect(screen.getByText(/rated 4\/5/)).toBeTruthy();
  expect(screen.getByRole("button", { name: /Export ratings \(1\)/ })).toBeTruthy();
});

test("the other tracks offer the mix, B+ where rendered, and its shuffled control", async () => {
  window.history.replaceState(null, "", "/compare?track=5176");
  render(<CompareView />);
  const names = ["Deterministic mix", ...(hasBplus("5176") ? ["MuseControlLite B+ (no cymbals, with Earth voices)"] : []), "Shuffled control"];
  await waitFor(() => expect(versionRadios()).toHaveLength(names.length));
  expect(versionRadios().map((r) => r.querySelector(".jb-style-name")?.firstChild?.textContent)).toEqual(names);
});
