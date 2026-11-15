// @vitest-environment jsdom
import { readFile } from "node:fs/promises";
import path from "node:path";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { ListenSwitch } from "./ListenSwitch";
import { getSeason } from "@/lib/season";
import { SourceProvider } from "@/components/jukebox/SourceProvider";

const PUBLIC = path.resolve(__dirname, "../../public");

beforeEach(() => {
  vi.stubGlobal("fetch", async (url: string) => {
    try {
      const text = await readFile(path.join(PUBLIC, decodeURIComponent(url)), "utf8");
      return { ok: true, status: 200, json: async () => JSON.parse(text) };
    } catch {
      return { ok: false, status: 404, json: async () => ({}) };
    }
  });
  vi.stubGlobal("matchMedia", (q: string) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {} }));
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.history.replaceState(null, "", "/");
});

test("a data track: title bar with the hamburger, big footage with the local video, score strip under it, controls beside it", async () => {
  window.history.replaceState(null, "", "/?track=5313");
  const { container } = render(<ListenSwitch season={getSeason()} />);
  expect(screen.getByRole("heading", { level: 1, name: "Night lights" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Tracks" }).getAttribute("aria-expanded")).toBe("false");
  await waitFor(() => expect(container.querySelector(".jb-deck .jb-deck-main .jb-frame video")).toBeTruthy());
  const main = container.querySelector(".jb-deck-main")!;
  // order: footage first, then the score strip, then the score panel
  await waitFor(() => expect([...main.children].map((c) => c.className)).toEqual(["jb-video jb-frame", "jb-strip", "jb-panel jb-score"]));
  // "About this track" is a visible section above the video, not a disclosure in the side column
  const about = container.querySelector("section.jb-about")!;
  expect(about).toBeTruthy();
  expect(about.compareDocumentPosition(container.querySelector(".jb-deck")!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect([...container.querySelectorAll(".jb-side summary")].map((s) => s.textContent)).not.toContain("About this track");
  // the "Does the music follow the data?" panel is gone
  expect(container.querySelector("#fid-h")).toBeNull();
  expect(container.querySelector(".jb-deck > .jb-side")).toBeTruthy();
  // the web copy plays first, NASA's own file stays as a fallback
  const srcs = [...container.querySelectorAll(".jb-frame source")].map((s) => s.getAttribute("src"));
  expect(srcs[0]).toBe("/jukebox/tracks/5313/video.mp4");
  expect(srcs[1]).toMatch(/^https:\/\/svs\.gsfc\.nasa\.gov\//);
  // clicking the footage opens the SVS page in a new tab
  const svs = screen.getByRole("link", { name: /Open NASA SVS page for .* \(new tab\)/ });
  expect(svs.getAttribute("href")).toBe("https://svs.gsfc.nasa.gov/5313");
  expect(svs.getAttribute("target")).toBe("_blank");
  await waitFor(() => expect(screen.getByRole("group", { name: "Sound groups" })).toBeTruthy());
  expect(screen.getByRole("slider", { name: "Playback position on the score" })).toBeTruthy();
  // the shuffled control is gone
  expect(screen.queryByRole("radio", { name: /Shuffled control/ })).toBeNull();
  expect(container.textContent).not.toMatch(/shuffled control/i);
});

test("a video that cannot play shows a message instead of a blank box, and the SVS link stays", async () => {
  window.history.replaceState(null, "", "/?track=5209");
  const { container } = render(<ListenSwitch season={getSeason()} />);
  await waitFor(() => expect(container.querySelector(".jb-frame video")).toBeTruthy());
  const sources = container.querySelectorAll(".jb-frame source");
  fireEvent.error(sources[sources.length - 1]);
  expect(container.querySelector(".jb-frame video")).toBeNull();
  expect(screen.getByRole("note").textContent).toMatch(/could not play/);
  expect(screen.getByRole("link", { name: /Open NASA SVS page/ })).toBeTruthy();
});

test("5669 shows the still image with its sweep line in the same frame", async () => {
  window.history.replaceState(null, "", "/?track=5669");
  const { container } = render(<ListenSwitch season={getSeason()} />);
  await waitFor(() => expect(container.querySelector(".jb-frame .jb-still img")).toBeTruthy());
  expect(container.querySelector(".jb-frame .jb-sweep-line")).toBeTruthy();
});

test("switching from the menu updates ?track= and the title; the hurricane season stays reachable", async () => {
  window.history.replaceState(null, "", "/?track=5209");
  render(
    <SourceProvider>
      <ListenSwitch season={getSeason()} />
    </SourceProvider>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Tracks" }));
  fireEvent.click(screen.getByRole("link", { name: /Florida after Ian/ }));
  expect(window.location.search).toBe("?track=5058");
  expect(screen.getByRole("heading", { level: 1, name: "Florida after Ian" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Tracks" }));
  fireEvent.click(screen.getByRole("link", { name: /Hurricane season/ }));
  expect(window.location.search).toBe("");
  expect(screen.getByRole("heading", { level: 1, name: "Hear a hurricane season." })).toBeTruthy();
  expect(screen.getByRole("link", { name: /Open NASA SVS page for the 2022 hurricane season/ }).getAttribute("href")).toBe("https://svs.gsfc.nasa.gov/5097/");
});
