// @vitest-environment jsdom
import { Profiler } from "react";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { StyleList } from "./StyleList";
import { createTimeStore } from "@/lib/timeStore";
import type { Track } from "@/lib/jukebox";

afterEach(cleanup);

const mk = (id: string, tempo: number): Track => ({
  id, tempo, scale: "major", duration: 100, audio: `/${id}.mp3`, midi: `/${id}.mid`, instruments: [], loudness: [], regions: [],
});
const tracks = [mk("ambient", 78), mk("cinematic", 90)];

test("the list does not re-render when time changes", () => {
  const spy = vi.fn();
  const store = createTimeStore();
  render(
    <Profiler id="s" onRender={spy}>
      <StyleList tracks={tracks} trackId="ambient" onChoose={() => {}} />
    </Profiler>,
  );
  act(() => {
    for (let i = 1; i <= 20; i++) store.set(i);
  });
  expect(spy).toHaveBeenCalledTimes(1);
});

test("clicking a style calls onChoose with its id", () => {
  const onChoose = vi.fn();
  render(<StyleList tracks={tracks} trackId="ambient" onChoose={onChoose} />);
  fireEvent.click(screen.getAllByRole("radio")[1]);
  expect(onChoose).toHaveBeenCalledWith("cinematic");
});
