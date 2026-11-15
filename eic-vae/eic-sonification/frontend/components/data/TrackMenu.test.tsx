// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { TrackMenu, listenItems } from "./TrackMenu";

afterEach(cleanup);

const DATA = ["5209", "5313", "3912", "5011", "4950", "5669", "5058", "5176", "5046"];
const button = () => screen.getByRole("button", { name: "Tracks" });
const nav = () => document.getElementById(button().getAttribute("aria-controls")!)!;

test("lists the hurricane season and all nine data tracks as ?track= links", () => {
  const items = listenItems();
  expect(items.map((i) => i.id)).toEqual(["season", ...DATA]);
  expect(items[0].href).toBe("/");
  expect(items[1].href).toBe("/?track=5209");
});

test("the button opens the menu, moves focus to the current track, and marks it with aria-current", () => {
  render(<TrackMenu items={listenItems()} current="5011" onChoose={() => {}} />);
  expect(button().getAttribute("aria-expanded")).toBe("false");
  expect(nav().hidden).toBe(true);
  fireEvent.click(button());
  expect(button().getAttribute("aria-expanded")).toBe("true");
  expect(nav().hidden).toBe(false);
  const links = screen.getAllByRole("link");
  expect(links).toHaveLength(10);
  const current = links.filter((a) => a.getAttribute("aria-current") === "page");
  expect(current).toHaveLength(1);
  expect(current[0].textContent).toMatch(/Lightning from the ISS/);
  expect(document.activeElement).toBe(current[0]);
});

test("arrow keys, Home and End move between tracks; Esc closes and returns focus to the button", () => {
  const outside = vi.fn();
  window.addEventListener("keydown", outside);
  render(<TrackMenu items={listenItems()} current="season" onChoose={() => {}} />);
  fireEvent.click(button());
  const links = screen.getAllByRole("link");
  expect(document.activeElement).toBe(links[0]);
  fireEvent.keyDown(document.activeElement!, { key: "ArrowDown" });
  expect(document.activeElement).toBe(links[1]);
  fireEvent.keyDown(document.activeElement!, { key: "ArrowUp" });
  fireEvent.keyDown(document.activeElement!, { key: "ArrowUp" });
  expect(document.activeElement).toBe(links[9]);
  fireEvent.keyDown(document.activeElement!, { key: "Home" });
  expect(document.activeElement).toBe(links[0]);
  fireEvent.keyDown(document.activeElement!, { key: "End" });
  expect(document.activeElement).toBe(links[9]);
  fireEvent.keyDown(document.activeElement!, { key: "Escape" });
  expect(button().getAttribute("aria-expanded")).toBe("false");
  expect(document.activeElement).toBe(button());
  // the player's own shortcuts (Home, Esc) never saw those keys
  expect(outside).not.toHaveBeenCalled();
  window.removeEventListener("keydown", outside);
});

test("choosing a track calls onChoose with its id, closes the menu and refocuses the button", () => {
  const onChoose = vi.fn();
  render(<TrackMenu items={listenItems()} current="season" onChoose={onChoose} />);
  fireEvent.click(button());
  fireEvent.click(screen.getByRole("link", { name: /Arctic sea ice/ }));
  expect(onChoose).toHaveBeenCalledWith("5046");
  expect(button().getAttribute("aria-expanded")).toBe("false");
  expect(document.activeElement).toBe(button());
});

test("a click outside closes it", () => {
  render(<TrackMenu items={listenItems()} current="season" onChoose={() => {}} />);
  fireEvent.click(button());
  fireEvent.pointerDown(document.body);
  expect(button().getAttribute("aria-expanded")).toBe("false");
});
