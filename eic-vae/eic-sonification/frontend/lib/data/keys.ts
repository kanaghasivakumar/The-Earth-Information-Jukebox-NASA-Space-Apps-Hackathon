/** Keyboard shortcuts for the data player and the Compare view, as pure functions so they can be tested. */
import { GROUPS, type Group } from "./types";

export type DataAction =
  | { type: "toggle" }
  | { type: "seek"; by: number }
  | { type: "step"; dir: 1 | -1 }
  | { type: "home" }
  | { type: "legend" }
  | { type: "value" }
  | { type: "sweep" }
  | { type: "overview" }
  | { type: "polarity" }
  | { type: "ab" }
  | { type: "hrtf" }
  | { type: "mute"; group: Group }
  | { type: "muteAll" }
  | { type: "help" }
  | { type: "stop" };

export interface KeyInput {
  key: string;
  shiftKey?: boolean;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
  target?: { tagName?: string; type?: string; role?: string | null } | null;
}

/** Describe a keydown target for the rules below; window and document have no tag. */
export function describeTarget(t: EventTarget | null): KeyInput["target"] {
  const el = t as (Partial<Element> & { type?: string }) | null;
  if (!el || typeof el.getAttribute !== "function") return null;
  return { tagName: el.tagName, type: el.type, role: el.getAttribute("role") };
}

/** A DOM KeyboardEvent as a KeyInput. */
export const fromEvent = (e: KeyboardEvent): KeyInput => ({
  key: e.key,
  shiftKey: e.shiftKey,
  ctrlKey: e.ctrlKey,
  metaKey: e.metaKey,
  altKey: e.altKey,
  target: describeTarget(e.target),
});

const NAV = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"];

/** Shared rules: when should a page-level shortcut stay out of the way of the focused control? */
function yields(e: KeyInput): boolean {
  const t = e.target ?? {};
  const tag = (t.tagName ?? "").toUpperCase();
  const type = (t.type ?? "").toLowerCase();
  const role = t.role ?? "";
  if (tag === "SELECT" || tag === "TEXTAREA") return true;
  if (tag === "INPUT" && !["checkbox", "range", "radio"].includes(type)) return true;
  // sliders and radio groups keep their own arrow keys
  if ((type === "range" || role === "slider" || role === "radio" || type === "radio") && NAV.includes(e.key)) return true;
  // Space and Enter activate the focused control
  const activates = ["BUTTON", "SUMMARY", "A"].includes(tag) || type === "checkbox" || type === "radio" || role === "radio";
  if ((e.key === " " || e.key === "Enter") && activates) return true;
  // the timeline slider handles Space itself
  if (role === "slider" && e.key === " ") return true;
  return false;
}

/**
 * Space play/pause, ←/→ 5 s, Shift+←/→ previous/next lead note, Home start, L legend, V value,
 * S sweep, Shift+S overview, P polarity, A real/shuffled, H headphone 3D, 1–7 mute a group, M mute all,
 * ? help, Esc stop speech, sweep or overview.
 */
export function keyAction(e: KeyInput, { helpOpen = false } = {}): DataAction | null {
  if (e.ctrlKey || e.metaKey || e.altKey) return null;
  if (e.key === "?") return { type: "help" };
  if (helpOpen) return null;
  if (yields(e)) return null;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  switch (k) {
    case " ":
      return { type: "toggle" };
    case "ArrowLeft":
      return e.shiftKey ? { type: "step", dir: -1 } : { type: "seek", by: -5 };
    case "ArrowRight":
      return e.shiftKey ? { type: "step", dir: 1 } : { type: "seek", by: 5 };
    case "Home":
      return { type: "home" };
    case "l":
      return { type: "legend" };
    case "v":
      return { type: "value" };
    case "s":
      return e.shiftKey ? { type: "overview" } : { type: "sweep" };
    case "p":
      return { type: "polarity" };
    case "a":
      return { type: "ab" };
    case "h":
      return { type: "hrtf" };
    case "m":
      return { type: "muteAll" };
    case "Escape":
      return { type: "stop" };
  }
  if (/^[1-7]$/.test(e.key)) return { type: "mute", group: GROUPS[Number(e.key) - 1] };
  return null;
}

export type CompareAction = { type: "toggle" } | { type: "seek"; by: number } | { type: "home" } | { type: "select"; index: number };

/** Compare view: Space play/pause, ←/→ 5 s, Home start, 1–9 choose a version (same moment). */
export function compareKeyAction(e: KeyInput, count: number): CompareAction | null {
  if (e.ctrlKey || e.metaKey || e.altKey) return null;
  if (yields(e)) return null;
  if (e.key === " ") return { type: "toggle" };
  if (e.key === "ArrowLeft") return { type: "seek", by: -5 };
  if (e.key === "ArrowRight") return { type: "seek", by: 5 };
  if (e.key === "Home") return { type: "home" };
  // digits on a native radio (the 1–5 rating) belong to that control, not to the version list
  if ((e.target?.type ?? "").toLowerCase() === "radio") return null;
  const n = Number(e.key);
  if (/^[1-9]$/.test(e.key) && n <= count) return { type: "select", index: n - 1 };
  return null;
}

/** The shortcut list, shown in the help dialog and on the Accessibility page. */
export const DATA_KEYS: [string, string][] = [
  ["Space", "Play or pause"],
  ["← →", "Skip 5 seconds back or forward"],
  ["Shift + ← →", "Step to the previous or next melody note, hear it and its value"],
  ["Home", "Back to the start"],
  ["L", "Spoken legend, then calibration tones at the lowest, middle and highest value"],
  ["V", "Say the date or time, the value and its trend"],
  ["S", "Sweep: every band at this moment, north to south, panned left to right"],
  ["Shift + S", "Overview: the whole melody in about 8 seconds"],
  ["P", "Flip polarity: higher values play lower (the melody is synthesized live)"],
  ["A", "Switch between the real data and the shuffled control"],
  ["H", "Headphone 3D sound (binaural)"],
  ["1 to 7", "Mute a group: 1 melody, 2 bands, 3 events, 4 harmony, 5 geophony, 6 anthrophony, 7 biophony"],
  ["M", "Mute or unmute everything"],
  ["Esc", "Stop speech, sweep or overview"],
  ["?", "Open or close the shortcut list"],
];
