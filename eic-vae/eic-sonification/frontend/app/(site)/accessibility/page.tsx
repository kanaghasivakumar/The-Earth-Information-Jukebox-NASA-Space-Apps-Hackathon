import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Accessibility",
  description: "Keyboard controls, screen reader support and written descriptions in Earth Jukebox.",
};

const KEYS: [string, string][] = [
  ["Space", "Play or pause"],
  ["← →", "Skip 5 seconds back or forward"],
  ["Shift + ← →", "Skip 15 seconds"],
  ["1 to 5", "Switch style and keep your place"],
  ["Home, End", "Jump to the start or end (when the timeline is focused)"],
  ["Page Up, Page Down", "Skip 30 seconds (when the timeline is focused)"],
];

export default function AccessibilityPage() {
  return (
    <div className="jb-section jb-page">
      <h1 className="jb-h1">Built to be used without looking.</h1>
      <p className="jb-how-lede">
        Earth science is mostly shown as pictures. Earth Jukebox starts from the sound, and treats the picture as optional.
      </p>

      <h2 className="jb-how-sub">What you get</h2>
      <ul className="jb-list">
        <li>
          <strong>Full keyboard control.</strong> Everything on the player works without a mouse.
        </li>
        <li>
          <strong>Screen reader announcements.</strong> When the picture moves into a new pattern group, or the scene changes, a
          short polite message says so. You can turn these off with &ldquo;Announce changes to screen readers&rdquo;.
        </li>
        <li>
          <strong>Written descriptions.</strong> What the footage shows is described in words, kept apart from what the model
          infers.
        </li>
        <li>
          <strong>Sound first.</strong> &ldquo;Sound only&rdquo; hides the video and shows the current pattern group as a large letter.
          The timeline is a labelled slider with spoken values, including the pattern group.
        </li>
        <li>
          <strong>Colour is never the only cue.</strong> Pattern groups carry letters, and animation is reduced when your system
          asks for it.
        </li>
      </ul>

      <h2 className="jb-how-sub">Keyboard shortcuts</h2>
      <dl className="jb-keys-table">
        {KEYS.map(([k, v]) => (
          <div key={k}>
            <dt>
              <kbd>{k}</kbd>
            </dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <p className="jb-next">
        <Link href="/">Back to the player</Link>
      </p>
    </div>
  );
}
