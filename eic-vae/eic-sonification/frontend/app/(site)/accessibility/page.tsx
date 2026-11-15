import type { Metadata } from "next";
import Link from "next/link";
import { DATA_KEYS } from "@/lib/data/keys";

export const metadata: Metadata = {
  title: "Accessibility",
  description: "Keyboard controls, screen reader support and written descriptions in Earth Jukebox.",
};

const COMPARE_KEYS: [string, string][] = [
  ["1 to 6", "Switch version at the same moment"],
  ["Space", "Play or pause"],
  ["← →", "Skip 5 seconds back or forward"],
  ["Home", "Back to the start"],
];

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

      <h2 className="jb-how-sub">On the data tracks</h2>
      <p className="jb-how-lede">
        The eight data tracks were built to be read by ear. Every melody note is a value you can ask for, step through and check.
      </p>
      <ul className="jb-list">
        <li>
          <strong>Ask for the value.</strong> Press V for the date, the value and whether it is rising or falling. Shift with the arrow keys steps
          from note to note, plays it and says it. Where the data has dates, years or sections, they are spoken instead of the playing time.
        </li>
        <li>
          <strong>Announcements at your pace.</strong> While a track plays, a polite message gives the date and value every 5 seconds, every 10, at
          each melody note, or not at all. They can also be spoken aloud with the browser voice.
        </li>
        <li>
          <strong>A legend you can hear.</strong> Press L: each voice is described, then calibration tones play the lowest, middle and highest value.
        </li>
        <li>
          <strong>Hear the shape.</strong> S plays every latitude band at this moment, north to south, from left to right. Shift and S plays the whole
          melody in about eight seconds.
        </li>
        <li>
          <strong>Choose what you hear.</strong> Seven groups (melody, bands, events, harmony and the three recorded voices) each have a switch and a
          volume, and keys 1 to 7. P flips the polarity, so higher values play lower.
        </li>
        <li>
          <strong>Headphone 3D.</strong> H places Earth sounds to the left, human sounds to the right, living sounds above and the data melody in
          front, and moves them with the data.
        </li>
        <li>
          <strong>The score in words.</strong> The picture of the score is hidden from screen readers. Beside it is a written summary (start, end,
          lowest, highest, trend) and a table of every melody note with its value.
        </li>
        <li>
          <strong>Visible focus, less motion, forced colours.</strong> Every control shows a vermilion focus ring. When your system asks for less
          motion, the video starts switched off. In Windows high contrast, group chips keep their outline and number.
        </li>
      </ul>

      <h2 className="jb-how-sub">Keyboard shortcuts on the data tracks</h2>
      <dl className="jb-keys-table">
        {DATA_KEYS.map(([k, v]) => (
          <div key={k}>
            <dt>
              <kbd>{k}</kbd>
            </dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>

      <h2 className="jb-how-sub">On the Compare page</h2>
      <dl className="jb-keys-table">
        {COMPARE_KEYS.map(([k, v]) => (
          <div key={k}>
            <dt>
              <kbd>{k}</kbd>
            </dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <p className="jb-how-lede">
        Ratings use ordinary radio buttons and a text box, so they work with any screen reader. They stay in this browser until you export them.
      </p>

      <p className="jb-next">
        <Link href="/">Back to the player</Link> · <Link href="/credits">Credits</Link>
      </p>
    </div>
  );
}
