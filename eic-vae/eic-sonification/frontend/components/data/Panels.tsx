"use client";

import { memo, useEffect, useRef } from "react";
import { GROUP_LABELS, GROUP_LINES, fmtNum, fmtValue, noteName, shortUnits } from "@/lib/data/music";
import { describeFidelity } from "@/lib/data/fidelity";
import { fmtTime, scoreSummary } from "@/lib/data/time";
import { DATA_KEYS } from "@/lib/data/keys";
import { GROUPS, type DataTrack, type Group } from "@/lib/data/types";
import type { GroupState } from "@/lib/data/engine";
import type { Prepared } from "@/lib/data/load";

/** Stem groups as channels, in the same row language as the style list. Key 1–7 mutes a group. */
export const StemMixer = memo(function StemMixer({
  track,
  present,
  state,
  onChange,
}: {
  track: DataTrack;
  present: Group[];
  state: Record<Group, GroupState>;
  onChange: (g: Group, patch: Partial<GroupState>) => void;
}) {
  const absent = GROUPS.filter((g) => !present.includes(g)).map((g) => GROUP_LABELS[g]);
  return (
    <div className="jb-styles jb-mixer" role="group" aria-labelledby="mixer-h">
      <h2 id="mixer-h" className="jb-h2 jb-mixer-h">
        What you are hearing
      </h2>
      {present.map((g) => {
        const n = GROUPS.indexOf(g) + 1;
        const st = state[g];
        const stems = track.stems.filter((s) => s.group === g).map((s) => s.label || s.name);
        return (
          <div key={g} className={`jb-group-row${st.muted ? " off" : ""}`}>
            <span className={`jb-gchip g-${g}`} aria-hidden="true">
              {n}
            </span>
            <label className="jb-group-name">
              <input
                type="checkbox"
                checked={!st.muted}
                onChange={(e) => onChange(g, { muted: !e.target.checked })}
                aria-keyshortcuts={String(n)}
                aria-describedby={`line-${g}`}
              />
              <span>{GROUP_LABELS[g]}</span>
            </label>
            <span className="jb-group-line" id={`line-${g}`}>
              {GROUP_LINES[g]}
              {stems.length > 1 ? <span className="jb-sr"> Stems: {stems.join(", ")}.</span> : null}
            </span>
            <input
              className="jb-group-vol"
              type="range"
              min={0}
              max={1.5}
              step={0.05}
              value={st.volume}
              onChange={(e) => onChange(g, { volume: Number(e.target.value) })}
              aria-label={`${GROUP_LABELS[g]} volume`}
              aria-valuetext={`${Math.round(st.volume * 100)} percent`}
            />
            <kbd aria-hidden="true">{n}</kbd>
          </div>
        );
      })}
      {absent.length ? <p className="jb-fine jb-mixer-absent">Not in this track: {absent.join(", ")}.</p> : null}
    </div>
  );
});

export const FidelityPanel = memo(function FidelityPanel({ track }: { track: DataTrack }) {
  const f = describeFidelity(track);
  return (
    <section className="jb-panel" aria-labelledby="fid-h">
      <h2 id="fid-h" className="jb-h2">
        Does the music follow the data?
      </h2>
      <p className="jb-fid-head">{f.headline}</p>
      {f.bars.length ? (
        <div className="jb-fid" aria-hidden="true">
          {f.bars.map((b) => (
            <div key={b.label} className="jb-fid-row">
              <span>{b.label}</span>
              <span className="jb-fid-track">
                <span className={`jb-fid-fill${b.kind === "control" ? " ctrl" : ""}`} style={{ width: `${Math.max(0, b.r2) * 100}%` }} />
              </span>
              <span className="jb-code">{fmtNum(b.r2)}</span>
            </div>
          ))}
        </div>
      ) : null}
      {f.detail ? <p className="jb-fine">{f.detail}</p> : null}
    </section>
  );
});

export const LegendPanel = memo(function LegendPanel({ track, prep, onLegend }: { track: DataTrack; prep: Prepared; onLegend: () => void }) {
  const u = shortUnits(track.units);
  return (
    <section className="jb-panel" aria-labelledby="legend-h">
      <h2 id="legend-h" className="jb-h2">
        Legend
      </h2>
      <dl className="jb-legend">
        {track.legend.map((l) => (
          <div key={l.voice}>
            <dt>{l.voice}</dt>
            <dd>{l.meaning}</dd>
          </div>
        ))}
      </dl>
      {prep.pm ? (
        <p className="jb-fine">
          Melody pitch: <span className="jb-code">{noteName(prep.pm.pitch(prep.pm.vmin))}</span> is {fmtValue(prep.pm.vmin, u, prep.range)}, up to{" "}
          <span className="jb-code">{noteName(prep.pm.pitch(prep.pm.vmax))}</span> at {fmtValue(prep.pm.vmax, u, prep.range)}.
        </p>
      ) : null}
      <button type="button" className="jb-chip-btn" onClick={onLegend} aria-keyshortcuts="L">
        Hear the legend <kbd>L</kbd>
      </button>
    </section>
  );
});

/** The score picture (decorative canvas) with its text alternative and the full table of melody notes. */
export function ScorePanel({
  track,
  prep,
  canvasRef,
  onSeek,
}: {
  track: DataTrack;
  prep: Prepared;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  onSeek: (t: number) => void;
}) {
  const u = shortUnits(track.units);
  return (
    <section className="jb-panel jb-score" aria-labelledby="score-h">
      <h2 id="score-h" className="jb-h2">
        The score
      </h2>
      <canvas
        ref={canvasRef}
        className="jb-roll"
        aria-hidden="true"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          onSeek(((e.clientX - r.left) / r.width) * track.duration_s);
        }}
      />
      <p className="jb-roll-text">{scoreSummary(track, prep)}</p>
      {prep.lead.length ? (
        <details className="jb-details">
          <summary>Every melody note as a table ({prep.lead.length} notes)</summary>
          <div className="jb-table-wrap" tabIndex={0} role="region" aria-label="Melody notes table">
            <table className="jb-table">
              <caption className="jb-sr">Melody notes with the data value each one plays</caption>
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">Time</th>
                  <th scope="col">Value{u ? ` (${u})` : ""}</th>
                  <th scope="col">Note</th>
                </tr>
              </thead>
              <tbody>
                {prep.lead.map((n, i) => (
                  <tr key={`${n.start}-${i}`}>
                    <td className="jb-code">{i + 1}</td>
                    <td className="jb-code">{fmtTime(n.start)}</td>
                    <td className="jb-code">{fmtValue(n.value, "", prep.range)}</td>
                    <td className="jb-code">{noteName(n.pitch)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      ) : null}
    </section>
  );
}

export function HelpDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      if (typeof d.showModal === "function") d.showModal();
      else d.setAttribute("open", "");
    } else if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog ref={ref} className="jb-help" aria-labelledby="help-h" onClose={onClose}>
      <h2 id="help-h" className="jb-h2">
        Keyboard shortcuts
      </h2>
      <p className="jb-fine">
        They work anywhere on the player except inside text fields. With a screen reader, switch to focus mode first, or use the buttons.
      </p>
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
      <form method="dialog">
        <button type="submit" className="jb-skip jb-help-close">
          Close
        </button>
      </form>
    </dialog>
  );
}
