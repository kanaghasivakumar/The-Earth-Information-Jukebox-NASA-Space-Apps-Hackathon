"use client";

import { memo, useCallback, useMemo, useRef } from "react";
import { useTime, type TimeStore } from "@/lib/timeStore";
import { fmt } from "@/lib/jukebox";
import { fmtTime, hasDates, leadIndexAt, timeLabel } from "@/lib/data/time";
import { fmtValue, shortUnits } from "@/lib/data/music";
import type { Prepared } from "@/lib/data/load";
import type { Controls, DataTrack } from "@/lib/data/types";

/**
 * The shared timeline for a data track, built from the same parts as the hurricane timeline: lanes on
 * chart paper, one slider layer with the vermilion playhead, axis ticks below.
 */
interface Props {
  track: DataTrack;
  prep: Prepared;
  controls: Controls | null;
  duration: number;
  store: TimeStore;
  onSeek: (seconds: number) => void;
  onToggle: () => void;
}

const W = 1000;
const H = 64;
const VOICES = [
  ["geophony", 5],
  ["anthrophony", 6],
  ["biophony", 7],
] as const;

function areaPath(points: [number, number][]) {
  if (!points.length) return "";
  return `M0,${H} L${points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L")} L${W},${H} Z`;
}

const Lanes = memo(function Lanes({ track, prep, controls, duration }: Omit<Props, "store" | "onSeek" | "onToggle">) {
  const lead = useMemo(() => {
    const { lead: notes, vmin, vmax } = prep;
    if (!notes.length) return "";
    const span = vmax - vmin || 1;
    const pts: [number, number][] = [];
    notes.forEach((n, i) => {
      const y = H - 2 - (((n.value as number) - vmin) / span) * (H - 8);
      const x0 = (n.start / duration) * W;
      const x1 = ((notes[i + 1]?.start ?? duration) / duration) * W;
      pts.push([x0, y], [x1, y]);
    });
    return areaPath(pts);
  }, [prep, duration]);

  const voices = useMemo(() => {
    if (!controls || controls.synthetic) return [];
    return VOICES.filter(([k]) => controls.curves[k]?.some((v) => v > 0.02)).map(([k, key]) => {
      const vals = controls.curves[k]!;
      const d = "M" + controls.t.map((t, j) => `${((t / duration) * W).toFixed(1)},${(H - 2 - vals[j] * (H - 8)).toFixed(1)}`).join(" L");
      return { k, key, d };
    });
  }, [controls, duration]);

  const segments = track.segments ?? [];
  const u = shortUnits(track.units);

  return (
    <>
      {prep.lead.length ? (
        <div className="jb-lane">
          <span className="jb-lane-label" id="lane-value">
            Melody value
            <small>
              The data the melody plays, {fmtValue(prep.vmin, u, prep.range)} to {fmtValue(prep.vmax, u, prep.range)}
            </small>
          </span>
          <svg className="jb-lane-body jb-curve" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-labelledby="lane-value">
            <path d={lead} className="jb-fill" />
            <path d={lead} className="jb-fill jb-played" />
          </svg>
        </div>
      ) : null}
      {voices.length ? (
        <div className="jb-lane">
          <span className="jb-lane-label" id="lane-voices">
            Earth voices
            <small>
              {voices.map((v, i) => (
                <span key={v.k} className="jb-voice-key">
                  {i ? " · " : ""}
                  <i className={`jb-gchip-sm g-${v.k}`} aria-hidden="true">
                    {v.key}
                  </i>{" "}
                  {v.k}
                </span>
              ))}
            </small>
          </span>
          <svg
            className="jb-lane-body jb-curve"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="none"
            role="img"
            aria-label={`How loud each recorded voice is over time: ${voices.map((v) => v.k).join(", ")}`}
          >
            {voices.map((v) => (
              <path key={v.k} d={v.d} className={`jb-voice-line g-${v.k}`} />
            ))}
          </svg>
        </div>
      ) : null}
      {segments.length ? (
        <div className="jb-lane">
          <span className="jb-lane-label" id="lane-seg">
            Sections
            <small>What the movie shows</small>
          </span>
          <div
            className="jb-lane-body jb-regions jb-segs"
            role="img"
            aria-label={`Sections: ${segments.map((s) => `${s.name} from ${fmt(s.start)} to ${fmt(s.end)}`).join(", ")}`}
          >
            {segments.map((s) => (
              <span key={s.start} className="jb-seg" style={{ left: `${(s.start / duration) * 100}%`, width: `${((s.end - s.start) / duration) * 100}%` }}>
                {s.name}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </>
  );
});

export function DataTimeline({ track, prep, controls, duration, store, onSeek, onToggle }: Props) {
  const time = useTime(store);
  const rail = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const seekFromPointer = useCallback(
    (clientX: number) => {
      const el = rail.current;
      if (!el || !duration) return;
      const r = el.getBoundingClientRect();
      onSeek(Math.min(1, Math.max(0, (clientX - r.left) / r.width)) * duration);
    },
    [duration, onSeek],
  );

  const pct = duration ? (Math.min(time, duration) / duration) * 100 : 0;
  const i = leadIndexAt(prep.lead, time);
  const value = i >= 0 ? fmtValue(prep.lead[i].value, shortUnits(track.units), prep.range) : null;

  const onKey = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 15 : 5;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") onSeek(time + step);
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") onSeek(time - step);
    else if (e.key === "Home") onSeek(0);
    else if (e.key === "End") onSeek(duration);
    else if (e.key === "PageUp") onSeek(time + 30);
    else if (e.key === "PageDown") onSeek(time - 30);
    else if (e.key === " ") onToggle();
    else return;
    e.preventDefault();
  };

  return (
    <div className="jb-timeline">
      <div className="jb-lanes-wrap" style={{ "--p": `${pct}%` } as React.CSSProperties}>
        <Lanes track={track} prep={prep} controls={controls} duration={duration} />
        <div className="jb-rail">
          <div
            ref={rail}
            className="jb-scrub"
            role="slider"
            tabIndex={0}
            aria-label="Playback position"
            aria-valuemin={0}
            aria-valuemax={Math.round(duration)}
            aria-valuenow={Math.round(time)}
            aria-valuetext={`${fmtTime(time)} of ${fmtTime(duration)}${hasDates(track) ? `, ${timeLabel(track, time, duration)}` : ""}${value ? `, ${value}` : ""}`}
            onKeyDown={onKey}
            onPointerDown={(e) => {
              dragging.current = true;
              e.currentTarget.setPointerCapture(e.pointerId);
              seekFromPointer(e.clientX);
            }}
            onPointerMove={(e) => dragging.current && seekFromPointer(e.clientX)}
            onPointerUp={() => (dragging.current = false)}
            onPointerCancel={() => (dragging.current = false)}
          >
            <span className="jb-playhead" style={{ left: `${pct}%` }} aria-hidden="true">
              <span className="jb-playhead-tag">
                {fmt(time)}
                {value ? ` · ${value}` : ""}
              </span>
            </span>
          </div>
        </div>
      </div>
      <div className="jb-axis" aria-hidden="true">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <span key={f} style={{ left: `${f * 100}%` }}>
            {fmt(f * duration)}
          </span>
        ))}
      </div>
    </div>
  );
}
