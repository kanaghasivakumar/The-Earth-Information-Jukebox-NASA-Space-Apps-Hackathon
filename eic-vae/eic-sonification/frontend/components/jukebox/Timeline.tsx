"use client";

import { memo, useCallback, useMemo, useRef } from "react";
import { useTime, type TimeStore } from "@/lib/timeStore";
import { fmt, regionAt, regionLetter, type Region } from "@/lib/jukebox";

interface Props {
  duration: number;
  store: TimeStore;
  regions: Region[];
  loudness: number[];
  movement: number[];
  onSeek: (seconds: number) => void;
  onToggle: () => void;
}

const W = 1000;

function area(values: number[], height: number) {
  const n = values.length;
  if (!n) return "";
  const pts = values.map((v, i) => `${((i / (n - 1)) * W).toFixed(1)},${(height - 2 - v * (height - 6)).toFixed(1)}`);
  return `M0,${height} L${pts.join(" L")} L${W},${height} Z`;
}

function line(values: number[], height: number) {
  const n = values.length;
  if (!n) return "";
  return "M" + values.map((v, i) => `${((i / (n - 1)) * W).toFixed(1)},${(height - 2 - v * (height - 6)).toFixed(1)}`).join(" L");
}

const Lanes = memo(function Lanes({ duration, regions, loudness, movement, currentStart }: Omit<Props, "store" | "onSeek" | "onToggle"> & { currentStart?: number }) {
  const loudArea = useMemo(() => area(loudness, 64), [loudness]);
  const moveLine = useMemo(() => line(movement, 64), [movement]);
  return (
    <>
      <div className="jb-lane">
        <span className="jb-lane-label" id="lane-regions">
          Pattern group
          <small>Which learned look the footage is closest to</small>
        </span>
        <div className="jb-lane-body jb-regions" role="img" aria-label={`Pattern groups across the timeline: ${regions.map(([a, b, id]) => `${regionLetter(id)} from ${fmt(a)} to ${fmt(b)}`).join(", ")}`}>
          {regions.map(([a, b, id]) => {
            const wide = (b - a) / duration > 0.035;
            return (
              <span
                key={`${a}-${id}`}
                className={`jb-region r${id}${currentStart === a ? " now" : ""}`}
                style={{ left: `${(a / duration) * 100}%`, width: `${((b - a) / duration) * 100}%` }}
              >
                {wide ? regionLetter(id) : ""}
              </span>
            );
          })}
        </div>
      </div>
      <div className="jb-lane">
        <span className="jb-lane-label" id="lane-loud">
          Loudness
          <small>Of the music you are hearing</small>
        </span>
        <svg className="jb-lane-body jb-curve" viewBox={`0 0 ${W} 64`} preserveAspectRatio="none" role="img" aria-labelledby="lane-loud">
          <path d={loudArea} className="jb-fill" />
          <path d={loudArea} className="jb-fill jb-played" />
        </svg>
      </div>
      <div className="jb-lane">
        <span className="jb-lane-label" id="lane-move">
          Visual change
          <small>How fast the footage is changing</small>
        </span>
        <svg className="jb-lane-body jb-curve" viewBox={`0 0 ${W} 64`} preserveAspectRatio="none" role="img" aria-labelledby="lane-move">
          <path d={moveLine} className="jb-stroke" />
        </svg>
      </div>
    </>
  );
});

export function Timeline({ duration, store, regions, loudness, movement, onSeek, onToggle }: Props) {
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

  const current = regionAt(regions, time);
  const pct = duration ? (time / duration) * 100 : 0;

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
        <Lanes duration={duration} regions={regions} loudness={loudness} movement={movement} currentStart={current?.[0]} />
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
            aria-valuetext={`${fmt(time)} of ${fmt(duration)}${current ? `, pattern group ${regionLetter(current[2])}` : ""}`}
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
              <span className="jb-playhead-tag">{fmt(time)}{current ? ` · ${regionLetter(current[2])}` : ""}</span>
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
