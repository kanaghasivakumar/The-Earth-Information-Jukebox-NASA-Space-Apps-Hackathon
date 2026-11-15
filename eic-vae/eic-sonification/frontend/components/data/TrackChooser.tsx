"use client";

import { memo } from "react";
import { CATALOGUE, TRACK_COPY } from "@/lib/data/copy";
import { fidelityTag } from "@/lib/data/fidelity";
import { fmt } from "@/lib/jukebox";

export const SEASON = "season";

/** One row of choices above the player: the hurricane season (artistic) and the eight data tracks. */
export const TrackChooser = memo(function TrackChooser({ current, onChoose }: { current: string; onChoose: (id: string) => void }) {
  const items = [
    { id: SEASON, name: "Hurricane season", kind: "Artistic sonification", value: "5 styles" },
    ...CATALOGUE.map((c) => {
      const tag = fidelityTag(c.r2);
      return { id: c.id, name: TRACK_COPY[c.id]?.name ?? c.title, kind: tag.kind, value: [tag.value, fmt(c.duration_s)].filter(Boolean).join(" · ") };
    }),
  ];
  const onKey = (e: React.KeyboardEvent, i: number) => {
    const dir = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : e.key === "ArrowUp" || e.key === "ArrowLeft" ? -1 : 0;
    if (!dir) return;
    e.preventDefault();
    e.stopPropagation();
    const next = items[(i + dir + items.length) % items.length];
    onChoose(next.id);
    document.getElementById(`pick-${next.id}`)?.focus();
  };
  return (
    <div role="radiogroup" aria-label="Choose a visualization" className="jb-tracks">
      {items.map((it, i) => {
        const on = it.id === current;
        return (
          <button
            key={it.id}
            id={`pick-${it.id}`}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            className="jb-track"
            onClick={() => onChoose(it.id)}
            onKeyDown={(e) => onKey(e, i)}
          >
            <span className="jb-track-name">{it.name}</span>
            <span className="jb-track-tag">
              {it.kind} <span className="jb-code">{it.value}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
});
