"use client";

import { memo } from "react";
import { AudioWaveform, Clapperboard, Disc3, Piano, Waves, Zap } from "lucide-react";
import { STYLE_COPY, type Track } from "@/lib/jukebox";

const STYLE_ICON = { ambient: Waves, cinematic: Clapperboard, classical: Piano, jazz: Disc3, electronic: Zap } as const;

interface Props {
  tracks: Track[];
  trackId: string;
  onChoose: (id: string) => void;
}

export const StyleList = memo(function StyleList({ tracks, trackId, onChoose }: Props) {
  const onRadioKey = (e: React.KeyboardEvent, i: number) => {
    const dir = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : e.key === "ArrowUp" || e.key === "ArrowLeft" ? -1 : 0;
    if (!dir) return;
    e.preventDefault();
    const next = tracks[(i + dir + tracks.length) % tracks.length];
    onChoose(next.id);
    document.getElementById(`style-${next.id}`)?.focus();
  };

  return (
    <div role="radiogroup" aria-label="Music style" className="jb-styles">
      {tracks.map((t, i) => {
        const c = STYLE_COPY[t.id] ?? { name: t.id, line: "" };
        const on = t.id === trackId;
        return (
          <button
            key={t.id}
            id={`style-${t.id}`}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            className="jb-style"
            onClick={() => onChoose(t.id)}
            onKeyDown={(e) => onRadioKey(e, i)}
          >
            <span className="jb-style-icon" aria-hidden="true">
              {(() => {
                const Icon = STYLE_ICON[t.id as keyof typeof STYLE_ICON] ?? AudioWaveform;
                return <Icon size={22} />;
              })()}
            </span>
            <span className="jb-style-name">{c.name}</span>
            <span className="jb-style-line">{c.line}</span>
            <span className="jb-style-meta">
              {t.tempo} BPM · {t.scale}
            </span>
            <kbd aria-hidden="true">{i + 1}</kbd>
          </button>
        );
      })}
    </div>
  );
});
