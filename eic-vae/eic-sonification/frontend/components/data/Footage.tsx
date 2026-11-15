"use client";

import { useState } from "react";
import type { DataTrack } from "@/lib/data/types";

/** Where the footage can come from, best first: the local web copy, then NASA's own files. */
export function videoSources(track: DataTrack, base: string): string[] {
  const local = track.video_local ? base + track.video_local : null;
  return [local, track.video_url, track.video_alt_url].filter((u): u is string => !!u);
}

const typeOf = (u: string) => (/\.webm$/i.test(u) ? "video/webm" : "video/mp4");

/** Keep a silent footage element in step with the audio clock (seconds of music). */
export function syncFootage(v: HTMLVideoElement | null, musicT: number, playing: boolean, force = false) {
  if (!v || !v.querySelector("source")) return;
  // video_rate < 1: the music runs slower than the movie (5046), so the video plays slowed down
  const rate = Number(v.dataset.rate) || 1;
  const t = musicT * rate;
  if (v.playbackRate !== rate) v.playbackRate = rate;
  if (playing) {
    if (v.paused) v.play().catch(() => {});
    if (!force && Math.abs(v.currentTime - t) <= 0.3) return;
  } else {
    if (!v.paused) v.pause();
    if (!force) return;
  }
  try {
    v.currentTime = t;
  } catch {
    /* not seekable yet */
  }
}

/**
 * The big footage frame of a data track: the NASA movie (or the 5669 still with its sweep line), sized
 * to its own aspect ratio and never cropped. The whole frame is also a link to the visualization's NASA
 * SVS page (new tab), with a visible "View on NASA SVS" caption; it takes no player keys. When the movie
 * cannot play, the poster (if any) and a message take its place, so the box is never blank.
 */
export function Footage({
  track,
  base,
  videoRef,
  sweepRef,
}: {
  track: DataTrack;
  base: string;
  videoRef?: React.Ref<HTMLVideoElement>;
  sweepRef?: React.Ref<HTMLDivElement>;
}) {
  const [ar, setAr] = useState<number | null>(null);
  const [failed, setFailed] = useState(false);
  const sources = videoSources(track, base);
  const poster = track.poster ? base + track.poster : undefined;
  const still = !sources.length && track.image_url ? track.image_url : null;
  const fail = () => setFailed(true);
  const style = { "--ar": String(ar ?? 16 / 9) } as React.CSSProperties;

  return (
    <div className="jb-video jb-frame" style={style}>
      {sources.length && !failed ? (
        <video
          key={track.id}
          ref={videoRef}
          data-rate={track.video_rate ?? 1}
          muted
          playsInline
          preload="metadata"
          poster={poster}
          aria-label={`Silent NASA visualization, ${track.title}, played in step with the music`}
          onLoadedMetadata={(e) => {
            const v = e.currentTarget;
            if (v.videoWidth && v.videoHeight) setAr(v.videoWidth / v.videoHeight);
          }}
          onError={fail}
        >
          {sources.map((u, i) => (
            // only the last source failing means nothing can play; earlier ones fall through to the next
            <source key={u} src={u} type={typeOf(u)} onError={i === sources.length - 1 ? fail : undefined} />
          ))}
        </video>
      ) : null}
      {sources.length && failed ? (
        <div className="jb-frame-fail" role="note">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          {poster ? <img src={poster} alt="" /> : null}
          <p>
            The NASA video could not play in this browser. The music still plays. Open the visualization on NASA SVS to watch it there.
          </p>
        </div>
      ) : null}
      {still ? (
        <div className="jb-still">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={still}
            alt={`NASA SVS image: ${track.title}. The music sweeps across it from west to east.`}
            onLoad={(e) => {
              const im = e.currentTarget;
              if (im.naturalWidth && im.naturalHeight) setAr(im.naturalWidth / im.naturalHeight);
            }}
          />
          <div ref={sweepRef} className="jb-sweep-line" aria-hidden="true" />
        </div>
      ) : null}
      {track.svs_url ? (
        <a className="jb-frame-link" href={track.svs_url} target="_blank" rel="noopener noreferrer" aria-label={`Open NASA SVS page for ${track.title} (new tab)`}>
          <span className="jb-frame-cap" aria-hidden="true">
            View on NASA SVS ↗
          </span>
        </a>
      ) : null}
    </div>
  );
}
