"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Eye, EyeOff, Pause, Play, RotateCcw, RotateCw } from "lucide-react";
import { Timeline } from "./Timeline";
import { StyleList } from "./StyleList";
import { createTimeStore, useTime, useTimeSelector, type TimeStore } from "@/lib/timeStore";
import {
  REGION_COLORS,
  STYLE_COPY,
  earlierVisit,
  fmt,
  regionAt,
  regionLetter,
  sceneAt,
  type Signals,
  type Track,
} from "@/lib/jukebox";

export interface Source {
  id: string;
  title: string;
  lede: string;
  credit: string;
  video: string;
  poster?: string;
  signals: Signals;
  tracks: Track[];
  /** Written description of the footage. Uploaded sources have none. */
  scenes: boolean;
}

const TIME_HZ = 20;
const DRIFT_S = 0.25;

function Clock({ store, duration }: { store: TimeStore; duration: number }) {
  const t = useTime(store);
  return (
    <span className="jb-clock jb-code" aria-hidden="true">
      {fmt(t)} / {fmt(duration)}
    </span>
  );
}

export function Player({ source }: { source: Source }) {
  const { tracks, signals } = source;
  const audio = useRef<HTMLAudioElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const resume = useRef<{ t: number; play: boolean } | null>(null);
  const lastNote = useRef("");
  const [showVideo, setShowVideo] = useState(true);

  const [trackId, setTrackId] = useState(tracks[0].id);
  const [playing, setPlaying] = useState(false);
  const store = useMemo(() => createTimeStore(0), []);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const [note, setNote] = useState("");
  const [narrate, setNarrate] = useState(true);
  const narrateRef = useRef(true);
  useEffect(() => {
    narrateRef.current = narrate;
  }, [narrate]);

  const track = tracks.find((t) => t.id === trackId) ?? tracks[0];
  const copy = STYLE_COPY[track.id] ?? { name: track.id, line: "", hear: "" };
  const duration = track.duration;
  const region = useTimeSelector(store, (t) => regionAt(track.regions, t));
  const scene = useTimeSelector(store, (t) => (source.scenes ? sceneAt(t) : null));

  // ---- transport -------------------------------------------------------
  const seek = useCallback(
    (t: number) => {
      const a = audio.current;
      if (!a) return;
      const next = Math.min(Math.max(0, t), duration - 0.05);
      a.currentTime = next;
      if (video.current) video.current.currentTime = next;
      store.set(next);
    },
    [duration, store],
  );

  const toggle = useCallback(() => {
    const a = audio.current;
    if (!a) return;
    if (a.paused) a.play().catch(() => setFailed(true));
    else a.pause();
  }, []);

  // audio is the clock; video follows it
  useEffect(() => {
    const a = audio.current;
    const v = video.current;
    if (!a) return;
    let raf = 0;
    let shown = -1;
    const tick = () => {
      const t = a.currentTime;
      const key = Math.floor(t * TIME_HZ);
      if (key !== shown) {
        shown = key;
        store.set(t);
      }
      if (v && !a.paused && Math.abs(v.currentTime - t) > DRIFT_S) v.currentTime = t;
      raf = requestAnimationFrame(tick);
    };
    const onPlay = () => {
      setPlaying(true);
      v?.play().catch(() => {});
      raf = requestAnimationFrame(tick);
    };
    const onPause = () => {
      setPlaying(false);
      v?.pause();
      cancelAnimationFrame(raf);
      store.set(a.currentTime);
    };
    const onEnd = () => {
      setPlaying(false);
      v?.pause();
      cancelAnimationFrame(raf);
    };
    const onWaiting = () => setBusy(true);
    const onReady = () => {
      setBusy(false);
      setFailed(false);
    };
    const onMeta = () => {
      const r = resume.current;
      if (!r) return;
      resume.current = null;
      a.currentTime = Math.min(r.t, (a.duration || duration) - 0.05);
      if (v) v.currentTime = a.currentTime;
      if (r.play) a.play().catch(() => setFailed(true));
    };
    const onError = () => {
      setBusy(false);
      setFailed(true);
    };
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onPause);
    a.addEventListener("ended", onEnd);
    a.addEventListener("waiting", onWaiting);
    a.addEventListener("canplay", onReady);
    a.addEventListener("playing", onReady);
    a.addEventListener("loadedmetadata", onMeta);
    a.addEventListener("error", onError);
    return () => {
      cancelAnimationFrame(raf);
      ["play", "pause", "ended", "waiting", "canplay", "playing", "loadedmetadata", "error"].forEach((n) => {
        a.removeEventListener(n, n === "play" ? onPlay : n === "pause" ? onPause : n === "ended" ? onEnd : n === "waiting" ? onWaiting : n === "loadedmetadata" ? onMeta : n === "error" ? onError : onReady);
      });
    };
  }, [duration, store]);

  // ---- screen reader announcements (polite, only on real changes) ----------
  const announce = useCallback((text: string) => {
    if (text === lastNote.current || !narrateRef.current) return;
    lastNote.current = text;
    setNote(text);
  }, []);

  // ---- style switching keeps your place ---------------------------------
  const choose = useCallback(
    (id: string) => {
      const a = audio.current;
      if (!a || id === trackId) return;
      const next = tracks.find((t) => t.id === id);
      if (!next) return;
      resume.current = { t: a.currentTime, play: !a.paused };
      a.pause();
      setBusy(true);
      setFailed(false);
      setTrackId(id);
      announce(`Switched to ${STYLE_COPY[id]?.name ?? id} at ${fmt(a.currentTime)}.`);
    },
    [trackId, tracks, announce],
  );

  useEffect(() => {
    const a = audio.current;
    if (!a) return;
    a.src = track.audio;
    a.load();
  }, [track.audio]);

  const regionKey = region ? region[0] : -1;
  const sceneKey = scene ? scene.from : -1;
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    if (!playing || !region) return;
    const back = earlierVisit(track.regions, region);
    announce(
      `Pattern group ${regionLetter(region[2])} begins at ${fmt(region[0])}.` +
        (back !== null ? ` Seen before at ${fmt(back)}, so its motif may return.` : ""),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [regionKey]);
  useEffect(() => {
    if (!playing || !scene) return;
    announce(`On screen: ${scene.text}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneKey]);

  // ---- keyboard -------------------------------------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      const n = Number(e.key);
      if (n >= 1 && n <= tracks.length) return void choose(tracks[n - 1].id);
      if (tag === "BUTTON" || el?.getAttribute("role") === "slider" || el?.getAttribute("role") === "radio") return;
      if (e.key === " ") {
        e.preventDefault();
        toggle();
      } else if (e.key === "ArrowRight") seek((audio.current?.currentTime ?? 0) + (e.shiftKey ? 15 : 5));
      else if (e.key === "ArrowLeft") seek((audio.current?.currentTime ?? 0) - (e.shiftKey ? 15 : 5));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [choose, seek, toggle, tracks]);

  const returning = useMemo(() => (region ? earlierVisit(track.regions, region) : null), [region, track.regions]);

  return (
    <div className="jb-stage">
      <audio ref={audio} preload="metadata" />

      <div className="jb-left">
        <div className={`jb-video${showVideo ? "" : " off"}`}>
          {!showVideo && region ? (
            <div className="jb-sound-only" aria-hidden="true">
              <span style={{ color: REGION_COLORS[region[2]] }}>{regionLetter(region[2])}</span>
              <small>pattern group</small>
            </div>
          ) : null}
          <video
            ref={video}
            src={source.video}
            poster={source.poster}
            muted
            hidden={!showVideo}
            playsInline
            preload="metadata"
            aria-label={source.scenes ? "Silent NASA visualization, played in step with the music" : "Your silent visualization, played in step with the music"}
          />
        </div>

        <div className="jb-transport-wrap">
          <div className="jb-transport">
            <button type="button" className="jb-play" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
              {playing ? <Pause size={22} aria-hidden="true" /> : <Play size={22} aria-hidden="true" />}
              <span>{playing ? "Pause" : "Play"}</span>
            </button>
            <button type="button" className="jb-skip" onClick={() => seek(store.get() - 15)} aria-label="Back 15 seconds">
              <RotateCcw size={18} aria-hidden="true" />
              <span>15s</span>
            </button>
            <button type="button" className="jb-skip" onClick={() => seek(store.get() + 15)} aria-label="Forward 15 seconds">
              <RotateCw size={18} aria-hidden="true" />
              <span>15s</span>
            </button>
            <button
              type="button"
              className="jb-skip jb-vtoggle"
              aria-pressed={showVideo}
              onClick={() => setShowVideo((v) => !v)}
              aria-label={showVideo ? "Hide video, sound only view" : "Show video"}
            >
              {showVideo ? <EyeOff size={18} aria-hidden="true" /> : <Eye size={18} aria-hidden="true" />}
              <span>{showVideo ? "Sound only" : "Show video"}</span>
            </button>
            <Clock store={store} duration={duration} />
          </div>
          <label className="jb-narrate">
            <input type="checkbox" checked={narrate} onChange={(e) => setNarrate(e.target.checked)} />
            <span>Announce changes to screen readers</span>
          </label>
          <p className="jb-status" role="status">
            {failed ? (
              <>
                This score did not load.{" "}
                <button type="button" className="jb-link" onClick={() => audio.current?.load()}>
                  Try again
                </button>
              </>
            ) : busy ? (
              "Loading this style…"
            ) : (
              <span className="jb-keys">
                <kbd>Space</kbd> play · <kbd>←</kbd> <kbd>→</kbd> skip 5s · <kbd>1</kbd>–<kbd>{tracks.length}</kbd> switch style
              </span>
            )}
          </p>
        </div>
      </div>

      <div className="jb-right">
        <div className="jb-intro">
          <h1 className="jb-h1">{source.title}</h1>
          <p className="jb-lede">{source.lede}</p>
        </div>
        <StyleList tracks={tracks} trackId={trackId} onChoose={choose} />
      </div>

      <section className="jb-timeline-wrap" aria-label="Timeline">
        <Timeline
          duration={duration}
          store={store}
          regions={track.regions}
          loudness={track.loudness}
          movement={signals.movement}
          onSeek={seek}
          onToggle={toggle}
        />
      </section>

      <section className="jb-onscreen" aria-labelledby="onscreen-h">
        <h2 id="onscreen-h" className="jb-h2">
          What is on screen
        </h2>
        {scene ? (
          <p>
            <span className="jb-code">{fmt(scene.from)}</span> {scene.text}
          </p>
        ) : (
          <p>Your uploaded visualization plays above, silent, in step with the music. No written description is available for it.</p>
        )}
        <small>
          {source.credit}
          {source.scenes ? " Descriptions of the footage are written by the team, not generated by the model." : ""}
        </small>
      </section>

      <section className="jb-now" aria-labelledby="now-h">
        <h2 id="now-h" className="jb-h2">
          Right now in {copy.name}
        </h2>
        <p className="jb-now-region">
          {region ? (
            <>
              <span className={`jb-chip r${region[2]}`} aria-hidden="true">
                {regionLetter(region[2])}
              </span>
              <span>
                The footage is closest to <strong>pattern group {regionLetter(region[2])}</strong>.
                {returning !== null
                  ? ` It was last seen near ${fmt(returning)}, so you may hear its motif come back.`
                  : " This is the first time this group appears."}
              </span>
            </>
          ) : null}
        </p>
        <p className="jb-now-hear">{copy.hear}</p>
        <dl className="jb-roles">
          {track.instruments.map((i) => (
            <div key={i.role}>
              <dt>{i.role}</dt>
              <dd>{i.sound}</dd>
            </div>
          ))}
        </dl>
      </section>

      <div className="jb-sr" aria-live="polite" aria-atomic="true">
        {note}
      </div>
    </div>
  );
}
