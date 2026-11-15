"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { Download, Pause, Play, RotateCcw, RotateCw } from "lucide-react";
import { createTimeStore, useTime, type TimeStore } from "@/lib/timeStore";
import { fmt } from "@/lib/jukebox";
import { CATALOGUE, TRACK_COPY, isDataTrack } from "@/lib/data/copy";
import { loadVersions, TRACKS_BASE } from "@/lib/data/load";
import { MultiPlayer, audioSupported } from "@/lib/data/engine";
import { compareKeyAction, fromEvent } from "@/lib/data/keys";
import { RATINGS_KEY, exportRatings, fidelityLine, setRating, switchVersion, type Ratings } from "@/lib/data/compare";
import { toggleSame } from "@/lib/data/speech";
import type { Version } from "@/lib/data/types";

const EVENT = "jb:compare";
const RATING_EVENT = "jb:ratings";
const KIND_TAG: Record<Version["kind"], string> = { deterministic: "Data-faithful", learned: "Learned layer" };

// ---- ?track= in the address bar -------------------------------------------------
function subscribeTrack(fn: () => void) {
  window.addEventListener("popstate", fn);
  window.addEventListener(EVENT, fn);
  return () => {
    window.removeEventListener("popstate", fn);
    window.removeEventListener(EVENT, fn);
  };
}
const readTrack = () => {
  const id = new URLSearchParams(window.location.search).get("track");
  return isDataTrack(id) ? id : CATALOGUE[0].id;
};

// ---- ratings in localStorage (an in-memory copy keeps working when storage is blocked) ----
let cachedRaw: string | null | undefined;
let cached: Ratings = {};
let memory: Ratings | null = null;
const EMPTY: Ratings = {};
function readRatings(): Ratings {
  if (memory) return memory;
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(RATINGS_KEY);
  } catch {
    raw = null;
  }
  if (raw !== cachedRaw) {
    cachedRaw = raw;
    try {
      const v = raw ? (JSON.parse(raw) as unknown) : {};
      cached = v && typeof v === "object" && !Array.isArray(v) ? (v as Ratings) : {};
    } catch {
      cached = {};
    }
  }
  return cached;
}
function subscribeRatings(fn: () => void) {
  window.addEventListener("storage", fn);
  window.addEventListener(RATING_EVENT, fn);
  return () => {
    window.removeEventListener("storage", fn);
    window.removeEventListener(RATING_EVENT, fn);
  };
}
function storageOrNull() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
/** Save one rating; returns whether it reached localStorage. Listeners re-read through readRatings(). */
function persistRating(trackId: string, versionId: string, patch: { rating?: number | null; notes?: string }): boolean {
  const r = setRating(readRatings(), trackId, versionId, patch, storageOrNull());
  memory = r.saved ? null : r.ratings;
  window.dispatchEvent(new Event(RATING_EVENT));
  return r.saved;
}

function Clock({ store, duration }: { store: TimeStore; duration: number }) {
  const t = useTime(store);
  return (
    <span className="jb-clock jb-code" aria-hidden="true">
      {fmt(t)} / {fmt(duration)}
    </span>
  );
}

function Position({ store, duration, onSeek }: { store: TimeStore; duration: number; onSeek: (t: number) => void }) {
  const t = useTime(store);
  return (
    <label className="jb-vae-slider jb-cmp-pos">
      <span className="jb-code">{fmt(t)}</span>
      <input
        type="range"
        min={0}
        max={Math.max(1, Math.round(duration * 10)) / 10}
        step={1}
        value={Math.min(t, duration)}
        onChange={(e) => onSeek(Number(e.target.value))}
        aria-label="Playback position"
        aria-valuetext={`${fmt(t)} of ${fmt(duration)}`}
      />
    </label>
  );
}

export function CompareView() {
  const trackId = useSyncExternalStore(subscribeTrack, readTrack, () => CATALOGUE[0].id);
  const ratings = useSyncExternalStore(subscribeRatings, readRatings, () => EMPTY);
  const store = useMemo(() => createTimeStore(0), []);
  const player = useRef<MultiPlayer | null>(null);
  const [versions, setVersions] = useState<{ id: string; list: Version[] } | null>(null);
  const [selected, setSelected] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState<string | null>(null);
  const [status, setStatus] = useState("Loading the versions…");
  const [said, setSaid] = useState("");
  const [saved, setSaved] = useState<string>("");

  const list = useMemo(() => (versions?.id === trackId ? versions.list : []), [versions, trackId]);
  const isReady = ready === trackId;
  const duration = CATALOGUE.find((c) => c.id === trackId)?.duration_s ?? 1;
  const current = list[selected] ?? list[0];

  const announce = useCallback((t: string) => setSaid((p) => toggleSame(p, t)), []);

  const chooseTrack = useCallback((id: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set("track", id);
    if (url.href !== window.location.href) window.history.pushState(null, "", url);
    window.dispatchEvent(new Event(EVENT));
  }, []);

  // load every version of the track and decode them onto one clock
  useEffect(() => {
    let alive = true;
    const mp = new MultiPlayer();
    player.current = mp;
    (async () => {
      try {
        const v = await loadVersions(trackId);
        if (!alive) return;
        setVersions({ id: trackId, list: v.versions });
        setSelected(0);
        if (!audioSupported()) {
          setStatus("This browser has no Web Audio, so versions cannot switch in place. Each one has a download link below.");
          return;
        }
        setStatus(`Loading ${v.versions.length} versions…`);
        const ok = await mp.load(
          v.versions.map((x) => `${TRACKS_BASE}/${trackId}/${x.file}`),
          (n, total) => alive && setStatus(`Loading version ${n} of ${total}…`),
        );
        if (!alive || !ok) return;
        store.set(0);
        setReady(trackId);
        setStatus("");
        announce(`${v.versions.length} versions of ${TRACK_COPY[trackId]?.name ?? trackId} are ready. Press 1 to ${v.versions.length} to switch, Space to play.`);
      } catch (e) {
        if (alive) setStatus(`The versions did not load (${e instanceof Error ? e.message : String(e)}).`);
      }
    })();
    return () => {
      alive = false;
      mp.close();
      player.current = null;
      setPlaying(false);
    };
  }, [trackId, store, announce]);

  // clock
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const mp = player.current;
      if (!mp) return;
      store.set(mp.time);
      if (mp.time >= mp.duration - 0.02) {
        mp.pause();
        setPlaying(false);
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, store]);

  const toggle = useCallback(async () => {
    const mp = player.current;
    if (!mp || !isReady) return;
    if (mp.playing) {
      mp.pause();
      store.set(mp.time);
      setPlaying(false);
    } else {
      await mp.play();
      setPlaying(true);
    }
  }, [isReady, store]);

  const seek = useCallback(
    (t: number) => {
      const mp = player.current;
      if (!mp || !isReady) return;
      mp.seek(t);
      store.set(mp.time);
    },
    [isReady, store],
  );

  const choose = useCallback(
    (i: number) => {
      if (!list.length) return;
      const r = switchVersion(list, selected, i, player.current?.time ?? 0);
      player.current?.select(r.selected);
      setSelected(r.selected);
      announce(r.announcement);
    },
    [list, selected, announce],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const a = compareKeyAction(fromEvent(e), list.length);
      if (!a) return;
      e.preventDefault();
      if (a.type === "toggle") void toggle();
      else if (a.type === "seek") seek((player.current?.time ?? 0) + a.by);
      else if (a.type === "home") seek(0);
      else choose(a.index);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [list.length, toggle, seek, choose]);

  const rate = (versionId: string, patch: { rating?: number | null; notes?: string }) => {
    const ok = persistRating(trackId, versionId, patch);
    setSaved(ok ? "Saved in this browser." : "Could not save in this browser (storage is blocked or full). Export before you leave.");
  };

  const doExport = async () => {
    const catalogue: Record<string, Version[]> = {};
    await Promise.all(
      CATALOGUE.map(async (c) => {
        try {
          catalogue[c.id] = (await loadVersions(c.id)).versions;
        } catch {
          /* labels fall back to version ids */
        }
      }),
    );
    const data = exportRatings(readRatings(), catalogue);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `earth-jukebox-ratings-${data.exported.slice(0, 10)}.json`;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    announce(`Exported ${data.ratings.length} rating${data.ratings.length === 1 ? "" : "s"}.`);
  };

  const mine = current ? ratings[trackId]?.[current.id] : undefined;
  const ratedCount = Object.values(ratings).reduce((n, byV) => n + Object.values(byV).filter((r) => r.rating != null || r.notes.trim()).length, 0);

  const onRowKey = (e: React.KeyboardEvent, i: number) => {
    const dir = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : e.key === "ArrowUp" || e.key === "ArrowLeft" ? -1 : 0;
    if (!dir) return;
    e.preventDefault();
    const n = (i + dir + list.length) % list.length;
    choose(n);
    document.getElementById(`ver-${list[n].id}`)?.focus();
  };

  return (
    <div className="jb-section jb-page jb-compare">
      <h1 className="jb-h1">Compare versions.</h1>
      <p className="jb-how-lede">
        Every version of a track plays from the same moment on one clock. Switch while it plays and only the sound changes, not your place. Each
        comes with its measured fidelity. Then say what you hear: a rating from 1 to 5 and a note, kept in this browser until you export them.
      </p>

      <div role="radiogroup" aria-label="Track to compare" className="jb-tracks">
        {CATALOGUE.map((c, i) => {
          const on = c.id === trackId;
          return (
            <button
              key={c.id}
              id={`cmp-${c.id}`}
              type="button"
              role="radio"
              aria-checked={on}
              tabIndex={on ? 0 : -1}
              className="jb-track"
              onClick={() => chooseTrack(c.id)}
              onKeyDown={(e) => {
                const dir = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : e.key === "ArrowUp" || e.key === "ArrowLeft" ? -1 : 0;
                if (!dir) return;
                e.preventDefault();
                e.stopPropagation();
                const next = CATALOGUE[(i + dir + CATALOGUE.length) % CATALOGUE.length];
                chooseTrack(next.id);
                document.getElementById(`cmp-${next.id}`)?.focus();
              }}
            >
              <span className="jb-track-name">{TRACK_COPY[c.id]?.name ?? c.title}</span>
              <span className="jb-track-tag">{c.versions} version{c.versions === 1 ? "" : "s"}</span>
            </button>
          );
        })}
      </div>

      <div className="jb-cmp">
        <div className="jb-cmp-main">
          <div className="jb-transport">
            <button type="button" className="jb-play" onClick={() => void toggle()} disabled={!isReady} aria-label={playing ? "Pause" : "Play"} aria-keyshortcuts="Space">
              {playing ? <Pause size={22} aria-hidden="true" /> : <Play size={22} aria-hidden="true" />}
              <span>{playing ? "Pause" : "Play"}</span>
            </button>
            <button type="button" className="jb-skip" onClick={() => seek(store.get() - 5)} disabled={!isReady} aria-label="Back 5 seconds">
              <RotateCcw size={18} aria-hidden="true" />
              <span>5s</span>
            </button>
            <button type="button" className="jb-skip" onClick={() => seek(store.get() + 5)} disabled={!isReady} aria-label="Forward 5 seconds">
              <RotateCw size={18} aria-hidden="true" />
              <span>5s</span>
            </button>
            <Clock store={store} duration={duration} />
          </div>
          <Position store={store} duration={duration} onSeek={seek} />
          <p className="jb-status" role="status">
            {status || (
              <span className="jb-keys">
                <kbd>1</kbd>–<kbd>{list.length}</kbd> switch version, same moment · <kbd>Space</kbd> play · <kbd>←</kbd> <kbd>→</kbd> skip 5s
              </span>
            )}
          </p>

          <div role="radiogroup" aria-label="Version" className="jb-styles jb-versions">
            {list.map((v, i) => {
              const on = i === selected;
              const r = ratings[trackId]?.[v.id];
              return (
                <button
                  key={v.id}
                  id={`ver-${v.id}`}
                  type="button"
                  role="radio"
                  aria-checked={on}
                  tabIndex={on ? 0 : -1}
                  className="jb-style"
                  onClick={() => choose(i)}
                  onKeyDown={(e) => onRowKey(e, i)}
                  aria-describedby={`ver-line-${v.id}`}
                >
                  <span className="jb-style-icon jb-code" aria-hidden="true">
                    {i + 1}
                  </span>
                  <span className="jb-style-name">
                    {v.label}
                    <span className="jb-kind">{KIND_TAG[v.kind]}</span>
                  </span>
                  <span className="jb-style-line" id={`ver-line-${v.id}`}>
                    {v.line} <span className="jb-ver-fid">{fidelityLine(v)}</span>
                  </span>
                  <span className="jb-style-meta">{r?.rating ? `rated ${r.rating}/5` : "not rated"}</span>
                  <kbd aria-hidden="true">{i + 1}</kbd>
                </button>
              );
            })}
          </div>
        </div>

        <section className="jb-panel jb-rate" aria-labelledby="rate-h">
          <h2 id="rate-h" className="jb-h2">
            Your judgement{current ? `: ${current.label}` : ""}
          </h2>
          {current ? (
            <>
              <p className="jb-fine">
                {current.fidelity.note} {fidelityLine(current)}.
              </p>
              <fieldset className="jb-fieldset">
                <legend>How good does it sound? 1 is poor, 5 is excellent.</legend>
                <div className="jb-checks">
                  {[1, 2, 3, 4, 5].map((n) => (
                    <label key={n} className="jb-check">
                      <input type="radio" name={`rate-${trackId}-${current.id}`} checked={mine?.rating === n} onChange={() => rate(current.id, { rating: n })} />
                      <span className="jb-code">{n}</span>
                    </label>
                  ))}
                  <button type="button" className="jb-link" onClick={() => rate(current.id, { rating: null })} disabled={mine?.rating == null}>
                    Clear
                  </button>
                </div>
              </fieldset>
              <label className="jb-notes">
                <span>Notes</span>
                <textarea
                  key={`${trackId}-${current.id}`}
                  rows={4}
                  defaultValue={mine?.notes ?? ""}
                  onBlur={(e) => {
                    if (e.target.value !== (mine?.notes ?? "")) rate(current.id, { notes: e.target.value });
                  }}
                  placeholder="What works, what clashes, where the melody gets lost…"
                />
              </label>
              <p className="jb-fine" role="status">
                {saved}
              </p>
              <p className="jb-fine">
                <a href={`${TRACKS_BASE}/${trackId}/${current.file}`} download>
                  Download this version (MP3)
                </a>
              </p>
            </>
          ) : null}
          <button type="button" className="jb-btn-ghost jb-export" onClick={() => void doExport()}>
            <Download size={18} aria-hidden="true" /> Export ratings ({ratedCount})
          </button>
        </section>
      </div>

      <p className="jb-honest">
        The numbers measure one thing: whether the data melody comes back from the finished audio. A pitch tracker listens to each melody note and the
        pitch is turned back into the data value. They say nothing about whether a version sounds good. That is what the ratings are for.
      </p>

      <div className="jb-sr" role="status" aria-atomic="true">
        {said}
      </div>
    </div>
  );
}
