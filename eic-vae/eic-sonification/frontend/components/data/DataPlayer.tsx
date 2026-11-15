"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw, RotateCw } from "lucide-react";
import { createTimeStore, useTime, useTimeSelector, type TimeStore } from "@/lib/timeStore";
import { fmt } from "@/lib/jukebox";
import { CATALOGUE, TRACK_COPY } from "@/lib/data/copy";
import { loadTrack, prepare, sweepVoices, groupOf } from "@/lib/data/load";
import { Engine, audioSupported, type GroupState } from "@/lib/data/engine";
import { Announcer, canSpeak, cancelSpeech, sleep, speak, toggleSame } from "@/lib/data/speech";
import { fromEvent, keyAction, type DataAction } from "@/lib/data/keys";
import { GROUP_LABELS, fmtValue, shortUnits, spokenTime, spokenValue } from "@/lib/data/music";
import { describeNow, fmtTime, hasDates, leadIndexAt, noteAt, scoreSummary, sweepFraction, timeLabel } from "@/lib/data/time";
import { Roll } from "@/lib/data/roll";
import { GROUPS, type Group, type LoadedTrack, type TimelineNote } from "@/lib/data/types";
import { DataTimeline } from "./DataTimeline";
import { Footage, syncFootage } from "./Footage";
import { HelpDialog, LegendPanel, ScorePanel, StemMixer } from "./Panels";

const TIME_HZ = 20;
type Cadence = "0" | "5" | "10" | "note";

const subscribeNothing = () => () => {};
function subscribeReduced(fn: () => void) {
  const m = window.matchMedia("(prefers-reduced-motion: reduce)");
  m.addEventListener("change", fn);
  return () => m.removeEventListener("change", fn);
}
const reducedNow = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const freshGroups = () => Object.fromEntries(GROUPS.map((g) => [g, { volume: 1, muted: false }])) as Record<Group, GroupState>;

function Clock({ store, duration }: { store: TimeStore; duration: number }) {
  const t = useTime(store);
  return (
    <span className="jb-clock jb-code" aria-hidden="true">
      {fmt(t)} / {fmt(duration)}
    </span>
  );
}

/** Compact scrub for the side column; the score strip under the video has the full timeline slider. */
function Position({ store, duration, disabled, onSeek, label }: { store: TimeStore; duration: number; disabled: boolean; onSeek: (t: number) => void; label?: (t: number) => string }) {
  const t = useTime(store);
  return (
    <label className="jb-vae-slider jb-side-pos">
      <span className="jb-sr">Playback position</span>
      <input
        type="range"
        min={0}
        max={Math.max(1, Math.round(duration * 10) / 10)}
        step={0.1}
        value={Math.min(t, duration)}
        disabled={disabled}
        onChange={(e) => onSeek(Number(e.target.value))}
        aria-valuetext={`${fmtTime(t)} of ${fmtTime(duration)}${label ? `, ${label(t)}` : ""}`}
      />
    </label>
  );
}

interface Busy {
  kind: "legend" | "sweep" | "overview";
  cancelled: boolean;
}

export function DataPlayer({ id }: { id: string }) {
  const copy = TRACK_COPY[id] ?? { name: id, heading: id, lede: "" };
  const entry = CATALOGUE.find((c) => c.id === id);
  const store = useMemo(() => createTimeStore(0), []);
  const [loaded, setLoaded] = useState<LoadedTrack | null>(null);
  const [status, setStatus] = useState("Loading the track…");
  const [failed, setFailed] = useState<string | null>(null);
  const [fallback, setFallback] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [polarity, setPolarity] = useState(1);
  const [binaural, setBinaural] = useState(false);
  const [allMuted, setAllMuted] = useState(false);
  const [groups, setGroups] = useState(freshGroups);
  const [cadence, setCadence] = useState<Cadence>("5");
  const [voice, setVoice] = useState(false);
  const [motionPref, setMotionPref] = useState<boolean | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [live, setLive] = useState("");
  const [said, setSaid] = useState("");

  const reduced = useSyncExternalStore(subscribeReduced, reducedNow, () => false);
  const speechOk = useSyncExternalStore(subscribeNothing, canSpeak, () => false);
  const motion = motionPref ?? !reduced;

  const prep = useMemo(() => (loaded ? prepare(loaded.track, loaded.timeline) : null), [loaded]);
  const track = loaded?.track ?? null;
  const duration = track?.duration_s ?? entry?.duration_s ?? 1;
  const sweeps = useMemo(() => (loaded ? sweepVoices(loaded.track, loaded.timeline) : []), [loaded]);

  const engine = useRef<Engine | null>(null);
  const announcer = useRef<Announcer | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const sweepLine = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const roll = useRef<Roll | null>(null);
  const busy = useRef<Busy | null>(null);
  // the latest settings, for the clock loop and the key handler (kept in sync after each render)
  const S = useRef({ polarity, binaural, cadence, motion, helpOpen, lastAnnounceT: -1, lastLeadIdx: -1 });
  useEffect(() => {
    Object.assign(S.current, { polarity, binaural, cadence, motion, helpOpen });
  }, [polarity, binaural, cadence, motion, helpOpen]);
  useEffect(() => {
    if (announcer.current) announcer.current.speech = voice;
  }, [voice]);

  // ---- load: track files, then decode every stem -------------------------
  useEffect(() => {
    let alive = true;
    const eng = new Engine();
    engine.current = eng;
    const ann = new Announcer(
      (t) => setLive((p) => toggleSame(p, t)),
      (t) => setSaid((p) => toggleSame(p, t)),
    );
    announcer.current = ann;
    (async () => {
      try {
        const lt = await loadTrack(id);
        if (!alive) return;
        setLoaded(lt);
        if (!audioSupported()) {
          setFallback(lt.base + (lt.track.mix ?? "mix.mp3"));
          setStatus("This browser has no Web Audio, so the stems cannot play separately. The full mix plays below.");
          return;
        }
        const ok = await eng.load(lt.track, lt.base, (n, total) => {
          if (alive) setStatus(`Loading sound ${n} of ${total}…`);
        });
        if (!alive || !ok) return;
        store.set(0);
        setReady(true);
        setStatus("");
        ann.status(`${lt.track.title} is ready, ${spokenTime(eng.duration)}. Press Space to play, L for the legend, question mark for all shortcuts.`);
      } catch (e) {
        if (!alive) return;
        const msg = e instanceof Error ? e.message : String(e);
        setFailed(msg);
        setFallback(`/jukebox/tracks/${id}/mix.mp3`);
        setStatus("");
      }
    })();
    return () => {
      alive = false;
      ann.dispose();
      busy.current = null;
      cancelSpeech();
      eng.close();
      engine.current = null;
      setPlaying(false);
    };
  }, [id, store]);

  // ---- the score picture ----------------------------------------------------
  useEffect(() => {
    if (!loaded || !canvas.current) return;
    const r = (roll.current ??= new Roll(canvas.current));
    const muted = Object.fromEntries(GROUPS.map((g) => [g, groups[g].muted])) as Record<Group, boolean>;
    r.setData({ notes: loaded.timeline, duration: loaded.track.duration_s, groupOf: (v) => groupOf(loaded.track, v), controls: loaded.controls, muted });
    r.draw(store.get());
  }, [loaded, groups, store]);
  useEffect(() => {
    const el = canvas.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => roll.current?.layout());
    ro.observe(el);
    return () => ro.disconnect();
  }, [loaded]);

  // ---- transport --------------------------------------------------------------
  const syncVideo = useCallback((force = false) => {
    const eng = engine.current;
    if (eng) syncFootage(video.current, eng.time, eng.playing && S.current.motion, force);
  }, []);

  const stopBusy = useCallback(() => {
    if (busy.current) busy.current.cancelled = true;
    busy.current = null;
    cancelSpeech();
    engine.current?.duck(1);
  }, []);

  const describe = useCallback(
    (t: number, full = false) =>
      track && prep ? describeNow(track, prep, t, { full, controls: loaded?.controls ?? null }) : "",
    [track, prep, loaded],
  );

  const togglePlay = useCallback(async () => {
    const eng = engine.current;
    if (!eng || !ready) return;
    stopBusy();
    if (eng.playing) {
      eng.pause();
      setPlaying(false);
      syncVideo(true);
      announcer.current?.status(`Paused. ${describe(eng.time)}`);
    } else {
      await eng.play();
      setPlaying(true);
      syncVideo(true);
      S.current.lastAnnounceT = eng.time;
    }
  }, [ready, stopBusy, syncVideo, describe]);

  const seek = useCallback(
    (t: number, announce = false) => {
      const eng = engine.current;
      if (!eng || !ready) return;
      eng.seek(t);
      store.set(eng.time);
      syncVideo(true);
      if (prep) S.current.lastLeadIdx = leadIndexAt(prep.lead, eng.time);
      if (announce) announcer.current?.say(describe(eng.time), { minGap: 0.4 });
    },
    [ready, store, syncVideo, prep, describe],
  );

  const pitchOf = useCallback(
    (n: TimelineNote) => (S.current.polarity < 0 && prep?.pm && n.value != null ? prep.pm.pitch(n.value, -1) : n.pitch),
    [prep],
  );

  const stepNote = useCallback(
    (dir: 1 | -1) => {
      const eng = engine.current;
      if (!eng || !ready || !prep?.lead.length || !track) return;
      const cur = leadIndexAt(prep.lead, eng.time);
      const target = Math.max(0, Math.min(prep.lead.length - 1, cur + dir));
      const n = prep.lead[target];
      seek(n.start);
      if (!eng.playing) {
        void eng.resume();
        eng.tone(pitchOf(n), { dur: Math.min(0.6, n.end - n.start), vel: 95 });
      }
      const prev = prep.lead[target - 1];
      const change = !prev || n.value == null || prev.value == null ? "" : n.value > prev.value ? ", up" : n.value < prev.value ? ", down" : ", same";
      announcer.current?.say(
        `Note ${target + 1} of ${prep.lead.length}, ${hasDates(track) ? timeLabel(track, n.start) : fmtTime(n.start)}: ${spokenValue(n.value, track.units, prep.range)}${change}.`,
      );
    },
    [ready, prep, track, seek, pitchOf],
  );

  const pausePlayback = useCallback(() => {
    const eng = engine.current;
    if (eng?.playing) {
      eng.pause();
      setPlaying(false);
      syncVideo(true);
    }
  }, [syncVideo]);

  // ---- listening tools ---------------------------------------------------------
  const legend = useCallback(async () => {
    const eng = engine.current;
    if (!track || !prep || !eng) return;
    if (busy.current?.kind === "legend") {
      stopBusy();
      announcer.current?.status("Legend stopped.");
      return;
    }
    pausePlayback();
    stopBusy();
    const job: Busy = { kind: "legend", cancelled: false };
    busy.current = job;
    await eng.resume();
    const u = track.units || "";
    const talk = (s: string) => {
      if (!canSpeak()) announcer.current?.say(s);
      return speak(s);
    };
    for (const s of [`Legend for ${track.title}.`, ...track.legend.map((l) => `${l.voice}: ${l.meaning}`)]) {
      if (job.cancelled) return;
      await talk(s);
    }
    const cal: [typeof prep.pm, string][] = [[prep.pm, "Melody"]];
    if (prep.bandPm && sweeps.length) cal.push([prep.bandPm, "Bands"]);
    for (const [pm, name] of cal) {
      if (!pm || job.cancelled) continue;
      await talk(`${name} calibration tones${S.current.polarity < 0 ? ", polarity flipped" : ""}.`);
      for (const [label, v] of [
        ["Lowest", pm.vmin],
        ["Middle", pm.vmid],
        ["Highest", pm.vmax],
      ] as const) {
        if (job.cancelled) return;
        await talk(`${label}: ${spokenValue(v, u, pm.vmax - pm.vmin)}.`);
        if (job.cancelled) return;
        eng.tone(pm.pitch(v, S.current.polarity), { dur: 0.8, vel: 100 });
        await sleep(1100);
      }
    }
    if (!job.cancelled) {
      await talk("End of legend. Press Space to play.");
      busy.current = null;
    }
  }, [track, prep, sweeps, pausePlayback, stopBusy]);

  const sayValue = useCallback(() => {
    const eng = engine.current;
    if (eng) announcer.current?.say(describe(eng.time, true));
  }, [describe]);

  const sweep = useCallback(async () => {
    const eng = engine.current;
    if (!eng || !loaded || !prep || !track) return;
    if (!sweeps.length) {
      announcer.current?.status("Sweep is not available for this track.");
      return;
    }
    stopBusy();
    const job: Busy = { kind: "sweep", cancelled: false };
    busy.current = job;
    await eng.resume();
    const ctx = eng.ensure();
    const t = eng.time;
    const hits = sweeps
      .map((v) => ({ ...v, n: noteAt(loaded.timeline.filter((n) => n.voice === v.voice), t) }))
      .filter((h): h is typeof h & { n: TimelineNote } => !!h.n);
    eng.duck(0.2);
    let when = ctx.currentTime + 0.05;
    for (const h of hits) {
      const pm = prep.bandPms[h.voice] ?? prep.bandPm;
      const p = S.current.polarity < 0 && pm && h.n.value != null ? pm.pitch(h.n.value, -1) : h.n.pitch;
      eng.tone(p, { when, dur: 0.32, vel: 100, pan: h.pan });
      when += 0.42;
    }
    await sleep((when - ctx.currentTime) * 1000 + 300);
    if (job.cancelled) return;
    eng.duck(1);
    busy.current = null;
    const range = prep.bandPm ? prep.bandPm.vmax - prep.bandPm.vmin : prep.range;
    announcer.current?.say(
      `Sweep, ${track.sweep?.label || "bands"}, at ${fmtTime(t)}: ` + hits.map((h) => `${h.label} ${spokenValue(h.n.value, track.units, range)}`).join("; ") + ".",
    );
  }, [loaded, prep, track, sweeps, stopBusy]);

  const overview = useCallback(async () => {
    const eng = engine.current;
    if (!eng || !prep?.lead.length || !track) return;
    pausePlayback();
    stopBusy();
    const job: Busy = { kind: "overview", cancelled: false };
    busy.current = job;
    await eng.resume();
    const ctx = eng.ensure();
    const val = (n: TimelineNote) => n.value as number;
    const lo = prep.lead.reduce((a, b) => (val(b) < val(a) ? b : a));
    const hi = prep.lead.reduce((a, b) => (val(b) > val(a) ? b : a));
    announcer.current?.say("Overview: the whole melody in 8 seconds, from left to right.");
    await sleep(2500);
    if (job.cancelled) return;
    const dur = track.duration_s;
    const k = 8 / dur;
    const t0 = ctx.currentTime + 0.05;
    for (const n of prep.lead) {
      eng.tone(pitchOf(n), { when: t0 + n.start * k, dur: Math.max(0.07, (n.end - n.start) * k * 0.9), vel: n.velocity, pan: -0.9 + 1.8 * (n.start / dur) });
    }
    await sleep(8400);
    if (job.cancelled) return;
    busy.current = null;
    announcer.current?.say(
      `End of overview. Lowest ${spokenValue(lo.value, track.units, prep.range)} at ${fmtTime(lo.start)}; highest ${spokenValue(hi.value, track.units, prep.range)} at ${fmtTime(hi.start)}.`,
    );
  }, [prep, track, pausePlayback, stopBusy, pitchOf]);

  const togglePolarity = useCallback(() => {
    const eng = engine.current;
    if (!eng || !prep) return;
    if (!prep.pm) {
      announcer.current?.status("Polarity flip is not available for this track.");
      return;
    }
    const next = -S.current.polarity;
    S.current.polarity = next;
    setPolarity(next);
    eng.setLiveLead(next < 0, prep.lead, pitchOf);
    announcer.current?.status(
      next < 0
        ? "Polarity flipped: higher values now play lower. The melody is synthesized live from the data."
        : "Normal polarity: higher pitch means a higher value. The recorded melody is back.",
    );
  }, [prep, pitchOf]);

  const driveBinaural = useCallback(
    (t: number) => {
      const eng = engine.current;
      if (!eng || !prep) return;
      for (const [g, notes] of Object.entries(prep.byGroup) as [Group, TimelineNote[]][]) {
        let sum = 0;
        let k = 0;
        for (const n of notes) {
          if (n.start > t) break;
          if (t < n.end) {
            sum += n.pan || 0;
            k++;
          }
        }
        if (k) eng.setGroupPan(g, Math.round((sum / k) * 20) / 20);
      }
    },
    [prep],
  );

  const toggleBinaural = useCallback(() => {
    const eng = engine.current;
    if (!eng) return;
    const on = !S.current.binaural;
    S.current.binaural = on;
    setBinaural(on);
    eng.ensure();
    eng.setBinaural(on);
    if (on) driveBinaural(eng.time);
    announcer.current?.status(on ? "Headphone 3D sound on. Use headphones." : "Headphone 3D sound off. Normal stereo.");
  }, [driveBinaural]);

  const toggleMuteAll = useCallback(() => {
    const eng = engine.current;
    if (!eng) return;
    const m = !eng.masterMuted;
    eng.setMasterMuted(m);
    setAllMuted(m);
    announcer.current?.status(m ? "All sound muted." : "Sound on.");
  }, []);

  const changeGroup = useCallback((g: Group, patch: Partial<GroupState>) => {
    engine.current?.setGroup(g, patch);
    setGroups((prev) => ({ ...prev, [g]: { ...prev[g], ...patch } }));
  }, []);

  const toggleGroup = useCallback(
    (g: Group) => {
      if (!prep?.present.includes(g)) {
        announcer.current?.status(`No ${GROUP_LABELS[g].toLowerCase()} in this track.`);
        return;
      }
      const muted = !(engine.current?.groupState[g].muted ?? false);
      changeGroup(g, { muted });
      announcer.current?.status(`${GROUP_LABELS[g]} ${muted ? "muted" : "on"}.`);
    },
    [prep, changeGroup],
  );

  // ---- clock loop, video sync, announcements --------------------------------
  useEffect(() => {
    if (!ready || !track || !prep) return;
    let raf = 0;
    let shown = -1;
    const still = !track.video_url && !track.video_local && !!track.image_url;
    const frame = () => {
      const eng = engine.current;
      if (!eng) return;
      const t = eng.time;
      const key = Math.floor(t * TIME_HZ);
      if (key !== shown) {
        shown = key;
        store.set(t);
      }
      roll.current?.draw(t);
      if (still && sweepLine.current) sweepLine.current.style.left = `${(sweepFraction(track, t) * 100).toFixed(2)}%`;
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    const tick = () => {
      const eng = engine.current;
      if (!eng) return;
      eng.tick();
      const t = eng.time;
      if (!eng.playing) return;
      if (t >= eng.duration - 0.02) {
        eng.pause();
        eng.seek(eng.duration);
        setPlaying(false);
        syncVideo(true);
        announcer.current?.status("End of track. Press Space to play again from the start.");
        return;
      }
      syncVideo();
      if (S.current.binaural) driveBinaural(t);
      const mode = S.current.cadence;
      if (mode === "0") return;
      if (mode === "note") {
        const i = leadIndexAt(prep.lead, t);
        if (i !== S.current.lastLeadIdx) {
          S.current.lastLeadIdx = i;
          announcer.current?.say(describe(t), { minGap: 1.5 });
        }
      } else if (t - S.current.lastAnnounceT >= Number(mode)) {
        S.current.lastAnnounceT = t;
        announcer.current?.say(describe(t), { minGap: 2 });
      }
    };
    const iv = setInterval(tick, 100);
    return () => {
      cancelAnimationFrame(raf);
      clearInterval(iv);
    };
  }, [ready, track, prep, store, syncVideo, driveBinaural, describe]);

  useEffect(() => {
    syncVideo(true);
  }, [motion, syncVideo]);

  // ---- keyboard ----------------------------------------------------------------
  const run = useCallback(
    (a: DataAction) => {
      const t = engine.current?.time ?? 0;
      switch (a.type) {
        case "toggle":
          return void togglePlay();
        case "seek":
          return seek(t + a.by, true);
        case "step":
          return stepNote(a.dir);
        case "home":
          return seek(0, true);
        case "legend":
          return void legend();
        case "value":
          return sayValue();
        case "sweep":
          return void sweep();
        case "overview":
          return void overview();
        case "polarity":
          return togglePolarity();
        case "hrtf":
          return toggleBinaural();
        case "mute":
          return toggleGroup(a.group);
        case "muteAll":
          return toggleMuteAll();
        case "help":
          return setHelpOpen((o) => !o);
        case "stop":
          if (busy.current) {
            stopBusy();
            announcer.current?.status("Stopped.");
          }
      }
    },
    [togglePlay, seek, stepNote, legend, sayValue, sweep, overview, togglePolarity, toggleBinaural, toggleGroup, toggleMuteAll, stopBusy, setHelpOpen],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const a = keyAction(fromEvent(e), { helpOpen: S.current.helpOpen });
      if (!a) return;
      e.preventDefault();
      run(a);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [run]);

  // ---- "right now" readout -------------------------------------------------------
  const leadIdx = useTimeSelector(
    store,
    useCallback((t: number) => (prep ? leadIndexAt(prep.lead, t) : -1), [prep]),
  );
  const dateNow = useTimeSelector(
    store,
    useCallback((t: number) => (track && hasDates(track) ? timeLabel(track, t) : ""), [track]),
  );
  const nowNote = prep && leadIdx >= 0 ? prep.lead[leadIdx] : null;
  const prevNote = prep && leadIdx > 0 ? prep.lead[leadIdx - 1] : null;
  const trend =
    nowNote?.value != null && prevNote?.value != null
      ? nowNote.value > prevNote.value + 1e-9
        ? "rising"
        : nowNote.value < prevNote.value - 1e-9
          ? "falling"
          : "steady"
      : "";
  const units = track ? shortUnits(track.units) : "";
  const base = loaded?.base ?? `/jukebox/tracks/${id}/`;

  return (
    <>
      <section className="jb-panel jb-about" aria-labelledby="about-h">
        <h2 id="about-h" className="jb-h2">
          About this track
        </h2>
        <p className="jb-about-lede">
          <strong>{copy.heading}</strong> {copy.lede}
        </p>
        {track?.description ? <p>{track.description}</p> : null}
        {track?.caveat ? <p className="jb-fine">Note: {track.caveat}</p> : null}
        <p className="jb-fine">
          Visualization: NASA Scientific Visualization Studio
          {track?.svs_url ? (
            <>
              , <a href={track.svs_url} target="_blank" rel="noopener noreferrer">SVS {track.id} (opens in a new tab)</a>
            </>
          ) : null}
          . The video has no sound. {track?.mix ? <a href={`/jukebox/tracks/${id}/${track.mix}`} download>Download the mix (MP3)</a> : null}
          {" · "}
          <Link href={`/compare?track=${id}`}>Compare versions of this track</Link>
        </p>
      </section>

      <div className="jb-deck jb-data">
        <div className="jb-deck-main">
          {track ? <Footage track={track} base={base} videoRef={video} sweepRef={sweepLine} /> : <div className="jb-video jb-frame" aria-hidden="true" />}

          <section className="jb-strip" aria-label="Score: the melody value over time">
            {track && prep ? (
              <>
                <DataTimeline
                  track={track}
                  prep={prep}
                  controls={loaded?.controls ?? null}
                  duration={duration}
                  store={store}
                  onSeek={(t) => seek(t, true)}
                  onToggle={() => void togglePlay()}
                  label="Playback position on the score"
                />
                <p className="jb-sr">{scoreSummary(track, prep)}</p>
              </>
            ) : null}
          </section>

          {track && prep ? <ScorePanel track={track} prep={prep} canvasRef={canvas} onSeek={(t) => seek(t, true)} /> : null}
        </div>

        <aside className="jb-side" aria-label="Player controls">
          <div className="jb-side-transport">
            <button type="button" className="jb-play" onClick={() => void togglePlay()} disabled={!ready} aria-label={playing ? "Pause" : "Play"} aria-keyshortcuts="Space">
              {playing ? <Pause size={22} aria-hidden="true" /> : <Play size={22} aria-hidden="true" />}
              <span>{playing ? "Pause" : "Play"}</span>
            </button>
            <Clock store={store} duration={duration} />
          </div>
          <div className="jb-side-skips">
            <button type="button" className="jb-skip jb-skip-sm" onClick={() => seek(store.get() - 5, true)} disabled={!ready} aria-label="Back 5 seconds" aria-keyshortcuts="ArrowLeft" title="Back 5 seconds">
              <RotateCcw size={18} aria-hidden="true" />
            </button>
            <button type="button" className="jb-skip jb-skip-sm" onClick={() => seek(store.get() + 5, true)} disabled={!ready} aria-label="Forward 5 seconds" aria-keyshortcuts="ArrowRight" title="Forward 5 seconds">
              <RotateCw size={18} aria-hidden="true" />
            </button>
            <button type="button" className="jb-skip jb-skip-sm" onClick={() => stepNote(-1)} disabled={!ready || !prep?.lead.length} aria-label="Previous melody note" aria-keyshortcuts="Shift+ArrowLeft" title="Previous melody note">
              <ChevronLeft size={18} aria-hidden="true" />
            </button>
            <button type="button" className="jb-skip jb-skip-sm" onClick={() => stepNote(1)} disabled={!ready || !prep?.lead.length} aria-label="Next melody note" aria-keyshortcuts="Shift+ArrowRight" title="Next melody note">
              <ChevronRight size={18} aria-hidden="true" />
            </button>
          </div>
          <Position store={store} duration={duration} disabled={!ready} onSeek={(t) => seek(t)} label={track && hasDates(track) ? (t) => timeLabel(track, t) : undefined} />

          <p className="jb-status" role="status">
            {failed ? (
              <>This track did not load ({failed}).</>
            ) : status ? (
              status
            ) : (
              <span className="jb-keys">
                <kbd>Space</kbd> play · <kbd>1</kbd>–<kbd>7</kbd> mute ·{" "}
                <button type="button" className="jb-link" onClick={() => setHelpOpen(true)}>
                  all shortcuts <kbd>?</kbd>
                </button>
              </span>
            )}
          </p>

          <section className="jb-now-mini" aria-labelledby="now-data-h">
            <h2 id="now-data-h" className="jb-side-h">
              Right now
            </h2>
            {nowNote ? (
              <p className="jb-readout">
                <span className="jb-readout-value jb-code">{fmtValue(nowNote.value, "", prep?.range ?? 10)}</span>
                <span>
                  {units}
                  {trend ? `, ${trend}` : ""}
                  {dateNow ? <strong>{dateNow}</strong> : null}
                </span>
              </p>
            ) : (
              <p className="jb-readout-none">{prep && !prep.lead.length ? "No single melody: each series has its own voice." : "The melody has not started yet."}</p>
            )}
          </section>

          {track && prep ? <StemMixer track={track} present={prep.present} state={groups} onChange={changeGroup} compact /> : null}

          <div className="jb-tools jb-side-tools" role="group" aria-label="Listening tools">
            <button type="button" className="jb-chip-btn" onClick={toggleBinaural} disabled={!ready} aria-pressed={binaural} aria-keyshortcuts="H">
              Headphones 3D <kbd>H</kbd>
            </button>
            <button type="button" className="jb-chip-btn" onClick={togglePolarity} disabled={!ready || !prep?.pm} aria-pressed={polarity < 0} aria-keyshortcuts="P">
              Flip polarity <kbd>P</kbd>
            </button>
            <button type="button" className="jb-chip-btn" onClick={() => void legend()} disabled={!ready} aria-keyshortcuts="L">
              Legend <kbd>L</kbd>
            </button>
            <button type="button" className="jb-chip-btn" onClick={sayValue} disabled={!ready} aria-keyshortcuts="V">
              Say value <kbd>V</kbd>
            </button>
            <button type="button" className="jb-chip-btn" onClick={() => void sweep()} disabled={!ready || !sweeps.length} aria-keyshortcuts="S">
              Sweep <kbd>S</kbd>
            </button>
            <button type="button" className="jb-chip-btn" onClick={() => void overview()} disabled={!ready || !prep?.lead.length} aria-keyshortcuts="Shift+S">
              Overview <kbd>⇧S</kbd>
            </button>
            <button type="button" className="jb-chip-btn" onClick={toggleMuteAll} disabled={!ready} aria-pressed={allMuted} aria-keyshortcuts="M">
              Mute all <kbd>M</kbd>
            </button>
            <button type="button" className="jb-chip-btn" onClick={() => setHelpOpen(true)} aria-keyshortcuts="Shift+Slash" aria-haspopup="dialog">
              Help <kbd>?</kbd>
            </button>
          </div>

          <details className="jb-details jb-side-more">
            <summary>Announcements and motion</summary>
            <div className="jb-settings">
              <label className="jb-select">
                <span>Announce while playing</span>
                <select value={cadence} onChange={(e) => setCadence(e.target.value as Cadence)}>
                  <option value="0">Off</option>
                  <option value="5">Every 5 seconds</option>
                  <option value="10">Every 10 seconds</option>
                  <option value="note">Each melody note (at most every 1.5 s)</option>
                </select>
              </label>
              <label className="jb-narrate">
                <input type="checkbox" checked={voice} disabled={!speechOk} onChange={(e) => setVoice(e.target.checked)} />
                <span>Also speak them with the browser voice</span>
              </label>
              {track?.video_url || track?.video_local ? (
                <label className="jb-narrate">
                  <input type="checkbox" checked={motion} onChange={(e) => setMotionPref(e.target.checked)} />
                  <span>Play the video (turn off to reduce motion)</span>
                </label>
              ) : null}
            </div>
          </details>

          {fallback ? (
            <audio controls preload="none" src={fallback} className="jb-fallback">
              <a href={fallback}>Download the mix</a>
            </audio>
          ) : null}
        </aside>
      </div>

      {track && prep ? (
        <div className="jb-data-more">
          <LegendPanel track={track} prep={prep} onLegend={() => void legend()} />
        </div>
      ) : null}

      <HelpDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
      <div className="jb-sr" aria-live="polite" aria-atomic="true">
        {live}
      </div>
      <div className="jb-sr" role="status" aria-atomic="true">
        {said}
      </div>
    </>
  );
}
