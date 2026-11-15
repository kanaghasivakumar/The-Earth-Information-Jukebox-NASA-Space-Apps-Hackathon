"use client";

import { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import { pipeline, type Note } from "@/lib/pipeline";
import { spriteStyle } from "@/lib/sprite";
import { mapX, mapY } from "@/lib/mapxy";
import { REGION_COLORS, STYLE_COPY, fmt, regionLetter } from "@/lib/jukebox";

const PIPE_HZ = 30;
const STYLES = ["ambient", "cinematic", "classical", "jazz", "electronic"] as const;
const { span, segs, frames, map, movement, path, lat, dt, cols, rows } = pipeline;
const notesByStyle = pipeline.notes;
const SEGLEN = span / segs.length;
const segOf = (g: number) => Math.max(0, Math.min(segs.length - 1, Math.floor(g / SEGLEN)));
/** tour clock -> seconds in the full piece */
const srcOf = (g: number) => segs[segOf(g)][0] + (g - segOf(g) * SEGLEN);

/** Static map: 445 dots that never change while the walkthrough plays. */
const MapDots = memo(function MapDots({ points }: { points: [number, number, number][] }) {
  return (
    <>
      {points.map(([x, y, r], i) => (
        <circle key={i} cx={mapX(x)} cy={mapY(y)} r={1.8} fill={REGION_COLORS[r]} opacity={0.4} />
      ))}
    </>
  );
});

const MovementArea = memo(function MovementArea({ values }: { values: number[] }) {
  const d = useMemo(
    () => `M0,46 L${values.map((v, i) => `${((i / (values.length - 1)) * 420).toFixed(1)},${(44 - v * 40).toFixed(1)}`).join(" L")} L420,46 Z`,
    [values],
  );
  return <path d={d} fill="rgb(124 196 255 / 0.35)" />;
});

const NoteLayer = memo(function NoteLayer({ notes, lo, hi, W, H, span, fill }: { notes: Note[]; lo: number; hi: number; W: number; H: number; span: number; fill: string }) {
  return (
    <>
      {notes.map(([s, d, p], i) => (
        <rect key={i} x={(s * W) / span} y={H - 10 - ((p - lo) / (hi - lo)) * (H - 20) - 2.5} width={Math.max(2, (d * W) / span)} height={5} rx={1} fill={fill} />
      ))}
    </>
  );
});

export function Pipeline() {
  const audio = useRef<HTMLAudioElement>(null);
  const [style, setStyle] = useState<(typeof STYLES)[number]>("ambient");
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [sound, setSound] = useState(true);
  const [note, setNote] = useState("");
  const tRef = useRef(0);
  const lastK = useRef(-1);
  const shownKey = useRef(-1);
  const clipId = `played-${useId().replace(/:/g, "")}`;

  const k = frames.reduce((acc, f, i) => (f.gi * dt <= t + 0.05 ? i : acc), 0);
  const frame = frames[k];
  const seg = segOf(t);
  // the real path through the map, interpolated to the playhead (never across a jump between clips)
  const pf = Math.min(path.length - 1.001, Math.max(0, t / dt));
  const pi = Math.floor(pf);
  const sameClip = path[pi][3] === path[pi + 1][3];
  const pr = sameClip ? pf - pi : 0;
  const lerp = (a: number, b: number) => a + (b - a) * pr;
  const here: [number, number] = [lerp(path[pi][0], path[pi + 1][0]), lerp(path[pi][1], path[pi + 1][1])];
  const group = path[Math.round(pf)][2];
  const nums = lat[pi].map((v, i) => lerp(v, lat[pi + 1][i]));
  const px = (x: number) => mapX(x).toFixed(1);
  const py = (y: number) => mapY(y).toFixed(1);
  const idx = Math.min(lat.length - 1, Math.max(0, Math.round(t / dt)));
  const mv = movement[Math.min(movement.length - 1, pi)];
  // trail: one short segment per sample, coloured by the pattern group it was in, fading with age
  const TAIL = 45;
  const tail = path.slice(Math.max(0, pi - TAIL), pi + 1).map((p, i, arr) => ({ x: p[0], y: p[1], g: p[2], age: (arr.length - 1 - i) / TAIL }));
  tail.push({ x: here[0], y: here[1], g: group, age: 0 });
  const notes = notesByStyle[style];
  const pitches = useMemo(() => {
    const p = notes.map((n) => n[2]);
    return { lo: Math.min(...p) - 1, hi: Math.max(...p) + 1 };
  }, [notes]);
  const inSegment = notes.filter((n) => n[0] >= frame.gi * dt && n[0] < frame.gi * dt + 4).length;

  // one clock: the audio when sound is on, otherwise wall time. Audio jumps between the three clips.
  useEffect(() => {
    if (!playing) return;
    const a = audio.current;
    const useAudio = sound && a;
    if (useAudio) {
      a.currentTime = srcOf(tRef.current);
      a.play().catch(() => setSound(false));
    }
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      let g = tRef.current;
      const s0 = segOf(g);
      if (useAudio && a && !a.paused && a.currentTime >= segs[s0][0] - 0.01) {
        const local = a.currentTime - segs[s0][0];
        g = local >= SEGLEN ? (s0 + 1) * SEGLEN : s0 * SEGLEN + local;
      } else {
        g += (now - last) / 1000;
      }
      last = now;
      if (g >= span) {
        a?.pause();
        tRef.current = span - 0.001;
        setT(span - 0.001);
        setPlaying(false);
        return;
      }
      if (useAudio && a && segOf(g) !== s0) a.currentTime = segs[segOf(g)][0];
      tRef.current = g;
      const key = Math.floor(g * PIPE_HZ);
      if (key !== shownKey.current) {
        shownKey.current = key;
        setT(g);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      a?.pause();
    };
  }, [playing, sound, style]);

  useEffect(() => {
    if (!playing || k === lastK.current) return;
    lastK.current = k;
    setNote(
      `${fmt(frame.t)} in the full piece: this frame goes in. The model describes it with eight numbers and places it in pattern group ${regionLetter(frame.region)}. The ${STYLE_COPY[style].name} score plays ${inSegment} notes over the next four seconds.`,
    );
  }, [k, playing, frame, style, inSegment, seg]);

  const jump = useCallback((g: number) => {
    tRef.current = g;
    setT(g);
    if (audio.current) audio.current.currentTime = srcOf(g);
  }, []);

  const toggle = () => {
    if (!playing && tRef.current >= span - 0.01) jump(0);
    setPlaying((p) => !p);
  };
  const replay = () => {
    jump(0);
    lastK.current = -1;
    setPlaying(true);
  };

  const W = 420;
  const H = 190;
  const justPlayed = notes.filter(([s]) => s <= t && s > t - 0.5);

  return (
    <div className="jb-pipe">
      <audio ref={audio} src={`/jukebox/audio/${style}.mp3`} preload="none" />

      <div className="jb-pipe-bar">
        <button type="button" className="jb-play" onClick={toggle} aria-label={playing ? "Pause walkthrough" : "Play walkthrough"}>
          {playing ? <Pause size={20} aria-hidden="true" /> : <Play size={20} aria-hidden="true" />}
          <span>{playing ? "Pause" : t > 0.01 ? "Resume" : "Play"}</span>
        </button>
        <button type="button" className="jb-skip" onClick={replay} aria-label="Replay from the start">
          <RotateCcw size={18} aria-hidden="true" />
          <span>Replay</span>
        </button>
        <label className="jb-pipe-sound">
          <input type="checkbox" checked={sound} onChange={(e) => setSound(e.target.checked)} />
          <span>Play the sound</span>
        </label>
        <div role="radiogroup" aria-label="Music style for this walkthrough" className="jb-pipe-styles">
          {STYLES.map((s) => (
            <button key={s} type="button" role="radio" aria-checked={s === style} className="jb-chip-btn" onClick={() => setStyle(s)}>
              {STYLE_COPY[s].name}
            </button>
          ))}
        </div>
      </div>

      <div className="jb-pipe-stage">
        {/* 1. footage */}
        <section className="jb-pipe-col" aria-labelledby="pp1">
          <h3 id="pp1">1. The footage</h3>
          <p className="jb-pipe-sub">The real frames, five a second.</p>
          <div className="jb-flip">
            <span className="jb-vae-img jb-flip-img" role="img" aria-label="The real picture at this moment" style={spriteStyle("/jukebox/pipeline/sheet_orig.png", idx, cols, rows)} />
            <span className="jb-flip-cap">What the model sees</span>
            <span className="jb-vae-img jb-flip-img" role="img" aria-label="The picture rebuilt from only the eight numbers" style={spriteStyle("/jukebox/pipeline/sheet_recon.png", idx, cols, rows)} />
            <span className="jb-flip-cap">Rebuilt from 8 numbers</span>
          </div>
        </section>

        <div className={`jb-pipe-flow${playing ? " run" : ""}`} aria-hidden="true">
          <svg viewBox="0 0 40 100" preserveAspectRatio="none">
            <path d="M0 50 H40" />
            <path d="M30 40 L40 50 L30 60" />
          </svg>
        </div>

        {/* 2. latent */}
        <section className="jb-pipe-col" aria-labelledby="pp2">
          <h3 id="pp2">2. What the model learned</h3>
          <p className="jb-pipe-sub">Eight numbers, changing as the picture changes.</p>
          <div className="jb-pipe-bars" role="img" aria-label={`Eight learned numbers at ${fmt(srcOf(t))}`}>
            {nums.map((v, i) => (
              <div key={i} className="jb-pipe-bar-col">
                <span className="jb-pipe-bar-track">
                  <span className="jb-pipe-bar-fill" style={{ height: `${Math.max(4, v * 100)}%`, transition: "none" }} />
                </span>
                <span className="jb-code">{v.toFixed(2).slice(1)}</span>
              </div>
            ))}
          </div>
          <p className="jb-pipe-sub">The dot follows the picture through the map:</p>
          <svg className="jb-pipe-map" viewBox="0 0 200 150" role="img" aria-label={`Map of every moment. The playhead is in pattern group ${regionLetter(group)}.`}>
            <MapDots points={map} />
            {tail.slice(1).map((p, i) => {
              const q = tail[i];
              const rec = 1 - p.age;
              return <line key={i} x1={px(q.x)} y1={py(q.y)} x2={px(p.x)} y2={py(p.y)} stroke={REGION_COLORS[p.g]} strokeWidth={0.8 + rec * 2.6} strokeLinecap="round" opacity={0.15 + rec * 0.85} />;
            })}
            <circle cx={mapX(here[0])} cy={mapY(here[1])} r={6 + mv * 16} fill="none" stroke="var(--accent)" strokeWidth={1.2} opacity={0.55} />
            <circle cx={mapX(here[0])} cy={mapY(here[1])} r={5} fill="var(--accent)" />
            <text x={100} y={148} textAnchor="middle" className="jb-pipe-axis">melody number →</text>
            <text x={4} y={75} textAnchor="middle" transform="rotate(-90 4 75)" className="jb-pipe-axis">chord number →</text>
          </svg>
          <p className="jb-pipe-group">
            <span className={`jb-chip r${group}`} aria-hidden="true">
              {regionLetter(group)}
            </span>
            <span>Pattern group {regionLetter(group)}</span>
          </p>
          <p className="jb-pipe-explain">
            <strong>What the dot is.</strong> Its position is two of the eight numbers: the two that vary most. Across is the one that steers the melody, up is the one that steers the chords. Dots that sit close together are moments that look alike, and the colours are the pattern groups. The farther the dot jumps, the faster the picture is changing, and the busier and louder the music gets.
          </p>
        </section>

        <div className={`jb-pipe-flow${playing ? " run" : ""}`} aria-hidden="true">
          <svg viewBox="0 0 40 100" preserveAspectRatio="none">
            <path d="M0 50 H40" />
            <path d="M30 40 L40 50 L30 60" />
          </svg>
        </div>

        {/* 3. music */}
        <section className="jb-pipe-col" aria-labelledby="pp3">
          <h3 id="pp3">3. The music</h3>
          <p className="jb-pipe-sub">
            {STYLE_COPY[style].name}: {notes.length} notes across these 30 seconds.
          </p>
          <p className="jb-pipe-sub jb-pipe-lane">How much the picture is changing</p>
          <svg className="jb-pipe-move" viewBox="0 0 420 46" preserveAspectRatio="none" role="img" aria-label="How quickly the picture changed over these 30 seconds">
            <MovementArea values={movement} />
            {segs.slice(1).map((_, i) => (
              <line key={i} x1={((i + 1) * SEGLEN * 420) / span} x2={((i + 1) * SEGLEN * 420) / span} y1={0} y2={46} stroke="rgb(232 238 242 / 0.4)" strokeDasharray="3 3" />
            ))}
            <line x1={(t * 420) / span} x2={(t * 420) / span} y1={0} y2={46} stroke="var(--accent)" strokeWidth={1.5} />
          </svg>
          <p className="jb-pipe-sub jb-pipe-lane">The notes the composer wrote</p>
          <svg className="jb-pipe-roll" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Piano roll of the ${STYLE_COPY[style].name} score for these 30 seconds`}>
            {segs.map(([start], i) => (
              <g key={i}>
                <rect x={(i * SEGLEN * W) / span} y={0} width={(SEGLEN * W) / span} height={H} fill={i === seg ? "rgb(255 255 255 / 0.07)" : "none"} stroke="rgb(124 196 255 / 0.2)" strokeWidth={0.5} />
                <text x={(i * SEGLEN * W) / span + 4} y={H - 4} className="jb-pipe-axis">from {fmt(start)}</text>
              </g>
            ))}
            <defs>
              <clipPath id={clipId}>
                <rect x={0} y={0} width={(t * W) / span} height={H} />
              </clipPath>
            </defs>
            <NoteLayer notes={notes} lo={pitches.lo} hi={pitches.hi} W={W} H={H} span={span} fill="rgb(124 196 255 / 0.55)" />
            <g clipPath={`url(#${clipId})`}>
              <NoteLayer notes={notes} lo={pitches.lo} hi={pitches.hi} W={W} H={H} span={span} fill="var(--accent)" />
            </g>
            {justPlayed.map(([s, d, p], i) => (
              <rect key={i} x={(s * W) / span} y={H - 10 - ((p - pitches.lo) / (pitches.hi - pitches.lo)) * (H - 20) - 2.5} width={Math.max(2, (d * W) / span)} height={5} rx={1} fill="#fff" />
            ))}
            <line x1={(t * W) / span} x2={(t * W) / span} y1={0} y2={H} stroke="var(--accent)" strokeWidth={1.5} />
          </svg>
          <p className="jb-pipe-sub">
            When the curve above rises, the composer writes busier, louder music. The ring around the dot grows when the picture changes fast.
          </p>
        </section>
      </div>

      <p className="jb-pipe-time jb-code" aria-hidden="true">
        {fmt(srcOf(t))} in the full piece
      </p>
      <div className="jb-sr" aria-live="polite">
        {note}
      </div>
    </div>
  );
}
