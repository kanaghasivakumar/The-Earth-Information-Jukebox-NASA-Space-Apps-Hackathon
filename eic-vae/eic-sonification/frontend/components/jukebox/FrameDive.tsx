"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { pipeline } from "@/lib/pipeline";
import { spriteStyle } from "@/lib/sprite";
import { mapX, mapY } from "@/lib/mapxy";
import { REGION_COLORS, fmt, regionLetter } from "@/lib/jukebox";

const { frames, map, lat, movement, path, span, notes, times, cols, rows } = pipeline;
const REVEAL_HZ = 30;
const REVEAL_MS = 3200;
const STEP_MS = 5200;
const END_HOLD_MS = 4500;
const [MELODY_DIM, HARMONY_DIM] = pipeline.dims;
const ambient = notes.ambient;
const pitchLo = Math.min(...ambient.map((n) => n[2])) - 1;
const pitchHi = Math.max(...ambient.map((n) => n[2])) + 1;

const STEPS = [
  { title: "A moment", text: "Five times a second, the model looks at a frame and the one just before it." },
  { title: "Pixels and movement", text: "It shrinks the picture to 64 × 64 pixels and measures which way each patch moved." },
  { title: "Eight numbers", text: "An encoder squeezes about 20,000 values down to eight numbers. To check that is enough, a decoder rebuilds the picture from only those eight. Slide to compare." },
  { title: "A dot on the map", text: "The numbers place the moment on a map. Moments that look alike land close together." },
  { title: "Numbers become notes", text: "Three things in the numbers steer the music: how fast they change, which two vary most, and which pattern group is back." },
] as const;

const clamp = (v: number) => Math.min(1, Math.max(0, v));

export function FrameDive() {
  const [fi, setFi] = useState(1);
  const [step, setStep] = useState(0);
  const [p, setP] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [note, setNote] = useState("");
  const frame = frames[fi];
  // the comparison slider starts on the chosen moment and resets whenever the moment changes
  const [slide, setSlide] = useState<{ fi: number; idx: number } | null>(null);
  const idx = slide && slide.fi === fi ? slide.idx : frames[fi].gi;
  const reduce = useRef(false);

  useEffect(() => {
    reduce.current = matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, []);

  // a slow reveal on every step change
  useEffect(() => {
    if (reduce.current) {
      setP(1);
      return;
    }
    let raf = 0;
    const start = performance.now();
    setP(0);
    let shown = -1;
    const tick = (now: number) => {
      const v = Math.min(1, (now - start) / REVEAL_MS);
      const key = v >= 1 ? REVEAL_HZ * 1000 : Math.floor(v * REVEAL_HZ * (REVEAL_MS / 1000));
      if (key !== shown) {
        shown = key;
        setP(v);
      }
      if (v < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [step, fi]);

  useEffect(() => {
    if (!playing) return;
    if (step >= STEPS.length - 1) {
      const id = setTimeout(() => setPlaying(false), END_HOLD_MS);
      return () => clearTimeout(id);
    }
    const id = setTimeout(() => setStep((s) => s + 1), STEP_MS);
    return () => clearTimeout(id);
  }, [playing, step]);

  // announce the step being played (state adjusted during render, not in an effect)
  const stepNote = playing ? `Step ${step + 1} of ${STEPS.length}: ${STEPS[step].title}. ${STEPS[step].text}` : null;
  if (stepNote !== null && stepNote !== note) setNote(stepNote);

  const e = 1 - Math.pow(1 - p, 3);
  const flow = frame.flow16;
  const scale = useMemo(() => {
    const all = flow.flat(2).map(Math.abs).sort((a, b) => a - b);
    return all[Math.floor(all.length * 0.97)] || 1;
  }, [flow]);

  return (
    <div className="jb-dive">
      <div className="jb-dive-bar">
        <button
          type="button"
          className="jb-play"
          aria-label={playing ? "Pause" : "Play the steps"}
          onClick={() => {
            if (!playing && step >= STEPS.length - 1) setStep(0);
            setPlaying((v) => !v);
          }}
        >
          {playing ? <Pause size={20} aria-hidden="true" /> : <Play size={20} aria-hidden="true" />}
          <span>{playing ? "Pause" : "Play"}</span>
        </button>
        <button type="button" className="jb-skip" onClick={() => { setPlaying(false); setStep((s) => Math.max(0, s - 1)); }} disabled={step === 0} aria-label="Previous step">
          <ChevronLeft size={18} aria-hidden="true" />
          <span>Back</span>
        </button>
        <button type="button" className="jb-skip" onClick={() => { setPlaying(false); setStep((s) => Math.min(STEPS.length - 1, s + 1)); }} disabled={step === STEPS.length - 1} aria-label="Next step">
          <span>Next</span>
          <ChevronRight size={18} aria-hidden="true" />
        </button>
        <div role="radiogroup" aria-label="Which moment to follow" className="jb-pipe-styles">
          {frames.map((f, i) => (
            <button key={f.img} type="button" role="radio" aria-checked={i === fi} className="jb-chip-btn" onClick={() => setFi(i)}>
              <span className="jb-code">{fmt(f.t)}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="jb-dive-body">
        <ol className="jb-dive-steps">
          {STEPS.map((s, i) => (
            <li key={s.title}>
              <button type="button" aria-current={i === step ? "step" : undefined} className={i === step ? "on" : i < step ? "done" : ""} onClick={() => { setPlaying(false); setStep(i); }}>
                <span className="jb-code">{i + 1}</span>
                {s.title}
              </button>
            </li>
          ))}
        </ol>

        <div className="jb-dive-stage">
          <div className="jb-dive-visual" key={`${step}-${fi}`}>
            {step === 0 && (
              <div className="jb-dive-pair">
                <figure style={{ opacity: clamp(e * 2), transform: `translateX(${(1 - clamp(e * 2)) * 40}px)` }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`/jukebox/pipeline/q${fi}.png`} alt="" width={64} height={64} className="jb-dive-pix soft" />
                  <figcaption className="jb-code">{(frame.t - 0.2).toFixed(1)}s</figcaption>
                </figure>
                <span className="jb-dive-op" aria-hidden="true">→</span>
                <figure style={{ opacity: clamp(e * 2 - 0.5), transform: `scale(${0.92 + clamp(e * 2 - 0.5) * 0.08})` }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={frame.img} alt="The hurricane-season globe at this moment" width={200} height={200} className="jb-dive-pix big" />
                  <figcaption className="jb-code">{frame.t.toFixed(1)}s</figcaption>
                </figure>
              </div>
            )}

            {step === 1 && (
              <figure className="jb-dive-flow">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={frame.img} alt="" width={200} height={200} className="jb-dive-pix pixel" style={{ opacity: 1 - clamp(e * 3) }} />
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`/jukebox/pipeline/p${fi}.png`} alt="The same moment shrunk to 64 by 64 pixels" width={64} height={64} className="jb-dive-pix pixel top" style={{ opacity: clamp(e * 3), filter: `brightness(${1 - clamp(e * 3 - 1) * 0.4})` }} />
                <svg viewBox="0 0 16 16" role="img" aria-label="Arrows showing which way each patch of the picture moved">
                  <defs>
                    <marker id="ah" viewBox="0 0 4 4" refX="3" refY="2" markerWidth="2.4" markerHeight="2.4" orient="auto">
                      <path d="M0 0 L4 2 L0 4 z" fill="var(--accent)" />
                    </marker>
                  </defs>
                  {flow[0].flatMap((row, y) =>
                    row.map((_, x) => {
                      const dx = (flow[0][y][x] / scale) * 1.4;
                      const dy = (flow[1][y][x] / scale) * 1.4;
                      const m = Math.hypot(dx, dy);
                      if (m < 0.12) return null;
                      return (
                        <line
                          key={`${x}-${y}`}
                          x1={x + 0.5 - dx / 2}
                          y1={y + 0.5 - dy / 2}
                          x2={x + 0.5 + dx / 2}
                          y2={y + 0.5 + dy / 2}
                          stroke="var(--accent)"
                          strokeWidth={0.13}
                          strokeLinecap="round"
                          markerEnd="url(#ah)"
                          style={{ animation: "jb-arrow-in 0.7s ease-out both", animationDelay: `${y * 90}ms`, ["--o" as string]: String(Math.min(1, 0.35 + m)) }}
                        />
                      );
                    }),
                  )}
                </svg>
              </figure>
            )}

            {step === 2 && (
              <div className="jb-vae">
                <div className="jb-vae-row">
                  <figure style={{ opacity: clamp(e * 5), transform: `translateX(${(1 - clamp(e * 5)) * -16}px)` }}>
                    <span className="jb-vae-img" role="img" aria-label="The real picture at this moment" style={spriteStyle("/jukebox/pipeline/sheet_orig.png", idx, cols, rows)} data-i={idx} />
                    <figcaption>What the model sees</figcaption>
                  </figure>
                  <div className="jb-vae-net enc" style={{ opacity: clamp(e * 5 - 1) }}>
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                      <path d="M0 4 L100 34 L100 66 L0 96 Z" />
                    </svg>
                    <span>Encoder</span>
                  </div>
                  <figure className="jb-vae-z" style={{ opacity: clamp(e * 5 - 2) }}>
                    <div className="jb-dive-eightbars" role="img" aria-label="The eight numbers at this moment">
                      {lat[idx].map((v, i) => (
                        <span key={i} className="jb-pipe-bar-track">
                          <span className="jb-pipe-bar-fill" style={{ height: `${Math.max(4, v * 100)}%`, transition: "none" }} />
                        </span>
                      ))}
                    </div>
                    <figcaption>8 numbers</figcaption>
                  </figure>
                  <div className="jb-vae-net dec" style={{ opacity: clamp(e * 5 - 3) }}>
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                      <path d="M0 34 L100 4 L100 96 L0 66 Z" />
                    </svg>
                    <span>Decoder</span>
                  </div>
                  <figure style={{ opacity: clamp(e * 5 - 4), transform: `translateX(${(1 - clamp(e * 5 - 4)) * 16}px)` }}>
                    <span className="jb-vae-img" role="img" aria-label="The picture the decoder rebuilds from only the eight numbers" style={spriteStyle("/jukebox/pipeline/sheet_recon.png", idx, cols, rows)} />
                    <figcaption>Rebuilt from only 8 numbers</figcaption>
                  </figure>
                </div>
                <label className="jb-vae-slider">
                  <span className="jb-code">{fmt(times[idx])}</span>
                  <input type="range" min={0} max={lat.length - 1} value={idx} onChange={(ev) => setSlide({ fi, idx: Number(ev.target.value) })} aria-label="Move through the thirty seconds" aria-valuetext={`${fmt(times[idx])} in the full piece`} />
                </label>
              </div>
            )}

            {step === 3 && (
              <div className="jb-dive-mapwrap">
                <svg className="jb-pipe-map" viewBox="0 0 200 150" role="img" aria-label={`Map of every moment. This one is in pattern group ${regionLetter(frame.region)}.`}>
                  {map.map(([x, y, r], i) => (
                    <circle key={i} cx={mapX(x)} cy={mapY(y)} r={1.8} fill={REGION_COLORS[r]} opacity={0.55 * clamp(e * 2)} />
                  ))}
                  <circle cx={mapX(frame.xy[0])} cy={mapY(frame.xy[1])} r={22 * (1 - clamp(e * 1.6)) + 6} fill="none" stroke="var(--accent)" strokeWidth={1.2} opacity={1 - clamp(e * 1.6)} />
                  <circle cx={mapX(frame.xy[0])} cy={mapY(frame.xy[1])} r={6 * clamp(e * 2 - 0.6)} fill="var(--accent)" />
                </svg>
                <p className="jb-pipe-group" style={{ opacity: clamp(e * 2 - 0.8) }}>
                  <span className={`jb-chip r${frame.region}`} aria-hidden="true">
                    {regionLetter(frame.region)}
                  </span>
                  <span>Pattern group {regionLetter(frame.region)}</span>
                </p>
              </div>
            )}
            {step === 4 && (
              <div className="jb-dive-music">
                <div className="jb-dive-eightbars sel" role="img" aria-label="Eight numbers, with the two that vary most marked as melody and harmony">
                  {frame.latent.map((v, i) => (
                    <span key={i} className={`jb-pipe-bar-track${i === MELODY_DIM || i === HARMONY_DIM ? " hot" : ""}`}>
                      <span className="jb-pipe-bar-fill" style={{ height: `${Math.max(4, v * 100)}%` }} />
                      {i === MELODY_DIM ? <em>melody</em> : i === HARMONY_DIM ? <em>harmony</em> : null}
                    </span>
                  ))}
                </div>
                <div className="jb-dive-rows">
                  {[
                    { label: "How busy and loud", sub: "how fast the numbers change", series: movement, area: true },
                    { label: "Shape of the melody", sub: "one of the two that vary most", series: lat.map((r) => r[MELODY_DIM]), area: false },
                    { label: "Colour of the chords", sub: "the other one", series: lat.map((r) => r[HARMONY_DIM]), area: false },
                  ].map((r) => {
                    const n = r.series.length;
                    const upto = Math.max(2, Math.round(clamp(e * 1.15) * n));
                    const pts = r.series.slice(0, upto).map((v, i) => `${((i / (n - 1)) * 300).toFixed(1)},${(34 - v * 30).toFixed(1)}`);
                    return (
                      <div key={r.label} className="jb-dive-row">
                        <span>
                          <strong>{r.label}</strong>
                          <small>{r.sub}</small>
                        </span>
                        <svg viewBox="0 0 300 36" preserveAspectRatio="none" role="img" aria-label={r.label}>
                          {r.area ? <path d={`M0,36 L${pts.join(" L")} L${((upto - 1) / (n - 1)) * 300},36 Z`} fill="rgb(255 107 74 / 0.4)" /> : null}
                          <polyline points={pts.join(" ")} fill="none" stroke={r.area ? "var(--accent)" : "#7cc4ff"} strokeWidth={1.8} strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
                        </svg>
                      </div>
                    );
                  })}
                  <div className="jb-dive-row">
                    <span>
                      <strong>Which motif returns</strong>
                      <small>the pattern group</small>
                    </span>
                    <div className="jb-dive-groups" aria-label="Pattern groups across the tour">
                      {path.map(([, , g], i) => (
                        <i key={i} style={{ background: REGION_COLORS[g], opacity: i / path.length <= clamp(e * 1.15) ? 1 : 0.15 }} />
                      ))}
                    </div>
                  </div>
                </div>
                <svg className="jb-dive-rollmini" viewBox="0 0 300 120" role="img" aria-label="The notes the composer wrote for these 30 seconds">
                  {ambient.map(([st, d, pitch], i) => (
                    <rect key={i} x={(st / span) * 300} y={112 - ((pitch - pitchLo) / (pitchHi - pitchLo)) * 104} width={Math.max(1.6, (d / span) * 300)} height={3.4} rx={1} fill={st / span <= clamp(e * 1.15) ? "var(--accent)" : "rgb(124 196 255 / 0.5)"} />
                  ))}
                </svg>
              </div>
            )}
          </div>
          <p className="jb-dive-text">{STEPS[step].text}</p>
        </div>
      </div>

      <div className="jb-sr" aria-live="polite">
        {note}
      </div>
    </div>
  );
}
