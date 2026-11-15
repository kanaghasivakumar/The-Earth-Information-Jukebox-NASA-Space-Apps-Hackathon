/**
 * Web Audio engines (browser only).
 *
 * Engine: the data player. Every stem starts on the same AudioContext tick, so stems stay sample-synced;
 * each stem group has a gain (mute, volume), a dry path and an HRTF path (headphone 3D); the shuffled
 * control stem plays alongside the melody and the two swap by gain (A/B); polarity flip re-synthesizes
 * the melody live from the timeline.
 *
 * MultiPlayer: the Compare view. Every version is decoded and started together; choosing a version only
 * moves gains, so the switch lands on the same sample.
 */
import { midiHz } from "./music";
import { GROUPS, type DataTrack, type Group, type TimelineNote } from "./types";

// Binaural home positions [azimuth°, elevation°] (+ right, + up). Note pan moves a group by up to ±75°.
// Earth sounds to the left, human sounds to the right, living sounds above, data melody in front.
const HOME: Record<Group, [number, number]> = {
  melody: [0, 0],
  bands: [0, 15],
  events: [0, 0],
  harmony: [-25, 0],
  geophony: [-55, 0],
  anthrophony: [55, 0],
  biophony: [0, 40],
};
const RAMP = 0.04;
const MASTER = 0.62; // the melody and bands sum to about 1.6× a stem

type Ctor = typeof AudioContext;
export const audioSupported = () =>
  typeof window !== "undefined" && !!(window.AudioContext || (window as unknown as { webkitAudioContext?: Ctor }).webkitAudioContext);

function newContext(): AudioContext {
  const AC = window.AudioContext || (window as unknown as { webkitAudioContext: Ctor }).webkitAudioContext;
  return new AC();
}

async function fetchDecode(ctx: AudioContext, url: string, signal?: AbortSignal): Promise<AudioBuffer> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const ab = await res.arrayBuffer();
  return new Promise((resolve, reject) => {
    const p = ctx.decodeAudioData(ab, resolve, reject);
    if (p && typeof p.then === "function") p.then(resolve, reject);
  });
}

interface GroupNode {
  gain: GainNode;
  dry: GainNode;
  wet: GainNode;
  panner: PannerNode;
  pan: number | null;
}

interface StemEntry {
  name: string;
  group: Group;
  buffer: AudioBuffer;
  gain: GainNode;
}

export interface GroupState {
  volume: number;
  muted: boolean;
}

export class Engine {
  ctx: AudioContext | null = null;
  private master!: GainNode;
  private duckGain!: GainNode;
  private ui!: GainNode;
  private groups = {} as Record<Group, GroupNode>;
  private stems: StemEntry[] = [];
  private control: (StemEntry & { replaces: Group }) | null = null;
  private sources: AudioBufferSourceNode[] = [];
  private liveVoices: OscillatorNode[] = [];
  private live: { notes: TimelineNote[]; pitchOf: (n: TimelineNote) => number; cursor: number | null; fresh: boolean } | null = null;
  private startedAt = 0;
  private loadToken = 0;
  playing = false;
  offset = 0;
  duration = 0;
  shuffled = false;
  liveLead = false;
  binaural = false;
  masterMuted = false;
  groupState = Object.fromEntries(GROUPS.map((g) => [g, { volume: 1, muted: false }])) as Record<Group, GroupState>;

  ensure(): AudioContext {
    if (this.ctx) return this.ctx;
    const ctx = (this.ctx = newContext());
    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -3;
    limiter.knee.value = 0;
    limiter.ratio.value = 20;
    limiter.attack.value = 0.003;
    limiter.release.value = 0.1;
    limiter.connect(ctx.destination);
    this.master = ctx.createGain();
    this.master.gain.value = this.masterMuted ? 0 : MASTER;
    this.master.connect(limiter);
    this.duckGain = ctx.createGain();
    this.duckGain.connect(this.master);
    this.ui = ctx.createGain();
    this.ui.gain.value = 0.9;
    this.ui.connect(this.master);
    for (const g of GROUPS) {
      const gain = ctx.createGain();
      const dry = ctx.createGain();
      const wet = ctx.createGain();
      wet.gain.value = 0;
      const panner = this.makePanner();
      gain.connect(dry).connect(this.duckGain);
      gain.connect(panner).connect(wet).connect(this.duckGain);
      this.groups[g] = { gain, dry, wet, panner, pan: null };
      this.setPannerAngle(panner, ...HOME[g]);
      const st = this.groupState[g];
      gain.gain.value = st.muted ? 0 : st.volume;
    }
    return ctx;
  }

  async resume() {
    const ctx = this.ensure();
    if (ctx.state !== "running") await ctx.resume();
  }

  private makePanner(): PannerNode {
    const p = this.ctx!.createPanner();
    p.panningModel = "HRTF";
    p.distanceModel = "inverse";
    p.refDistance = 1;
    p.rolloffFactor = 0;
    return p;
  }

  private setPannerAngle(p: PannerNode, azDeg: number, elDeg = 0) {
    const az = (azDeg * Math.PI) / 180;
    const el = (elDeg * Math.PI) / 180;
    const x = Math.sin(az) * Math.cos(el);
    const y = Math.sin(el);
    const z = -Math.cos(az) * Math.cos(el);
    if (p.positionX) {
      const t = this.ctx!.currentTime;
      p.positionX.setTargetAtTime(x, t, 0.05);
      p.positionY.setTargetAtTime(y, t, 0.05);
      p.positionZ.setTargetAtTime(z, t, 0.05);
    } else (p as unknown as { setPosition: (x: number, y: number, z: number) => void }).setPosition(x, y, z);
  }

  /** Decode every stem (and the control) of a track. Resolves to false when a newer load superseded it. */
  async load(track: DataTrack, base: string, onProgress: (done: number, total: number) => void = () => {}): Promise<boolean> {
    const token = ++this.loadToken;
    this.pause();
    this.offset = 0;
    const ctx = this.ensure();
    const jobs = track.stems.map((s) => ({ name: s.name, group: s.group, url: base + s.file, isControl: false }));
    if (track.control?.file) jobs.push({ name: "__control", group: track.control.replaces ?? "melody", url: base + track.control.file, isControl: true });
    let done = 0;
    const decoded = await Promise.all(
      jobs.map(async (j) => {
        const buffer = await fetchDecode(ctx, j.url);
        onProgress(++done, jobs.length);
        return { ...j, buffer };
      }),
    );
    if (token !== this.loadToken) return false;
    for (const s of this.stems) s.gain.disconnect();
    this.control?.gain.disconnect();
    this.stems = [];
    this.control = null;
    for (const j of decoded) {
      const gain = ctx.createGain();
      const group = GROUPS.includes(j.group) ? j.group : "melody";
      gain.connect(this.groups[group].gain);
      const entry = { name: j.name, group, buffer: j.buffer, gain };
      if (j.isControl) this.control = { ...entry, replaces: group };
      else this.stems.push(entry);
    }
    this.duration = Math.max(track.duration_s || 0, ...decoded.map((d) => d.buffer.duration));
    this.shuffled = false;
    this.setLiveLead(false);
    this.applyStemGains(true);
    return true;
  }

  get hasControl() {
    return !!this.control;
  }

  get time(): number {
    if (!this.playing || !this.ctx) return this.offset;
    return Math.min(this.duration, this.ctx.currentTime - this.startedAt);
  }

  async play() {
    await this.resume();
    const ctx = this.ctx!;
    if (this.playing) return;
    if (this.offset >= this.duration - 0.05) this.offset = 0;
    const when = ctx.currentTime + 0.03;
    this.startedAt = when - this.offset;
    const all = [...this.stems, ...(this.control ? [this.control] : [])];
    this.sources = all
      .filter((s) => this.offset < s.buffer.duration)
      .map((s) => {
        const src = ctx.createBufferSource();
        src.buffer = s.buffer;
        src.connect(s.gain);
        src.start(when, this.offset);
        return src;
      });
    this.playing = true;
    if (this.live) {
      this.live.cursor = this.offset;
      this.live.fresh = true;
    }
  }

  private stopSources() {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
      s.disconnect();
    }
    this.sources = [];
    for (const o of this.liveVoices) {
      try {
        o.stop();
      } catch {
        /* already stopped */
      }
    }
    this.liveVoices = [];
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.time;
    this.stopSources();
    this.playing = false;
  }

  seek(t: number) {
    const was = this.playing;
    if (was) this.pause();
    this.offset = Math.max(0, Math.min(this.duration, t));
    if (was) void this.play();
  }

  private ramp(param: AudioParam, v: number, immediate = false) {
    if (!this.ctx) return;
    if (immediate) param.value = v;
    else param.setTargetAtTime(v, this.ctx.currentTime, RAMP);
  }

  private applyStemGains(immediate = false) {
    const ctrlOn = this.shuffled && !!this.control;
    for (const s of this.stems) {
      const off = (ctrlOn && s.group === this.control!.replaces) || (this.liveLead && s.group === "melody");
      this.ramp(s.gain.gain, off ? 0 : 1, immediate);
    }
    if (this.control) this.ramp(this.control.gain.gain, ctrlOn ? 1 : 0, immediate);
  }

  setShuffled(on: boolean) {
    this.shuffled = on;
    this.applyStemGains();
  }

  setGroup(g: Group, { volume, muted }: Partial<GroupState> = {}) {
    const st = this.groupState[g];
    if (volume != null) st.volume = volume;
    if (muted != null) st.muted = muted;
    if (this.ctx) this.ramp(this.groups[g].gain.gain, st.muted ? 0 : st.volume);
  }

  setMasterMuted(m: boolean) {
    this.masterMuted = m;
    if (this.ctx) this.ramp(this.master.gain, m ? 0 : MASTER);
  }

  setBinaural(on: boolean) {
    this.binaural = on;
    if (!this.ctx) return;
    for (const g of GROUPS) {
      this.ramp(this.groups[g].dry.gain, on ? 0 : 1);
      this.ramp(this.groups[g].wet.gain, on ? 1.4 : 0); // HRTF output is quieter than the dry stereo
    }
  }

  /** Move a group's binaural source: pan −1 (west, left) to +1 (east, right), added to its home angle. */
  setGroupPan(g: Group, pan: number) {
    const grp = this.groups[g];
    if (!grp || grp.pan === pan) return;
    grp.pan = pan;
    const [az, el] = HOME[g];
    this.setPannerAngle(grp.panner, az + (pan || 0) * 75, el);
  }

  duck(level: number) {
    if (this.ctx) this.ramp(this.duckGain.gain, level);
  }

  /** One synthesized note on the UI bus (legend and sweep tones; ignores mutes) or a group bus. */
  tone(
    pitch: number,
    { when = 0, dur = 0.5, vel = 90, pan = 0, bus = "ui" as "ui" | Group, timbre = "pluck" as "pluck" | "flat" } = {},
  ): { end: number; oscs: OscillatorNode[] } {
    const ctx = this.ensure();
    const t0 = Math.max(ctx.currentTime + 0.01, when || 0);
    const d = Math.max(0.06, dur);
    const f = midiHz(pitch);
    const env = ctx.createGain();
    const peak = (vel / 127) * 0.32;
    const rel = timbre === "pluck" ? 0.3 : 0.15;
    env.gain.setValueAtTime(0, t0);
    env.gain.linearRampToValueAtTime(peak, t0 + 0.008);
    if (timbre === "pluck") env.gain.setTargetAtTime(peak * 0.35, t0 + 0.01, 0.18);
    env.gain.setValueAtTime(timbre === "pluck" ? peak * 0.35 : peak, t0 + d);
    env.gain.setTargetAtTime(0, t0 + d, rel / 3);
    const oscs = ([[1, "sine", 1], [2, "triangle", 0.35], [3, "sine", 0.12]] as const).map(([k, type, a]) => {
      const o = ctx.createOscillator();
      o.type = type;
      o.frequency.value = f * k;
      const g = ctx.createGain();
      g.gain.value = a;
      o.connect(g).connect(env);
      o.start(t0);
      o.stop(t0 + d + rel + 0.1);
      return o;
    });
    let out: AudioNode = env;
    if (bus === "ui") {
      if (this.binaural) {
        const p = this.makePanner();
        this.setPannerAngle(p, pan * 80, 0);
        env.connect(p);
        out = p;
      } else if (ctx.createStereoPanner) {
        const p = ctx.createStereoPanner();
        p.pan.value = pan;
        env.connect(p);
        out = p;
      }
      out.connect(this.ui);
    } else {
      env.connect(this.groups[bus]?.gain ?? this.ui);
    }
    oscs[0].onended = () => {
      env.disconnect();
      if (out !== env) out.disconnect();
    };
    return { end: t0 + d + rel, oscs };
  }

  /** Polarity flip: silence the recorded melody and play it live from the timeline instead. */
  setLiveLead(on: boolean, notes: TimelineNote[] = [], pitchOf: (n: TimelineNote) => number = (n) => n.pitch) {
    this.liveLead = on;
    for (const o of this.liveVoices) {
      try {
        o.stop();
      } catch {
        /* already stopped */
      }
    }
    this.liveVoices = [];
    this.live = on ? { notes, pitchOf, cursor: this.time, fresh: true } : null;
    this.applyStemGains();
  }

  /** Call every ~100 ms: schedules live melody notes inside a short look-ahead window. */
  tick(lookahead = 0.35) {
    if (!this.live || !this.playing) return;
    const L = this.live;
    const now = this.time;
    const until = now + lookahead;
    if (L.cursor == null || L.cursor < now - 0.25 || L.cursor > until + 1) {
      L.cursor = now;
      L.fresh = true;
    }
    for (const n of L.notes) {
      const inWindow = n.start >= L.cursor && n.start < until;
      const sounding = L.fresh && n.start < L.cursor && n.end > L.cursor;
      if (!inWindow && !sounding) continue;
      const start = Math.max(n.start, L.cursor);
      const { oscs } = this.tone(L.pitchOf(n), { when: this.startedAt + start, dur: n.end - start, vel: n.velocity, bus: "melody" });
      this.liveVoices.push(...oscs);
    }
    L.fresh = false;
    L.cursor = until;
    if (this.liveVoices.length > 300) this.liveVoices.splice(0, 150);
  }

  close() {
    this.loadToken++;
    this.pause();
    void this.ctx?.close().catch(() => {});
    this.ctx = null;
  }
}

/** Sample-synced versions for the Compare view. Choosing a version only moves gains. */
export class MultiPlayer {
  ctx: AudioContext | null = null;
  private out!: GainNode;
  private buffers: AudioBuffer[] = [];
  private gains: GainNode[] = [];
  private sources: AudioBufferSourceNode[] = [];
  private startedAt = 0;
  private token = 0;
  playing = false;
  offset = 0;
  duration = 0;
  selected = 0;

  async load(urls: string[], onProgress: (done: number, total: number) => void = () => {}): Promise<boolean> {
    const token = ++this.token;
    this.pause();
    this.offset = 0;
    if (!this.ctx) {
      this.ctx = newContext();
      this.out = this.ctx.createGain();
      this.out.connect(this.ctx.destination);
    }
    const ctx = this.ctx;
    let done = 0;
    const buffers = await Promise.all(
      urls.map(async (u) => {
        const b = await fetchDecode(ctx, u);
        onProgress(++done, urls.length);
        return b;
      }),
    );
    if (token !== this.token) return false;
    this.gains.forEach((g) => g.disconnect());
    this.buffers = buffers;
    this.gains = buffers.map(() => {
      const g = ctx.createGain();
      g.connect(this.out);
      return g;
    });
    this.duration = Math.max(...buffers.map((b) => b.duration));
    this.select(Math.min(this.selected, buffers.length - 1), true);
    return true;
  }

  get time(): number {
    if (!this.playing || !this.ctx) return this.offset;
    return Math.min(this.duration, this.ctx.currentTime - this.startedAt);
  }

  async play() {
    const ctx = this.ctx;
    if (!ctx || this.playing || !this.buffers.length) return;
    if (ctx.state !== "running") await ctx.resume();
    if (this.offset >= this.duration - 0.05) this.offset = 0;
    const when = ctx.currentTime + 0.05;
    this.startedAt = when - this.offset;
    this.sources = this.buffers.map((b, i) => {
      const src = ctx.createBufferSource();
      src.buffer = b;
      src.connect(this.gains[i]);
      if (this.offset < b.duration) src.start(when, this.offset);
      return src;
    });
    this.playing = true;
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.time;
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* not started or already stopped */
      }
      s.disconnect();
    }
    this.sources = [];
    this.playing = false;
  }

  seek(t: number) {
    const was = this.playing;
    if (was) this.pause();
    this.offset = Math.max(0, Math.min(this.duration, t));
    if (was) void this.play();
  }

  /** Switch to version i at the same position: a 10 ms crossfade on the shared clock. */
  select(i: number, immediate = false) {
    this.selected = i;
    if (!this.ctx) return;
    const gains = versionGains(this.gains.length, i);
    const now = this.ctx.currentTime;
    this.gains.forEach((g, k) => {
      if (immediate) g.gain.value = gains[k];
      else g.gain.setTargetAtTime(gains[k], now, 0.01);
    });
  }

  close() {
    this.token++;
    this.pause();
    void this.ctx?.close().catch(() => {});
    this.ctx = null;
  }
}

/** Gain per version: 1 for the chosen one, 0 for the rest. */
export function versionGains(count: number, selected: number): number[] {
  return Array.from({ length: count }, (_, k) => (k === selected ? 1 : 0));
}
