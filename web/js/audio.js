// Web Audio engine: synced stem playback, per-group gain, A/B control, HRTF binaural, live synth tones.
import { GROUPS, midiHz } from './music.js';

// Binaural home positions (degrees azimuth, + = right; elevation up). Note pan moves a group ±75°.
// Earth sounds to the left, human sounds to the right, living sounds above; data melody in front.
const HOME = {
  melody: [0, 0], bands: [0, 15], events: [0, 0], harmony: [-25, 0],
  geophony: [-55, 0], anthrophony: [55, 0], biophony: [0, 40],
};
const RAMP = 0.04;

export class Engine {
  constructor() {
    this.ctx = null;
    this.stems = [];          // {name, group, buffer, gain}
    this.control = null;      // {buffer, gain, replaces}
    this.sources = [];
    this.playing = false;
    this.offset = 0;
    this.startedAt = 0;
    this.duration = 0;
    this.shuffled = false;
    this.liveLead = false;
    this.binaural = false;
    this.groupState = Object.fromEntries(GROUPS.map((g) => [g, { volume: 1, muted: false }]));
    this.live = null;         // {notes, pitchOf, idx}
  }

  get supported() { return !!(window.AudioContext || window.webkitAudioContext); }

  ensure() {
    if (this.ctx) return this.ctx;
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = (this.ctx = new AC());
    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -3; limiter.knee.value = 0; limiter.ratio.value = 20;
    limiter.attack.value = 0.003; limiter.release.value = 0.1;
    limiter.connect(ctx.destination);
    this.master = ctx.createGain(); this.master.gain.value = 0.62; // lead+bands sum to ~1.6x a stem
    this.master.connect(limiter);
    this.duckGain = ctx.createGain(); this.duckGain.connect(this.master);
    this.ui = ctx.createGain(); this.ui.gain.value = 0.9; this.ui.connect(this.master);
    this.groups = {};
    for (const g of GROUPS) {
      const gain = ctx.createGain();
      const dry = ctx.createGain();
      const wet = ctx.createGain(); wet.gain.value = 0;
      const panner = this.makePanner();
      gain.connect(dry).connect(this.duckGain);
      gain.connect(panner).connect(wet).connect(this.duckGain);
      this.groups[g] = { gain, dry, wet, panner, pan: null };
      this.setPannerAngle(panner, ...HOME[g]);
    }
    return ctx;
  }

  makePanner() {
    const p = this.ctx.createPanner();
    p.panningModel = 'HRTF';
    p.distanceModel = 'inverse'; p.refDistance = 1; p.rolloffFactor = 0;
    return p;
  }

  setPannerAngle(p, azDeg, elDeg = 0) {
    const az = (azDeg * Math.PI) / 180, el = (elDeg * Math.PI) / 180;
    const x = Math.sin(az) * Math.cos(el), y = Math.sin(el), z = -Math.cos(az) * Math.cos(el);
    if (p.positionX) {
      const t = this.ctx.currentTime;
      p.positionX.setTargetAtTime(x, t, 0.05); p.positionY.setTargetAtTime(y, t, 0.05); p.positionZ.setTargetAtTime(z, t, 0.05);
    } else p.setPosition(x, y, z);
  }

  async load(track, base, onProgress = () => {}) {
    this.stop();
    const ctx = this.ensure();
    for (const s of this.stems) s.gain.disconnect();
    if (this.control) this.control.gain.disconnect();
    this.stems = []; this.control = null; this.offset = 0;
    const jobs = (track.stems || []).map((s) => ({ ...s, url: base + s.file }));
    if (track.control?.file) jobs.push({ name: '__control', group: track.control.replaces || 'melody', url: base + track.control.file, isControl: true });
    let done = 0;
    const decoded = await Promise.all(jobs.map(async (j) => {
      const res = await fetch(j.url);
      if (!res.ok) throw new Error(`${j.url}: HTTP ${res.status}`);
      const buf = await decode(ctx, await res.arrayBuffer());
      onProgress(++done, jobs.length);
      return { ...j, buffer: buf };
    }));
    for (const j of decoded) {
      const gain = ctx.createGain();
      const group = GROUPS.includes(j.group) ? j.group : 'melody';
      gain.connect(this.groups[group].gain);
      const entry = { name: j.name, group, buffer: j.buffer, gain, label: j.label };
      if (j.isControl) this.control = { ...entry, replaces: group };
      else this.stems.push(entry);
    }
    this.duration = Math.max(track.duration_s || 0, ...decoded.map((d) => d.buffer.duration));
    this.applyStemGains(true);
    return this.stems.map((s) => s.group);
  }

  get time() {
    if (!this.playing || !this.ctx) return this.offset;
    return Math.min(this.duration, this.ctx.currentTime - this.startedAt);
  }

  async play() {
    const ctx = this.ensure();
    if (ctx.state !== 'running') await ctx.resume();
    if (this.playing) return;
    if (this.offset >= this.duration - 0.05) this.offset = 0;
    const when = ctx.currentTime + 0.03;
    this.startedAt = when - this.offset;
    const all = [...this.stems, ...(this.control ? [this.control] : [])];
    this.sources = all.filter((s) => this.offset < s.buffer.duration).map((s) => {
      const src = ctx.createBufferSource();
      src.buffer = s.buffer; src.connect(s.gain); src.start(when, this.offset);
      return src;
    });
    this.playing = true;
    if (this.live) { this.live.cursor = this.offset; this.live.fresh = true; }
  }

  stop() {
    for (const s of this.sources) { try { s.stop(); } catch { /* already stopped */ } s.disconnect(); }
    this.sources = [];
    for (const o of this.liveVoices || []) { try { o.stop(); } catch { /* */ } }
    this.liveVoices = [];
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.time;
    this.stop();
    this.playing = false;
  }

  seek(t) {
    const was = this.playing;
    if (was) { this.offset = this.time; this.stop(); this.playing = false; }
    this.offset = Math.max(0, Math.min(this.duration, t));
    if (was) this.play();
  }

  ramp(param, v, immediate = false) {
    if (immediate) { param.value = v; return; }
    param.setTargetAtTime(v, this.ctx.currentTime, RAMP);
  }

  applyStemGains(immediate = false) {
    const ctrlOn = this.shuffled && !!this.control;
    for (const s of this.stems) {
      const off = (ctrlOn && s.group === this.control.replaces) || (this.liveLead && s.group === 'melody');
      this.ramp(s.gain.gain, off ? 0 : 1, immediate);
    }
    if (this.control) this.ramp(this.control.gain.gain, ctrlOn ? 1 : 0, immediate);
  }

  setShuffled(on) { this.shuffled = on; this.applyStemGains(); }

  setGroup(g, { volume, muted } = {}) {
    const st = this.groupState[g];
    if (volume != null) st.volume = volume;
    if (muted != null) st.muted = muted;
    if (this.ctx) this.ramp(this.groups[g].gain.gain, st.muted ? 0 : st.volume);
  }

  setMasterMuted(m) { this.masterMuted = m; if (this.ctx) this.ramp(this.master.gain, m ? 0 : 0.62); }

  setBinaural(on) {
    this.binaural = on;
    if (!this.ctx) return;
    for (const g of GROUPS) {
      this.ramp(this.groups[g].dry.gain, on ? 0 : 1);
      this.ramp(this.groups[g].wet.gain, on ? 1.4 : 0); // HRTF output is quieter than the dry stereo
    }
  }

  /** Move a group's binaural source: pan -1 (west/left) .. +1 (east/right), added to its home angle. */
  setGroupPan(g, pan) {
    const grp = this.groups?.[g];
    if (!grp || grp.pan === pan) return;
    grp.pan = pan;
    const [az, el] = HOME[g];
    this.setPannerAngle(grp.panner, az + (pan || 0) * 75, el);
  }

  duck(level) { if (this.ctx) this.ramp(this.duckGain.gain, level); }

  /**
   * One synthesized note. bus: 'ui' (legend/sweep tones, ignore mutes) or a group name.
   * pan -1..1 places it (stereo, or HRTF when binaural is on). Returns the end time.
   */
  tone(pitch, { when = 0, dur = 0.5, vel = 90, pan = 0, bus = 'ui', timbre = 'pluck' } = {}) {
    const ctx = this.ensure();
    const t0 = Math.max(ctx.currentTime + 0.01, when || 0);
    dur = Math.max(0.06, dur);
    const f = midiHz(pitch);
    const env = ctx.createGain();
    const peak = (vel / 127) * 0.32;
    const rel = timbre === 'pluck' ? 0.3 : 0.15;
    env.gain.setValueAtTime(0, t0);
    env.gain.linearRampToValueAtTime(peak, t0 + 0.008);
    if (timbre === 'pluck') env.gain.setTargetAtTime(peak * 0.35, t0 + 0.01, 0.18);
    env.gain.setValueAtTime(timbre === 'pluck' ? peak * 0.35 : peak, t0 + dur);
    env.gain.setTargetAtTime(0, t0 + dur, rel / 3);
    const oscs = [[1, 'sine', 1], [2, 'triangle', 0.35], [3, 'sine', 0.12]].map(([k, type, a]) => {
      const o = ctx.createOscillator(); o.type = type; o.frequency.value = f * k;
      const g = ctx.createGain(); g.gain.value = a; o.connect(g).connect(env);
      o.start(t0); o.stop(t0 + dur + rel + 0.1);
      return o;
    });
    let out = env;
    if (bus === 'ui') {
      if (this.binaural) {
        const p = this.makePanner(); this.setPannerAngle(p, pan * 80, 0); env.connect(p); out = p;
      } else if (ctx.createStereoPanner) {
        const p = ctx.createStereoPanner(); p.pan.value = pan; env.connect(p); out = p;
      }
      out.connect(this.ui);
    } else {
      env.connect(this.groups[bus]?.gain || this.ui);
    }
    oscs[0].onended = () => { env.disconnect(); if (out !== env) out.disconnect(); };
    return { end: t0 + dur + rel, oscs };
  }

  // ---- live lead: re-synthesize the melody from the timeline (used for polarity flip) ----
  setLiveLead(on, notes = [], pitchOf = (n) => n.pitch) {
    this.liveLead = on;
    for (const o of this.liveVoices || []) { try { o.stop(); } catch { /* */ } }
    this.liveVoices = [];
    this.live = on ? { notes, pitchOf, cursor: this.time, fresh: true } : null;
    this.applyStemGains();
  }

  /** Call every ~50–200 ms: schedules live-lead notes inside a short look-ahead window. */
  tick(lookahead = 0.35) {
    if (!this.live || !this.playing) return;
    const L = this.live, now = this.time, until = now + lookahead;
    if (L.cursor == null || L.cursor < now - 0.25 || L.cursor > until + 1) { L.cursor = now; L.fresh = true; }
    for (const n of L.notes) {
      const startsInWindow = n.start >= L.cursor && n.start < until;
      const sounding = L.fresh && n.start < L.cursor && n.end > L.cursor; // note already under way at (re)start
      if (!startsInWindow && !sounding) continue;
      const start = Math.max(n.start, L.cursor);
      const { oscs } = this.tone(L.pitchOf(n), { when: this.startedAt + start, dur: n.end - start, vel: n.velocity, bus: 'melody' });
      this.liveVoices.push(...oscs);
    }
    L.fresh = false;
    L.cursor = until;
    if (this.liveVoices.length > 300) this.liveVoices.splice(0, 150);
  }
}

function decode(ctx, ab) {
  return new Promise((resolve, reject) => {
    const p = ctx.decodeAudioData(ab, resolve, reject);
    if (p && p.then) p.then(resolve, reject);
  });
}
