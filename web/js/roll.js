// Decorative piano roll + value plot on canvas (aria-hidden; the page carries a text alternative).

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export class Roll {
  constructor(canvas, { onSeek } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = null;
    this.t = 0;
    this.layer = document.createElement('canvas');
    canvas.addEventListener('click', (e) => {
      if (!this.data || !onSeek) return;
      const r = canvas.getBoundingClientRect();
      onSeek(((e.clientX - r.left) / r.width) * this.data.duration);
    });
    if ('ResizeObserver' in window) new ResizeObserver(() => this.layout()).observe(canvas);
    window.matchMedia?.('(prefers-color-scheme: light)').addEventListener?.('change', () => this.layout());
  }

  /** data: {notes, duration, groupOf(voice), controls} */
  setData(data) { this.data = data; this.layout(); }

  layout() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    this.canvas.width = this.layer.width = w;
    this.canvas.height = this.layer.height = h;
    this.dpr = dpr;
    this.paintStatic();
    this.draw(this.t);
  }

  paintStatic() {
    const g = this.layer.getContext('2d');
    const { width: W, height: H } = this.layer;
    g.clearRect(0, 0, W, H);
    if (!this.data) return;
    const { notes, duration, groupOf, controls } = this.data;
    const lanes = controls ? ['geophony', 'anthrophony', 'biophony'].filter((k) => controls.curves?.[k]?.some((v) => v > 0)) : [];
    const laneH = lanes.length ? 14 * this.dpr : 0;
    const top = 8 * this.dpr, bottom = H - 8 * this.dpr - laneH * lanes.length;
    const ps = notes.map((n) => n.pitch);
    const pmin = Math.min(...ps) - 2, pmax = Math.max(...ps) + 2;
    const x = (t) => (t / duration) * W;
    const y = (p) => bottom - ((p - pmin) / (pmax - pmin || 1)) * (bottom - top);
    const rowH = Math.max(2 * this.dpr, (bottom - top) / (pmax - pmin || 1));

    // octave grid
    g.strokeStyle = css('--border'); g.globalAlpha = 0.5; g.lineWidth = 1;
    for (let p = Math.ceil(pmin / 12) * 12; p <= pmax; p += 12) { g.beginPath(); g.moveTo(0, y(p)); g.lineTo(W, y(p)); g.stroke(); }
    g.globalAlpha = 1;

    const order = (n) => (groupOf(n.voice) === 'melody' ? 1 : 0); // lead on top
    for (const n of [...notes].sort((a, b) => order(a) - order(b))) {
      const grp = groupOf(n.voice);
      const lead = grp === 'melody';
      g.fillStyle = css(`--${grp}`) || css('--muted');
      g.globalAlpha = lead ? 0.95 : 0.4;
      const h = lead ? rowH * 1.1 : rowH * 0.7;
      g.fillRect(x(n.start), y(n.pitch) - h / 2, Math.max(2, x(n.end) - x(n.start) - this.dpr), h);
    }
    g.globalAlpha = 1;

    // lead value line
    const lead = notes.filter((n) => groupOf(n.voice) === 'melody');
    if (lead.length) {
      g.strokeStyle = css('--text'); g.globalAlpha = 0.55; g.lineWidth = 1.5 * this.dpr; g.beginPath();
      lead.forEach((n, i) => { const xx = x((n.start + n.end) / 2), yy = y(n.pitch); i ? g.lineTo(xx, yy) : g.moveTo(xx, yy); });
      g.stroke(); g.globalAlpha = 1;
    }

    // voice-balance lanes
    lanes.forEach((k, i) => {
      const y0 = bottom + 8 * this.dpr + i * laneH, vals = controls.curves[k], ts = controls.t;
      g.fillStyle = css(`--${k}`); g.globalAlpha = 0.7; g.beginPath(); g.moveTo(0, y0 + laneH);
      ts.forEach((t, j) => g.lineTo(x(t), y0 + laneH - vals[j] * (laneH - 2)));
      g.lineTo(x(ts[ts.length - 1]), y0 + laneH); g.closePath(); g.fill(); g.globalAlpha = 1;
    });
  }

  draw(t) {
    this.t = t;
    const c = this.ctx, { width: W, height: H } = this.canvas;
    c.clearRect(0, 0, W, H);
    c.drawImage(this.layer, 0, 0);
    if (!this.data) return;
    const xx = (t / this.data.duration) * W;
    c.fillStyle = css('--accent');
    c.fillRect(xx - this.dpr, 0, 2 * this.dpr, H);
  }
}
