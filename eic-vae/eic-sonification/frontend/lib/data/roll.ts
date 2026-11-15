/**
 * The score picture: a piano roll with the melody's value line and the Earth-voice balance underneath.
 * Drawn on a canvas that is aria-hidden; the page carries the text alternative and the note table.
 * Colours come from the stem-group tokens on the .jb root (--g-melody … --g-biophony), see DESIGN.md.
 */
import type { Controls, Group, TimelineNote } from "./types";

export interface RollData {
  notes: TimelineNote[];
  duration: number;
  groupOf: (voice: string) => Group;
  controls: Controls | null;
  muted: Partial<Record<Group, boolean>>;
}

export class Roll {
  private ctx: CanvasRenderingContext2D | null;
  private layer: HTMLCanvasElement;
  private data: RollData | null = null;
  private dpr = 1;
  private t = 0;
  private colours: Record<string, string> = {};

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d");
    this.layer = document.createElement("canvas");
  }

  private readColours() {
    const root = this.canvas.closest(".jb") ?? document.documentElement;
    const cs = getComputedStyle(root);
    for (const k of ["melody", "bands", "events", "harmony", "geophony", "anthrophony", "biophony"]) {
      this.colours[k] = cs.getPropertyValue(`--g-${k}`).trim() || "#e8eef2";
    }
    this.colours.rule = cs.getPropertyValue("--rule").trim() || "rgb(200 220 235 / 0.16)";
    this.colours.accent = cs.getPropertyValue("--accent").trim() || "#ff6b4a";
    this.colours.ink = cs.getPropertyValue("--ink").trim() || "#e8eef2";
  }

  setData(data: RollData) {
    this.data = data;
    this.layout();
  }

  layout() {
    this.readColours();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    this.canvas.width = this.layer.width = w;
    this.canvas.height = this.layer.height = h;
    this.dpr = dpr;
    this.paintStatic();
    this.draw(this.t);
  }

  private paintStatic() {
    const g = this.layer.getContext("2d");
    if (!g) return;
    const W = this.layer.width;
    const H = this.layer.height;
    g.clearRect(0, 0, W, H);
    if (!this.data || !this.data.notes.length) return;
    const { notes, duration, groupOf, controls, muted } = this.data;
    const lanes = controls
      ? (["geophony", "anthrophony", "biophony"] as const).filter((k) => controls.curves[k]?.some((v) => v > 0))
      : [];
    const laneH = lanes.length ? 14 * this.dpr : 0;
    const top = 8 * this.dpr;
    const bottom = H - 8 * this.dpr - (laneH + 4 * this.dpr) * lanes.length;
    const ps = notes.map((n) => n.pitch);
    const pmin = Math.min(...ps) - 2;
    const pmax = Math.max(...ps) + 2;
    const x = (t: number) => (t / duration) * W;
    const y = (p: number) => bottom - ((p - pmin) / (pmax - pmin || 1)) * (bottom - top);
    const rowH = Math.max(2 * this.dpr, (bottom - top) / (pmax - pmin || 1));

    g.strokeStyle = this.colours.rule;
    g.lineWidth = 1;
    for (let p = Math.ceil(pmin / 12) * 12; p <= pmax; p += 12) {
      g.beginPath();
      g.moveTo(0, y(p));
      g.lineTo(W, y(p));
      g.stroke();
    }

    const order = (n: TimelineNote) => (groupOf(n.voice) === "melody" ? 1 : 0);
    for (const n of [...notes].sort((a, b) => order(a) - order(b))) {
      const grp = groupOf(n.voice);
      const lead = grp === "melody";
      g.fillStyle = this.colours[grp];
      g.globalAlpha = (muted[grp] ? 0.25 : 1) * (lead ? 0.95 : 0.45);
      const h = lead ? rowH * 1.1 : rowH * 0.7;
      g.fillRect(x(n.start), y(n.pitch) - h / 2, Math.max(2, x(n.end) - x(n.start) - this.dpr), h);
    }
    g.globalAlpha = 1;

    const lead = notes.filter((n) => groupOf(n.voice) === "melody" && n.value != null);
    if (lead.length) {
      g.strokeStyle = this.colours.ink;
      g.globalAlpha = 0.5;
      g.lineWidth = 1.5 * this.dpr;
      g.beginPath();
      lead.forEach((n, i) => {
        const xx = x((n.start + n.end) / 2);
        const yy = y(n.pitch);
        if (i) g.lineTo(xx, yy);
        else g.moveTo(xx, yy);
      });
      g.stroke();
      g.globalAlpha = 1;
    }

    lanes.forEach((k, i) => {
      const y0 = bottom + 8 * this.dpr + i * (laneH + 4 * this.dpr);
      const vals = controls!.curves[k]!;
      const ts = controls!.t;
      g.fillStyle = this.colours[k];
      g.globalAlpha = muted[k] ? 0.2 : 0.7;
      g.beginPath();
      g.moveTo(0, y0 + laneH);
      ts.forEach((t, j) => g.lineTo(x(t), y0 + laneH - vals[j] * (laneH - 2)));
      g.lineTo(x(ts[ts.length - 1]), y0 + laneH);
      g.closePath();
      g.fill();
      g.globalAlpha = 1;
    });
  }

  /** The played part keeps full strength; the rest is dimmed; the playhead is the vermilion pen. */
  draw(t: number) {
    this.t = t;
    const c = this.ctx;
    if (!c) return;
    const W = this.canvas.width;
    const H = this.canvas.height;
    c.clearRect(0, 0, W, H);
    if (!this.data) return;
    const xx = Math.round((t / this.data.duration) * W);
    if (xx > 0) c.drawImage(this.layer, 0, 0, xx, H, 0, 0, xx, H);
    if (W - xx > 0) {
      c.globalAlpha = 0.45;
      c.drawImage(this.layer, xx, 0, W - xx, H, xx, 0, W - xx, H);
      c.globalAlpha = 1;
    }
    c.fillStyle = this.colours.accent;
    c.fillRect(xx - this.dpr, 0, 2 * this.dpr, H);
  }
}
