"use client";

import { useEffect, useRef, useState } from "react";
import { UploadCloud } from "lucide-react";
import { STYLE_COPY, type Signals, type Track } from "@/lib/jukebox";

export interface UploadedSource {
  id: string;
  video: string;
  poster?: string;
  signals: Signals;
  tracks: Track[];
}

interface Props {
  api: string;
  current: string | null;
  onReady: (source: UploadedSource, filename: string) => void;
  onReset: () => void;
}

type Phase =
  | { k: "idle" }
  | { k: "sending"; pct: number }
  | { k: "working"; stage: string; pct: number }
  | { k: "error"; msg: string; offline?: boolean };

const ALL = ["ambient", "cinematic", "classical", "jazz", "electronic"];
const ACCEPT = ".mp4,.mov,.m4v,.webm,.avi,.gif,video/*";

export function UploadPanel({ api, current, onReady, onReset }: Props) {
  const [phase, setPhase] = useState<Phase>({ k: "idle" });
  const [file, setFile] = useState<File | null>(null);
  const [styles, setStyles] = useState<string[]>(ALL);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const busy = phase.k === "sending" || phase.k === "working";

  const poll = (id: string, name: string) => {
    const step = async () => {
      if (!alive.current) return;
      try {
        const r = await fetch(`${api}/api/jobs/${id}`);
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || "Lost track of the job.");
        if (d.status === "completed") {
          setPhase({ k: "idle" });
          setFile(null);
          onReady(d.source, name);
          return;
        }
        if (d.status === "failed") {
          setPhase({ k: "error", msg: d.error || "The generator stopped without a message." });
          return;
        }
        setPhase({ k: "working", stage: d.stage, pct: d.progress ?? 0 });
        timer.current = setTimeout(step, 1500);
      } catch (e) {
        setPhase({ k: "error", msg: e instanceof Error ? e.message : "Lost contact with the generator.", offline: true });
      }
    };
    step();
  };

  const start = () => {
    if (!file || busy || !styles.length) return;
    const body = new FormData();
    body.append("file", file);
    body.append("styles", styles.join(","));
    setPhase({ k: "sending", pct: 0 });
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${api}/api/jobs`);
    xhr.upload.onprogress = (e) => e.lengthComputable && setPhase({ k: "sending", pct: e.loaded / e.total });
    xhr.onerror = () => setPhase({ k: "error", msg: "Could not reach the generator.", offline: true });
    xhr.onload = () => {
      let d: { job_id?: string; detail?: string } = {};
      try {
        d = JSON.parse(xhr.responseText);
      } catch {}
      if (xhr.status === 202 && d.job_id) poll(d.job_id, file.name);
      else setPhase({ k: "error", msg: d.detail || `Upload failed (${xhr.status}).` });
    };
    xhr.send(body);
  };

  const pick = (f: File | undefined | null) => {
    if (!f) return;
    setFile(f);
    setPhase({ k: "idle" });
  };

  const toggleStyle = (id: string) =>
    setStyles((s) => (s.includes(id) ? s.filter((x) => x !== id) : ALL.filter((x) => x === id || s.includes(x))));

  const pct = phase.k === "sending" ? phase.pct * 0.15 : phase.k === "working" ? phase.pct : 0;
  const status =
    phase.k === "sending"
      ? `Uploading ${Math.round(phase.pct * 100)}%`
      : phase.k === "working"
        ? phase.stage
        : "";

  return (
    <section id="upload" className="jb-section jb-upload jb-page" aria-labelledby="upload-h">
      <div className="jb-upload-copy">
        <h1 id="upload-h" className="jb-h1">
          Bring your own visualization.
        </h1>
        <p>
          Upload an animation from the NASA Earth Information Center, or any silent time-lapse of the Earth. The model
          watches it, finds its visual patterns and scores it in the styles you choose. Longer videos take a few minutes.
        </p>
        <p className="jb-fine">MP4, MOV, WebM, AVI or GIF · up to 300 MB and 10 minutes · processed on this computer, not uploaded to a hosted service.</p>
        {current ? (
          <p className="jb-using">
            Now playing your upload <strong>{current}</strong>.{" "}
            <button type="button" className="jb-link" onClick={onReset}>
              Back to the hurricane season
            </button>
          </p>
        ) : null}
      </div>

      <div className="jb-upload-form">
        <div
          className={`jb-drop${over ? " over" : ""}${file ? " has" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            if (!busy) pick(e.dataTransfer.files?.[0]);
          }}
        >
          <UploadCloud size={28} aria-hidden="true" />
          {file ? (
            <p className="jb-drop-name">
              <strong>{file.name}</strong>
              <span>{(file.size / 1048576).toFixed(1)} MB</span>
            </p>
          ) : (
            <p className="jb-drop-name">Drop a video here</p>
          )}
          <input
            ref={input}
            id="file"
            type="file"
            accept={ACCEPT}
            className="jb-sr"
            disabled={busy}
            onChange={(e) => pick(e.target.files?.[0])}
          />
          <label htmlFor="file" className="jb-btn-ghost" aria-disabled={busy}>
            {file ? "Choose a different file" : "Choose a file"}
          </label>
        </div>

        <fieldset className="jb-fieldset" disabled={busy}>
          <legend>Styles to generate</legend>
          <div className="jb-checks">
            {ALL.map((id) => (
              <label key={id} className="jb-check">
                <input type="checkbox" checked={styles.includes(id)} onChange={() => toggleStyle(id)} />
                <span>{STYLE_COPY[id].name}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <button type="button" className="jb-play jb-go" onClick={start} disabled={!file || busy || !styles.length}>
          {busy ? "Working…" : "Make music from this"}
        </button>

        {busy ? (
          <div className="jb-progress">
            <div role="progressbar" aria-label="Generating music" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct * 100)}>
              <span style={{ width: `${Math.max(4, pct * 100)}%` }} />
            </div>
            <p role="status">{status}</p>
          </div>
        ) : null}

        {phase.k === "error" ? (
          <div className="jb-error" role="alert">
            <p>
              <strong>{phase.offline ? "The generator is not running." : "That did not work."}</strong> {phase.msg}
            </p>
            {phase.offline ? (
              <details className="jb-fine">
                <summary>For the team: how to start it</summary>
                <code>cd eic-vae/eic-sonification &amp;&amp; npm run dev:all</code>
              </details>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}
