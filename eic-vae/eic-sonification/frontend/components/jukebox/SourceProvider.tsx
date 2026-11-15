"use client";

import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { Source } from "./Player";
import type { UploadedSource } from "./UploadPanel";

export const API = (process.env.NEXT_PUBLIC_JUKEBOX_API || "http://localhost:8000").replace(/\/$/, "");

interface Ctx {
  upload: Source | null;
  uploadName: string | null;
  applyUpload: (u: UploadedSource, filename: string) => void;
  reset: () => void;
}

const SourceContext = createContext<Ctx | null>(null);

/** Lives in the shared layout, so an upload survives moving between pages. */
export function SourceProvider({ children }: { children: React.ReactNode }) {
  const [upload, setUpload] = useState<Source | null>(null);
  const [uploadName, setUploadName] = useState<string | null>(null);

  const applyUpload = useCallback((u: UploadedSource, filename: string) => {
    const abs = (p: string) => API + p;
    setUploadName(filename);
    setUpload({
      id: u.id,
      title: "Your visualization, as music.",
      lede:
        "The same model and composer that scored the hurricane season have listened to your file. Pick a style and press play. The pattern groups are found fresh in your footage.",
      credit: `Your upload: ${filename}. Processed on this computer.`,
      video: abs(u.video),
      poster: u.poster ? abs(u.poster) : undefined,
      signals: u.signals,
      tracks: u.tracks.map((t) => ({ ...t, audio: abs(t.audio), midi: abs(t.midi) })),
      scenes: false,
    });
  }, []);

  const reset = useCallback(() => {
    setUploadName(null);
    setUpload(null);
  }, []);

  const value = useMemo(() => ({ upload, uploadName, applyUpload, reset }), [upload, uploadName, applyUpload, reset]);
  return <SourceContext.Provider value={value}>{children}</SourceContext.Provider>;
}

export function useSource() {
  const ctx = useContext(SourceContext);
  if (!ctx) throw new Error("useSource must be used inside SourceProvider");
  return ctx;
}
