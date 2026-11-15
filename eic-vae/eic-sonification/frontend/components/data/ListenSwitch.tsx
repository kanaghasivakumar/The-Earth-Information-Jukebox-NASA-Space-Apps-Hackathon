"use client";

import { useCallback, useSyncExternalStore } from "react";
import type { Source } from "@/components/jukebox/Player";
import { Listen } from "@/components/jukebox/Listen";
import { isDataTrack } from "@/lib/data/copy";
import { DataPlayer } from "./DataPlayer";
import { SEASON, TrackChooser } from "./TrackChooser";

const EVENT = "jb:track";

function subscribe(fn: () => void) {
  window.addEventListener("popstate", fn);
  window.addEventListener(EVENT, fn);
  return () => {
    window.removeEventListener("popstate", fn);
    window.removeEventListener(EVENT, fn);
  };
}
const readTrack = () => {
  const id = new URLSearchParams(window.location.search).get("track");
  return isDataTrack(id) ? id : SEASON;
};

/** The Listen page: the hurricane season (Vincent's player) or one of the eight data tracks, chosen by ?track=. */
export function ListenSwitch({ season }: { season: Source }) {
  const current = useSyncExternalStore(subscribe, readTrack, () => SEASON);
  const choose = useCallback((id: string) => {
    const url = new URL(window.location.href);
    if (id === SEASON) url.searchParams.delete("track");
    else url.searchParams.set("track", id);
    if (url.href !== window.location.href) window.history.pushState(null, "", url);
    window.dispatchEvent(new Event(EVENT));
  }, []);
  return (
    <>
      <TrackChooser current={current} onChoose={choose} />
      {current === SEASON ? <Listen season={season} /> : <DataPlayer key={current} id={current} />}
    </>
  );
}
