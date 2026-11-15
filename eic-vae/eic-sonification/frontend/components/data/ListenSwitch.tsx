"use client";

import { useCallback, useMemo, useSyncExternalStore } from "react";
import type { Source } from "@/components/jukebox/Player";
import { Listen } from "@/components/jukebox/Listen";
import { isDataTrack, trackName } from "@/lib/data/copy";
import { DataPlayer } from "./DataPlayer";
import { DataSubtitle, SEASON, TitleBar, TrackMenu, listenItems } from "./TrackMenu";

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

/**
 * The Listen page: a title bar with the hamburger track menu, then the hurricane season (Vincent's player)
 * or one of the nine data tracks, chosen by ?track=.
 */
export function ListenSwitch({ season }: { season: Source }) {
  const current = useSyncExternalStore(subscribe, readTrack, () => SEASON);
  const items = useMemo(() => listenItems(), []);
  const choose = useCallback((id: string) => {
    const url = new URL(window.location.href);
    if (id === SEASON) url.searchParams.delete("track");
    else url.searchParams.set("track", id);
    if (url.href !== window.location.href) window.history.pushState(null, "", url);
    window.dispatchEvent(new Event(EVENT));
  }, []);
  const menu = <TrackMenu items={items} current={current} onChoose={choose} />;
  return (
    <div className={`jb-listen-in${current === SEASON ? " jb-season" : ""}`}>
      {current === SEASON ? (
        // Vincent's player keeps its own h1, so the bar's title is not a second heading
        <TitleBar
          heading={false}
          title="Hurricane season"
          subtitle={
            <>
              NASA SVS <span className="jb-code">5097</span> · <span className="jb-code">2022</span> · Artistic sonification · 5 styles
            </>
          }
          menu={menu}
        />
      ) : (
        <TitleBar title={trackName(current)} subtitle={<DataSubtitle id={current} />} menu={menu} />
      )}
      {current === SEASON ? <Listen season={season} /> : <DataPlayer key={current} id={current} />}
    </div>
  );
}
