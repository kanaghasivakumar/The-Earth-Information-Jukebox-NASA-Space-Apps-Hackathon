"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Menu, X } from "lucide-react";
import { CATALOGUE, TRACK_COPY } from "@/lib/data/copy";
import { fidelityTag } from "@/lib/data/fidelity";
import { fmt } from "@/lib/jukebox";

export const SEASON = "season";

export interface TrackItem {
  id: string;
  name: string;
  /** "Artistic sonification" or "Data-faithful" */
  kind: string;
  /** mono numbers after the kind: R², length */
  value: string;
  href: string;
}

/** Every track the site plays: the hurricane season (Vincent's player) first, then the nine data tracks. */
export function listenItems(): TrackItem[] {
  return [
    { id: SEASON, name: "Hurricane season", kind: "Artistic sonification", value: "5 styles", href: "/" },
    ...CATALOGUE.map((c) => {
      const tag = fidelityTag(c.r2);
      return {
        id: c.id,
        name: TRACK_COPY[c.id]?.name ?? c.title,
        kind: tag.kind,
        value: [tag.value, fmt(c.duration_s)].filter(Boolean).join(" · "),
        href: `/?track=${c.id}`,
      };
    }),
  ];
}

/** The data tracks only, for the Compare page. */
export function compareItems(): TrackItem[] {
  return CATALOGUE.map((c) => ({
    id: c.id,
    name: TRACK_COPY[c.id]?.name ?? c.title,
    kind: `${c.versions} version${c.versions === 1 ? "" : "s"}`,
    value: fmt(c.duration_s),
    href: `/compare?track=${c.id}`,
  }));
}

const linksIn = (ul: HTMLUListElement | null) => Array.from(ul?.querySelectorAll<HTMLAnchorElement>("a") ?? []);
const plainClick = (e: React.MouseEvent) => e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey;

/**
 * Hamburger track switcher: a disclosure button that opens a list of links. Focus moves to the current
 * track, ↑ ↓ Home End move between tracks, Esc closes and returns focus to the button, and the current
 * track carries aria-current. Each item is a real link (?track=<id>), so it also works without script.
 */
export function TrackMenu({ items, current, onChoose, label = "All tracks" }: { items: TrackItem[]; current: string; onChoose: (id: string) => void; label?: string }) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const panelId = useId();

  const close = useCallback((refocus: boolean) => {
    setOpen(false);
    if (refocus) button.current?.focus();
  }, []);

  // focus moves into the menu, onto the current track
  useEffect(() => {
    if (!open) return;
    const all = linksIn(list.current);
    (all.find((a) => a.hasAttribute("aria-current")) ?? all[0])?.focus();
  }, [open]);

  // a click outside closes it without stealing focus
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!wrap.current?.contains(e.target as Node)) close(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open, close]);

  const onKey = (e: React.KeyboardEvent) => {
    if (!open) return;
    const all = linksIn(list.current);
    const i = all.indexOf(document.activeElement as HTMLAnchorElement);
    let next = -1;
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      close(true);
      return;
    }
    if (e.key === "ArrowDown") next = i < 0 ? 0 : (i + 1) % all.length;
    else if (e.key === "ArrowUp") next = i < 0 ? all.length - 1 : (i - 1 + all.length) % all.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = all.length - 1;
    else return;
    // the player's own shortcuts (Home = back to the start) must not fire while the menu has the key
    e.preventDefault();
    e.stopPropagation();
    all[next]?.focus();
  };

  return (
    <div
      ref={wrap}
      className="jb-menu"
      onKeyDown={onKey}
      onBlur={(e) => {
        if (open && e.relatedTarget && !wrap.current?.contains(e.relatedTarget as Node)) close(false);
      }}
    >
      <button
        ref={button}
        type="button"
        className="jb-menu-btn"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => (open ? close(true) : setOpen(true))}
      >
        {open ? <X size={20} aria-hidden="true" /> : <Menu size={20} aria-hidden="true" />}
        <span>Tracks</span>
      </button>
      <nav id={panelId} className="jb-menu-panel" aria-label={label} hidden={!open}>
        <ul ref={list} className="jb-menu-list">
          {items.map((it) => {
            const on = it.id === current;
            return (
              <li key={it.id}>
                <a
                  href={it.href}
                  className="jb-track jb-menu-item"
                  aria-current={on ? "page" : undefined}
                  onClick={(e) => {
                    if (!plainClick(e)) return;
                    e.preventDefault();
                    if (!on) onChoose(it.id);
                    close(true);
                  }}
                >
                  <span className="jb-track-name">{it.name}</span>
                  <span className="jb-track-tag">
                    {it.kind}
                    {it.value ? (
                      <>
                        {" "}
                        <span className="jb-code">{it.value}</span>
                      </>
                    ) : null}
                  </span>
                </a>
              </li>
            );
          })}
        </ul>
      </nav>
    </div>
  );
}

/** The bar above the footage: the hamburger, the track title and one line of what it is. */
export function TitleBar({ title, subtitle, menu, heading = true }: { title: string; subtitle: React.ReactNode; menu: React.ReactNode; heading?: boolean }) {
  const Title = heading ? "h1" : "p";
  return (
    <div className="jb-titlebar">
      {menu}
      <div className="jb-titlebar-text">
        <Title className="jb-title">{title}</Title>
        <p className="jb-subtitle">{subtitle}</p>
      </div>
    </div>
  );
}

/** "NASA SVS 5209 · 1880–2023 · Data-faithful R² 0.97" for a data track. */
export function DataSubtitle({ id }: { id: string }) {
  const c = CATALOGUE.find((x) => x.id === id);
  const tag = fidelityTag(c?.r2 ?? null);
  const years = TRACK_COPY[id]?.years;
  return (
    <>
      NASA SVS <span className="jb-code">{id}</span>
      {years ? (
        <>
          {" · "}
          <span className="jb-code">{years}</span>
        </>
      ) : null}
      {" · "}
      {tag.kind}
      {tag.value ? (
        <>
          {" "}
          <span className="jb-code">{tag.value}</span>
        </>
      ) : null}
    </>
  );
}
