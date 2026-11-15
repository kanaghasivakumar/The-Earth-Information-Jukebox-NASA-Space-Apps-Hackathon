import { useCallback, useSyncExternalStore } from "react";

/**
 * A tiny external store for the playback clock. Components that need the time subscribe to it,
 * so a 20 Hz clock re-renders only the clock, the timeline and the "now" text, never the style list.
 */
export interface TimeStore {
  get: () => number;
  set: (t: number) => void;
  subscribe: (fn: () => void) => () => void;
}

export function createTimeStore(initial = 0): TimeStore {
  let t = initial;
  const subs = new Set<() => void>();
  return {
    get: () => t,
    set: (next: number) => {
      if (next === t) return;
      t = next;
      subs.forEach((fn) => fn());
    },
    subscribe: (fn) => {
      subs.add(fn);
      return () => {
        subs.delete(fn);
      };
    },
  };
}

export function useTime(store: TimeStore): number {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}

/** Re-render only when the selected value changes. The selector must return stable values (no fresh objects). */
export function useTimeSelector<T>(store: TimeStore, select: (t: number) => T): T {
  const get = useCallback(() => select(store.get()), [store, select]);
  return useSyncExternalStore(store.subscribe, get, get);
}
