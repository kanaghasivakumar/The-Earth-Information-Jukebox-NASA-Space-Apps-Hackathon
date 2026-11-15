/** Throttled announcements for the ARIA live region, and the browser voice (speechSynthesis). */

export const canSpeak = () => typeof window !== "undefined" && "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;

/** Speak text; resolves when done, or after a timeout because `onend` is unreliable in some browsers. */
export function speak(text: string, { interrupt = true, rate = 1.05 } = {}): Promise<void> {
  return new Promise((resolve) => {
    if (!canSpeak()) {
      setTimeout(resolve, Math.min(6000, 400 + text.length * 55));
      return;
    }
    if (interrupt) window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = rate;
    u.lang = document.documentElement.lang || "en";
    let done = false;
    const finish = () => {
      if (!done) {
        done = true;
        clearTimeout(timer);
        resolve();
      }
    };
    const timer = setTimeout(finish, 2500 + text.length * 90);
    u.onend = finish;
    u.onerror = finish;
    window.speechSynthesis.speak(u);
  });
}

export function cancelSpeech() {
  if (canSpeak()) window.speechSynthesis.cancel();
}

export const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * Two channels, like the plain-JS player: `say` for the polite live region (throttled, newest wins)
 * and `status` for confirmations of what you just did (never throttled). Repeated text still re-announces.
 */
export class Announcer {
  speech = false;
  private last = 0;
  private pending: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private writeLive: (text: string) => void,
    private writeStatus: (text: string) => void,
  ) {}

  say(text: string, { minGap = 0, voice = false } = {}) {
    const now = performance.now() / 1000;
    if (this.pending) clearTimeout(this.pending);
    const go = () => {
      this.last = performance.now() / 1000;
      this.writeLive(text);
      if (this.speech || voice) void speak(text);
    };
    const wait = this.last + minGap - now;
    if (wait > 0) this.pending = setTimeout(go, wait * 1000);
    else go();
  }

  status(text: string) {
    this.writeStatus(text);
    if (this.speech) void speak(text);
  }

  dispose() {
    if (this.pending) clearTimeout(this.pending);
  }
}

/** A live-region writer that re-announces identical text by toggling a trailing space. */
export const toggleSame = (prev: string, next: string) => (prev === next ? `${next} ` : next);
