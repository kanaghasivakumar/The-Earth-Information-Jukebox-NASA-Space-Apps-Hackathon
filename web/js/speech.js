// ARIA live-region announcer (throttled) and Web Speech helpers.

export class Announcer {
  constructor(liveEl, statusEl) {
    this.live = liveEl;
    this.statusEl = statusEl;
    this.speech = false;   // also voice announcements with speechSynthesis
    this.last = 0;
    this.pending = null;
  }

  /** Write to the polite live region. Repeated text still re-announces (trailing space toggles). */
  write(el, text) {
    el.textContent = el.textContent === text ? `${text} ` : text;
  }

  /** Throttled: at most one message per `minGap` seconds; the newest one wins. */
  say(text, { minGap = 0, speak: forceSpeak = false } = {}) {
    const now = performance.now() / 1000;
    clearTimeout(this.pending);
    const go = () => {
      this.last = performance.now() / 1000;
      this.write(this.live, text);
      if (this.speech || forceSpeak) speak(text);
    };
    const wait = this.last + minGap - now;
    if (wait > 0) this.pending = setTimeout(go, wait * 1000);
    else go();
  }

  /** Confirmation of a user action (toggle, load); never throttled. */
  status(text) {
    this.write(this.statusEl, text);
    if (this.speech) speak(text);
  }
}

export const canSpeak = () => 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;

/** Speak text; resolves when finished (or after a timeout, since `onend` is unreliable in some browsers). */
export function speak(text, { interrupt = true, rate = 1.05 } = {}) {
  return new Promise((resolve) => {
    if (!canSpeak()) { setTimeout(resolve, Math.min(6000, 400 + text.length * 55)); return; }
    if (interrupt) window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = rate;
    u.lang = document.documentElement.lang || 'en';
    let done = false;
    const finish = () => { if (!done) { done = true; clearTimeout(timer); resolve(); } };
    const timer = setTimeout(finish, 2500 + text.length * 90);
    u.onend = finish; u.onerror = finish;
    window.speechSynthesis.speak(u);
  });
}

export function cancelSpeech() { if (canSpeak()) window.speechSynthesis.cancel(); }

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
