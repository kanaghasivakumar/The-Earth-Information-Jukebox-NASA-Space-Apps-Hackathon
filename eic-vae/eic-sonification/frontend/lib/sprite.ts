import type { CSSProperties } from "react";

/** One tile of a cols × rows sprite sheet, as a CSS background. */
export function spriteStyle(url: string, idx: number, cols: number, rows: number): CSSProperties {
  const i = Math.max(0, Math.min(cols * rows - 1, idx));
  const c = i % cols;
  const r = Math.floor(i / cols);
  return {
    backgroundImage: `url(${url})`,
    backgroundSize: `${cols * 100}% ${rows * 100}%`,
    backgroundPosition: `${cols > 1 ? (c / (cols - 1)) * 100 : 0}% ${rows > 1 ? (r / (rows - 1)) * 100 : 0}%`,
    backgroundRepeat: "no-repeat",
  };
}
