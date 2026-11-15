/** "Does the music follow the data?" in plain language, from track.json's fidelity block. */
import { fmtNum, fmtValue, shortUnits } from "./music";
import type { DataTrack } from "./types";

export interface FidelityBar {
  label: string;
  r2: number;
  kind: "real" | "mix" | "control";
}

export interface FidelityText {
  headline: string;
  detail: string;
  bars: FidelityBar[];
}

const pct = (r: number) => `${Math.round(Math.max(0, r) * 100)}%`;

export function describeFidelity(t: DataTrack): FidelityText {
  const f = t.fidelity ?? { r2: null };
  if (f.r2 == null) {
    const series = Object.entries(f.detail ?? {})
      .map(([k, v]) => [k, v.final_stem] as const)
      .filter(([, r]) => r?.pitch_within_half_semitone != null);
    return {
      headline: series.length ? "This track has no single melody, so each series was checked on its own." : "No fidelity check for this track yet.",
      detail: series.length
        ? "Each series was played back into a pitch tracker by itself: " +
          series.map(([k, r]) => `${k}, ${Math.round((r!.pitch_within_half_semitone as number) * 100)}% of ${r!.n} notes on the right pitch`).join("; ") +
          "."
        : "",
      bars: [],
    };
  }
  const follow = f.r2 >= 0.9 ? "follows the data closely" : f.r2 >= 0.6 ? "follows the data fairly well" : "follows the data only loosely";
  const ctrl =
    f.control_r2 == null ? "." : f.control_r2 < 0.2 ? ". The same notes shuffled in time do not." : ". Shuffled data scores almost as well, so treat this track with care.";
  const share = (r: number) => (r <= 0.005 ? "none of them" : pct(r));
  let detail =
    `We played the melody back into a pitch tracker and compared it with the data. The pitch explains ${pct(f.r2)} of the ups and downs ` +
    `(R² ${fmtNum(f.r2)}${f.mae != null ? `, average error ${fmtValue(f.mae, shortUnits(t.units), 1)}` : ""}).`;
  if (f.mix_r2 != null) {
    detail += ` Measured on the full mix, with the harmony and Earth sounds playing, it still explains ${pct(f.mix_r2)} (R² ${fmtNum(f.mix_r2)}).`;
  }
  if (f.control_r2 != null) {
    detail += ` The same notes in shuffled order explain ${share(f.control_r2)} (R² ${fmtNum(f.control_r2)}). Press A to hear the difference.`;
  }
  const bars: FidelityBar[] = [{ label: "Melody alone", r2: f.r2, kind: "real" }];
  if (f.mix_r2 != null) bars.push({ label: "Full mix", r2: f.mix_r2, kind: "mix" });
  if (f.control_r2 != null) bars.push({ label: "Shuffled control", r2: f.control_r2, kind: "control" });
  return { headline: `The melody ${follow}${ctrl}`, detail, bars };
}

/** Short tag for the track chooser and the player: words, then the number (set in mono). */
export function fidelityTag(r2: number | null): { kind: string; value: string } {
  return r2 == null ? { kind: "Data-faithful, checked per series", value: "" } : { kind: "Data-faithful", value: `R² ${fmtNum(r2)}` };
}
