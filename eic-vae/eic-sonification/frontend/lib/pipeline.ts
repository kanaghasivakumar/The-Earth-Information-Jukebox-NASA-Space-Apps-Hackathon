import data from "@/public/jukebox/pipeline.json";

/** [start seconds, duration seconds, MIDI pitch, velocity] within the 30-second tour. */
export type Note = [number, number, number, number];

export interface PipelineFrame {
  t: number;
  gi: number;
  img: string;
  latent: number[];
  xy: [number, number];
  region: number;
  flow16: number[][][];
}

export interface Pipeline {
  segs: [number, number][];
  span: number;
  cols: number;
  rows: number;
  frames: PipelineFrame[];
  map: [number, number, number][];
  notes: Record<"ambient" | "cinematic" | "classical" | "jazz" | "electronic", Note[]>;
  movement: number[];
  /** [x, y, pattern group, clip index] per sample */
  path: [number, number, number, number][];
  lat: number[][];
  times: number[];
  dt: number;
  dims: number[];
}

export const pipeline = data as unknown as Pipeline;
