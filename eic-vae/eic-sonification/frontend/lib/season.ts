import tracksJson from "@/public/jukebox/tracks.json";
import signalsJson from "@/public/jukebox/signals.json";
import type { Source } from "@/components/jukebox/Player";
import type { Signals, Track } from "./jukebox";

const tracks = tracksJson as unknown as Track[];
const signals = signalsJson as unknown as Signals;

/** How many moments of the animation the model looked at. */
export const frameCount = signals.frames;

/** The bundled hurricane season: five styles scored from the same learned patterns. */
export function getSeason(): Source {
  return {
    id: "season",
    title: "Hear a hurricane season.",
    lede: "A model watched NASA's animation of the 2022 Atlantic hurricane season and learned its visual patterns. A composer turned those patterns into music. Pick a style and press play: the footage, the score and the timeline move together.",
    credit: "Animation: NASA/Goddard Space Flight Center Scientific Visualization Studio, the 2022 hurricane season (SVS 5097).",
    video: "/jukebox/video/hurricane.mp4",
    poster: "/jukebox/video/poster.jpg",
    signals,
    tracks,
    scenes: true,
    svs: "https://svs.gsfc.nasa.gov/5097/",
  };
}
