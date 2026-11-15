/** Words for the data tracks, in the voice of the rest of the site. Numbers come from catalogue.json. */
import catalogueJson from "./catalogue.json";

export interface CatalogueEntry {
  id: string;
  title: string;
  voice: string | null;
  summary: string | null;
  duration_s: number;
  svs_url: string | null;
  /** full-mix R² of the data melody, or null when the track has no single melody (5669) */
  r2: number | null;
  control_r2: number | null;
  versions: number;
  learned: boolean;
  /** the underlying dataset, in one line */
  data: string;
}

export const CATALOGUE = catalogueJson as CatalogueEntry[];

export const TRACK_COPY: Record<string, { name: string; heading: string; lede: string }> = {
  "5209": {
    name: "Warming, 1880 to 2023",
    heading: "Hear 144 years of warming.",
    lede: "One note a year. The flute plays the global temperature anomaly and eight harps play the latitude zones, from the Arctic down to Antarctica. Higher means warmer.",
  },
  "5313": {
    name: "Night lights",
    heading: "Hear the lights come on.",
    lede: "Night lights seen from space, 2012 to 2023. The flute rises as the world gets brighter, the city hum grows, and the birds and insects fade.",
  },
  "3912": {
    name: "Ocean temperature",
    heading: "Hear the ocean warm and cool.",
    lede: "The flute follows the global sea surface temperature, frame by frame, over a hydrophone in the Pacific. Higher means warmer.",
  },
  "5011": {
    name: "Lightning from the ISS",
    heading: "Hear six years of lightning.",
    lede: "Every drum hit is a cluster of new flashes seen from the space station, panned to where the storms are. The flute follows the storm belt north and south with the seasons.",
  },
  "4950": {
    name: "Forest height",
    heading: "Hear the forests go by.",
    lede: "The globe turns under a lidar in orbit. The flute follows how tall the visible forest is, and a forest chorus swells over the rainforests.",
  },
  "5669": {
    name: "Aerosols",
    heading: "Hear what is in the air.",
    lede: "One snapshot of the world's aerosols, swept from west to east. Each kind has its own instrument: double bass for sea salt, clarinet for dust, oboe for sulfate, harp for smoke.",
  },
  "5058": {
    name: "Florida after Ian",
    heading: "Hear a city go dark.",
    lede: "Southwest Florida's lights before and after Hurricane Ian. Trumpet runs fall as the lights go out, the traffic fades, and the frogs and insects take over.",
  },
  "5176": {
    name: "Sea temperature, daily",
    heading: "Hear the ocean run warm.",
    lede: "The daily sea surface temperature anomaly, 2024 to 2026. The flute follows the global ocean's departure from normal; higher means warmer than usual.",
  },
  "5046": {
    name: "Arctic sea ice, daily",
    heading: "Hear the ice melt and return.",
    lede: "Two years of Arctic sea ice, one day per frame, April 2021 to April 2023. The flute is the ice extent: it falls each summer and climbs back each winter. A bowed-glass pad plays the 1981 to 2010 average, so the gap is the missing ice.",
  },
};

export const trackName = (id: string) => TRACK_COPY[id]?.name ?? CATALOGUE.find((c) => c.id === id)?.title ?? id;
export const isDataTrack = (id: string | null | undefined): id is string => !!id && CATALOGUE.some((c) => c.id === id);
