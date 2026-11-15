import type { Metadata } from "next";
import Link from "next/link";
import voices from "@/lib/data/voice-credits.json";
import { CATALOGUE, TRACK_COPY } from "@/lib/data/copy";

export const metadata: Metadata = {
  title: "Credits",
  description: "The NASA visualizations, models, instrument sounds and field recordings behind Earth Jukebox, with their licences.",
};

interface Voice {
  group: string;
  title: string;
  author: string;
  license: string;
  license_url: string | null;
  source: string;
  url: string | null;
}

const VOICES = voices as Voice[];
const GROUPS = [
  ["geophony", "Earth sounds (geophony)"],
  ["anthrophony", "Human sounds (anthrophony)"],
  ["biophony", "Living sounds (biophony)"],
] as const;

export default function CreditsPage() {
  return (
    <div className="jb-section jb-page">
      <h1 className="jb-h1">Who made the sounds.</h1>
      <p className="jb-how-lede">
        Earth Jukebox stands on NASA&rsquo;s visualizations, open music models and recordings that people chose to share. This is the full list,
        with the licence each one comes under.
      </p>

      <p className="jb-honest jb-powered">
        <strong>Powered by Stability AI.</strong> The MuseControlLite versions on the Compare page are generated with Stable Audio Open 1.0 under the
        Stability AI Community License. The weights are not redistributed here.
      </p>

      <h2 className="jb-how-sub">Visualizations and data</h2>
      <ul className="jb-list">
        <li>
          <strong>Hurricane season, 2022.</strong> NASA/Goddard Space Flight Center Scientific Visualization Studio,{" "}
          <a href="https://svs.gsfc.nasa.gov/5097/">SVS 5097</a>. Scored by our variational autoencoder: an artistic sonification.
        </li>
        {CATALOGUE.map((c) => (
          <li key={c.id}>
            <strong>{TRACK_COPY[c.id]?.name ?? c.title}.</strong> &ldquo;{c.title}&rdquo;, NASA Scientific Visualization Studio,{" "}
            {c.svs_url ? <a href={c.svs_url}>SVS {c.id}</a> : `SVS ${c.id}`}. Data: {c.data}.
          </li>
        ))}
      </ul>
      <p className="jb-fine jb-credits-note">
        NASA media are not copyrighted; NASA asks for credit. This is an independent project and is not endorsed by NASA.
      </p>

      <h2 className="jb-how-sub">Models and instrument sounds</h2>
      <ul className="jb-list">
        <li>
          <strong>Stable Audio Open 1.0</strong> by Stability AI, Stability AI Community License. Trained on Freesound and the Free Music Archive (CC).
        </li>
        <li>
          <strong>MuseControlLite</strong>, code and adapters under the MIT licence. Tsai et al., <em>MuseControlLite</em>, ICML 2025 (arXiv:2506.18729).
        </li>
        <li>
          <strong>Magenta RealTime 2</strong> by Google, weights under CC BY 4.0, code under Apache-2.0. Magenta team, <em>Live Music Models</em>,
          arXiv:2508.04651. Google claims no rights in the generated mood layers.
        </li>
        <li>
          <strong>Anticipatory Music Transformer</strong> (stanford-crfm/music-medium-800k), Apache-2.0. Thickstun, Hall, Donahue and Liang, 2023. It
          writes the harmony and bass; it never changes a data note.
        </li>
        <li>
          <strong>FluidR3_GM SoundFont</strong> by Frank Wen and Toby Smithe, MIT licence, rendered with FluidSynth. Every data instrument you hear.
        </li>
        <li>
          <strong>SigLIP</strong> (google/siglip-base-patch16-224), Apache-2.0, Zhai et al., ICCV 2023: it weighs the mood prompts from the video frames.{" "}
          <strong>DINOv2</strong>, Apache-2.0, Oquab et al., 2023: it finds the section changes in 3912.
        </li>
      </ul>

      <h2 className="jb-how-sub">Field recordings</h2>
      <p className="jb-how-lede">
        {VOICES.length} recordings, cut into short grains and shaped by the data. Only public-domain, CC0 and CC BY material is used. MBARI hydrophone
        data: Ryan et al., <em>Ocean Sound in Monterey Bay</em>, OCEANS 2016. National Park Service recordings are in the public domain; credit the
        National Park Service.
      </p>
      {GROUPS.map(([g, label]) => {
        const rows = VOICES.filter((v) => v.group === g);
        return (
          <details key={g} className="jb-details jb-credits">
            <summary>
              {label}, {rows.length} recordings
            </summary>
            <ul className="jb-list">
              {rows.map((v) => (
                <li key={`${v.title}-${v.author}`}>
                  {v.url ? <a href={v.url}>{v.title}</a> : v.title}, {v.author}.{" "}
                  {v.license_url ? <a href={v.license_url}>{v.license}</a> : v.license}. {v.source}.
                </li>
              ))}
            </ul>
          </details>
        );
      })}

      <p className="jb-next">
        <Link href="/">Back to the player</Link> · <Link href="/accessibility">Accessibility</Link>
      </p>
    </div>
  );
}
