import type { Metadata } from "next";
import Link from "next/link";
import { FrameDive } from "@/components/jukebox/FrameDive";
import { Pipeline } from "@/components/jukebox/Pipeline";
import { frameCount } from "@/lib/season";
import { CATALOGUE, TRACK_COPY } from "@/lib/data/copy";
import { fmtNum } from "@/lib/data/music";

export const metadata: Metadata = {
  title: "How it works",
  description: "Follow one frame of a NASA hurricane animation through a variational autoencoder, then watch 30 seconds of it become music.",
};

const frames = frameCount.toLocaleString("en-US");

export default function HowItWorksPage() {
  return (
    <div className="jb-section jb-page">
      <h1 className="jb-h1">How a picture becomes music.</h1>
      <p className="jb-how-lede">
        Everything on this page is real: the frames, the pixels the model saw, the numbers it produced and the notes the
        composer wrote. Start with a single moment. Then press play and watch thirty seconds of the storm turn into notes.
      </p>

      <h2 className="jb-how-sub">One moment, up close</h2>
      <FrameDive />

      <h2 className="jb-how-sub">Now let it play.</h2>
      <Pipeline />

      <h2 className="jb-how-sub">The short version</h2>
      <ol className="jb-steps">
        <li>
          <h3>It watches.</h3>
          <p>
            The animation is sampled {frames} times, about five times a second. For each moment the model sees the picture and
            how much it moved since the last one.
          </p>
        </li>
        <li>
          <h3>It learns patterns.</h3>
          <p>
            A variational autoencoder squeezes each moment into eight numbers. They are learned patterns of how the picture
            looks, not temperature, rainfall or wind speed. Moments that look alike get grouped, and those are the lettered
            pattern groups on the timeline.
          </p>
        </li>
        <li>
          <h3>It composes.</h3>
          <p>
            When the picture changes fast, the music gets busier and louder. When a pattern group returns, a short motif from
            its first appearance returns with it. Each style picks the tempo, scale and instruments.
          </p>
        </li>
      </ol>
      <p className="jb-honest">
        This is an artistic sonification of imagery. It is not a forecast, a measurement, or a reading of the storm. The choice
        of key, chords and instruments is ours.
      </p>

      <h2 className="jb-how-sub" id="data-tracks">
        The data tracks: every note is a number.
      </h2>
      <p className="jb-how-lede">
        The eight other tracks on the player work the other way round. Nothing is learned from how the picture looks. The values the picture shows are
        read back out of it, and each one becomes a note you can check. We call these tracks data-faithful.
      </p>
      <ol className="jb-steps">
        <li>
          <h3>It reads the colours.</h3>
          <p>
            Each pixel&rsquo;s colour is matched against the visualization&rsquo;s own colour bar and turned back into a number: degrees, metres,
            flashes. Where NASA publishes the table behind a movie, we check against it. For the warming track the match is r 0.9997.
          </p>
        </li>
        <li>
          <h3>Numbers become notes.</h3>
          <p>
            Every value becomes one note. A higher value plays a higher pitch, snapped to a scale. A flute plays the main series; harps, pads and drums
            play the latitude bands and the events.
          </p>
        </li>
        <li>
          <h3>A model adds harmony.</h3>
          <p>
            The Anticipatory Music Transformer writes strings and a bass line around the melody. It never changes a data note, stays below the lowest
            one, and carries no data.
          </p>
        </li>
        <li>
          <h3>The Earth joins in.</h3>
          <p>
            Recorded wind, water and thunder, traffic and city hum, birds, frogs and insects get louder and softer with the data: the Earth&rsquo;s,
            people&rsquo;s and living things&rsquo; voices.
          </p>
        </li>
        <li>
          <h3>We check it.</h3>
          <p>
            A pitch tracker listens to the finished mix and turns each melody note back into a number. The track passes when that matches the data.
            To be sure the test is not fooled by any tune, we also ran it on the same notes shuffled in time, which fail it.
          </p>
        </li>
      </ol>

      <section className="jb-panel jb-fid-all" aria-labelledby="fid-all-h">
        <h3 id="fid-all-h" className="jb-h2">
          How closely each melody follows its data
        </h3>
        <table className="jb-table jb-fid-table">
          <caption className="jb-sr">R squared of the melody recovered from the full mix against the data. 1 is a perfect match.</caption>
          <thead>
            <tr>
              <th scope="col">Track</th>
              <th scope="col">R² against the data</th>
            </tr>
          </thead>
          <tbody>
            {CATALOGUE.filter((c) => c.r2 != null).map((c) => (
              <tr key={c.id}>
                <th scope="row">{TRACK_COPY[c.id]?.name ?? c.title}</th>
                <td>
                  <span className="jb-fid-track" aria-hidden="true">
                    <span className="jb-fid-fill" style={{ width: `${Math.max(0, c.r2 as number) * 100}%` }} />
                  </span>
                  <span className="jb-code">{fmtNum(c.r2 as number)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="jb-fine">
          R² is the share of the data&rsquo;s ups and downs that the melody&rsquo;s pitch explains: 1 is a perfect match, and 0 or below means it does
          not follow at all. The aerosol track has no single melody; each of its four series is checked on its own.
        </p>
      </section>

      <p className="jb-honest">
        So there are two ways of listening here. The hurricane season is an artistic sonification of imagery: its eight numbers are learned patterns,
        and the key, chords and instruments are our choices. The data tracks are data-faithful: the melody is the measured value, and the check above
        shows it. Learned layers on top of them are optional versions you can <Link href="/compare">compare</Link>.
      </p>
      <p className="jb-next">
        <Link href="/">Back to the player</Link> · <Link href="/upload">Try your own visualization</Link> · <Link href="/credits">Credits</Link>
      </p>
    </div>
  );
}
