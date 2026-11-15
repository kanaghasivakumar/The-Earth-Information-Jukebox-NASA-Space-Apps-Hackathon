import type { Metadata } from "next";
import Link from "next/link";
import { FrameDive } from "@/components/jukebox/FrameDive";
import { Pipeline } from "@/components/jukebox/Pipeline";
import { frameCount } from "@/lib/season";

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
      <p className="jb-next">
        <Link href="/">Back to the player</Link> · <Link href="/upload">Try your own visualization</Link>
      </p>
    </div>
  );
}
