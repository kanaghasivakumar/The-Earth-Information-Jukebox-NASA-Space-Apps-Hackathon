import Link from "next/link";
import { getSeason } from "@/lib/season";
import { Listen } from "@/components/jukebox/Listen";

export default function ListenPage() {
  return (
    <>
      <section id="listen" className="jb-section jb-listen" aria-label="Listen">
        <Listen season={getSeason()} />
      </section>
      <section className="jb-section jb-explore" aria-label="Keep exploring">
        <Link href="/how-it-works" className="jb-explore-link">
          <strong>See how a picture becomes music</strong>
          <span>Follow one frame through the model, then watch 30 seconds of the storm turn into notes.</span>
        </Link>
        <Link href="/upload" className="jb-explore-link">
          <strong>Try your own visualization</strong>
          <span>Upload an Earth animation and hear the same model score it in five styles.</span>
        </Link>
      </section>
    </>
  );
}
