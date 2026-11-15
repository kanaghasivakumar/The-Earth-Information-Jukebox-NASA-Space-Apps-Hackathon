import Link from "next/link";
import "./jukebox.css";

export default function NotFound() {
  return (
    <div className="jb">
      <div className="jb-section jb-page">
        <h1 className="jb-h1">That page isn&rsquo;t here.</h1>
        <p className="jb-how-lede">The link may be old, or mistyped. These are the places that do exist.</p>
        <ul className="jb-list">
          <li><Link href="/">Listen to the hurricane season</Link></li>
          <li><Link href="/how-it-works">How a picture becomes music</Link></li>
          <li><Link href="/upload">Upload your own visualization</Link></li>
          <li><Link href="/accessibility">Accessibility and keyboard shortcuts</Link></li>
        </ul>
      </div>
    </div>
  );
}
