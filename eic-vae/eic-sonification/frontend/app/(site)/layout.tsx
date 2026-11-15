import Link from "next/link";
import { SiteNav } from "@/components/site/SiteNav";
import { SourceProvider } from "@/components/jukebox/SourceProvider";
import "../jukebox.css";

export default function SiteLayout({ children }: LayoutProps<"/">) {
  return (
    <div className="jb">
      <a href="#main" className="jb-skip-link">
        Skip to the content
      </a>
      <header className="jb-bar">
        <div className="jb-bar-in">
          <Link href="/" className="jb-brand" aria-label="Earth Jukebox, home">
            <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
              <circle cx="14" cy="14" r="12" fill="none" stroke="currentColor" strokeWidth="2" />
              <path d="M5 14h3l2-5 3 10 3-8 2 3h5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span>
              Earth Jukebox
              <small>NASA Space Apps Challenge</small>
            </span>
          </Link>
          <SiteNav />
        </div>
      </header>

      <SourceProvider>
        <main id="main">{children}</main>
      </SourceProvider>

      <footer className="jb-foot">
        <div className="jb-foot-in">
          <p>
            Built for the NASA Space Apps Challenge. This is an independent project and is not endorsed by NASA. Animation
            credit: NASA/Goddard Space Flight Center Scientific Visualization Studio.
          </p>
          <p>
            <Link href="/how-it-works">How it works</Link> · <Link href="/accessibility">Accessibility</Link>
          </p>
        </div>
      </footer>
    </div>
  );
}
