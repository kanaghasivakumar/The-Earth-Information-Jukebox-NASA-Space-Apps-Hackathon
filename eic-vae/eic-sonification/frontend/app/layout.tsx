import type { Metadata, Viewport } from "next";
import { Figtree, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const figtree = Figtree({ variable: "--font-figtree", subsets: ["latin"], weight: ["400", "500", "600", "700", "800"] });
const jbmono = JetBrains_Mono({ variable: "--font-jbmono", subsets: ["latin"] });

export const viewport: Viewport = { colorScheme: "dark", themeColor: "#0c1116" };

export const metadata: Metadata = {
  title: { default: "Earth Jukebox · Hear a hurricane season", template: "%s · Earth Jukebox" },
  description:
    "A NASA hurricane-season visualization turned into five styles of music by a model that learned its patterns. Built for the NASA Space Apps Challenge.",
};


export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${figtree.variable} ${jbmono.variable} h-full antialiased`}
    >
      <body style={{ margin: 0, minHeight: "100%", background: "#0c1116" }}>{children}</body>
    </html>
  );
}
