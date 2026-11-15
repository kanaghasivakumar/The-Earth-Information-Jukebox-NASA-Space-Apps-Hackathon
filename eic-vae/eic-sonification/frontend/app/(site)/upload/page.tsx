import type { Metadata } from "next";
import { UploadView } from "@/components/jukebox/UploadView";

export const metadata: Metadata = {
  title: "Upload your own",
  description: "Turn your own Earth visualization into music with the same model and composer.",
};

export default function UploadPage() {
  return <UploadView />;
}
