"use client";

import { useRouter } from "next/navigation";
import { UploadPanel } from "./UploadPanel";
import { API, useSource } from "./SourceProvider";

/** On success the result becomes the shared source and we move to the player. */
export function UploadView() {
  const router = useRouter();
  const { uploadName, applyUpload, reset } = useSource();
  return (
    <UploadPanel
      api={API}
      current={uploadName}
      onReady={(u, filename) => {
        applyUpload(u, filename);
        router.push("/");
      }}
      onReset={reset}
    />
  );
}
