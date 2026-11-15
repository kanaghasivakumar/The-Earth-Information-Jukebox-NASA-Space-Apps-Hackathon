"use client";

import Link from "next/link";
import { Player, type Source } from "./Player";
import { useSource } from "./SourceProvider";

export function Listen({ season }: { season: Source }) {
  const { upload, uploadName, reset } = useSource();
  const source = upload ?? season;
  return (
    <>
      {uploadName ? (
        <p className="jb-using jb-using-bar">
          Playing your upload <strong>{uploadName}</strong>.{" "}
          <button type="button" className="jb-link" onClick={reset}>
            Back to the hurricane season
          </button>{" "}
          · <Link href="/upload">Upload another</Link>
        </p>
      ) : null}
      <Player key={source.id} source={source} />
    </>
  );
}
