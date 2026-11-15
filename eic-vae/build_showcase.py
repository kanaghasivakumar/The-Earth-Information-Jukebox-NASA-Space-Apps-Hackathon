#!/usr/bin/env python3
"""Build the static listening page from existing saved outputs (no inference)."""
from __future__ import annotations

import argparse
from pathlib import Path

from music.showcase import build_showcase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='results', help='folder containing saved style WAVs and shared arrays')
    parser.add_argument('--source-video', help='optional local source video for the synchronized preview')
    args = parser.parse_args()
    destination = build_showcase(args.output_dir, project_root=Path(__file__).resolve().parent,
                                 source_video=args.source_video)
    print(f'Wrote {destination}')


if __name__ == '__main__':
    main()
