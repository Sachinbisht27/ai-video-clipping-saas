"""Local command-line entry point for media preparation."""

import argparse
import sys
from pathlib import Path

from clipforge.services.ingestion import IngestionError, extract_audio


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract audio from a local MP4 video")
    parser.add_argument("video", type=Path, help="Path to the source MP4")
    parser.add_argument("audio", type=Path, help="Path for the new WAV file")
    args = parser.parse_args()
    try:
        metadata = extract_audio(args.video, args.audio)
    except IngestionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Extracted audio from {metadata.duration_seconds:.2f}s video to {args.audio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
