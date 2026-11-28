"""Export a manually selected video clip using a matching source WAV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clipforge.services.video_export import VideoExportError, export_video_clip


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a selected local MP4 clip")
    parser.add_argument("video", type=Path, help="Original MP4 used for audio extraction")
    parser.add_argument("audio", type=Path, help="WAV used for the selected clip's timestamps")
    parser.add_argument("output", type=Path, help="Path for a new MP4 file")
    parser.add_argument("start_ms", type=int, help="Selected start in WAV milliseconds")
    parser.add_argument("end_ms", type=int, help="Selected end in WAV milliseconds")
    args = parser.parse_args()
    try:
        result = export_video_clip(args.video, args.audio, args.output, args.start_ms, args.end_ms)
    except VideoExportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {result.duration_ms / 1000:.3f}s clip to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
