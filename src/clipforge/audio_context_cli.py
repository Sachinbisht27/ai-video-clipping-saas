"""Create a local preview with bounded audio after an existing clip."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from clipforge.services.audio_context import AudioContextError, create_context_preview


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview audio after a clip ending; no reaction is detected or selected"
    )
    parser.add_argument("audio", type=Path, help="Source mono 16 kHz PCM WAV")
    parser.add_argument("start_ms", type=int, help="Original clip start in source milliseconds")
    parser.add_argument("end_ms", type=int, help="Original clip end in source milliseconds")
    parser.add_argument(
        "output", type=Path, help="New WAV preview; never overwrites existing files"
    )
    parser.add_argument(
        "--context-ms", type=int, default=3000, help="Additional audio in milliseconds (0–5000)"
    )
    args = parser.parse_args()
    try:
        preview = create_context_preview(
            args.audio, args.start_ms, args.end_ms, args.context_ms, args.output
        )
    except AudioContextError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(preview), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
