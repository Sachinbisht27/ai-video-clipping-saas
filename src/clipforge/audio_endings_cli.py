"""Find possible audio endings for human review, without changing the clip."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from clipforge.candidates_cli import _TranscriptData
from clipforge.services.audio_endings import AudioEndingError, propose_audio_endings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find possible clip endings for listening review; no audio is changed"
    )
    parser.add_argument("audio", type=Path, help="The source PCM WAV used for transcription")
    parser.add_argument("start_ms", type=int, help="Original clip start in source milliseconds")
    parser.add_argument("end_ms", type=int, help="Original clip end in source milliseconds")
    parser.add_argument("output", type=Path, help="Path for a new review JSON file")
    parser.add_argument(
        "--transcript", type=Path, help="Optional matching transcript for word-overlap warnings"
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        print("Error: output must be a new file in an existing directory", file=sys.stderr)
        return 1
    try:
        ranges = None
        if args.transcript is not None:
            transcript = _TranscriptData.model_validate_json(
                args.transcript.read_text(encoding="utf-8")
            )
            ranges = tuple((word.start_ms, word.end_ms) for word in transcript.words)
        review = propose_audio_endings(
            args.audio, args.start_ms, args.end_ms, protected_ranges=ranges
        )
    except AudioEndingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError, UnicodeError, ValidationError:
        print("Error: could not read a valid transcript JSON file", file=sys.stderr)
        return 1

    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(asdict(review), output, indent=2)
                output.write("\n")
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
    except OSError:
        print("Error: could not create the review file", file=sys.stderr)
        return 1
    print(f"Saved {len(review.suggestions)} unverified ending options to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
