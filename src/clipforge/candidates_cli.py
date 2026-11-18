"""Create unranked candidate windows from a local transcript and its WAV."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from clipforge.services.candidates import CandidateSelectionError, select_candidates
from clipforge.services.transcription import InvalidAudioError, TranscriptWord, _wav_duration


class _WordData(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    speaker: int | None = None
    timing_adjusted: bool = False
    original_start_ms: int | None = None
    original_end_ms: int | None = None


class _TranscriptData(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    words: list[_WordData]


def main() -> int:
    parser = argparse.ArgumentParser(description="Find unranked candidate windows in a transcript")
    parser.add_argument("transcript", type=Path, help="Transcript JSON from transcribe_cli")
    parser.add_argument("audio", type=Path, help="The same WAV used to produce the transcript")
    parser.add_argument("output", type=Path, help="Path for a new candidate JSON file")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        print("Error: output must be a new file in an existing directory", file=sys.stderr)
        return 1
    try:
        if args.audio.suffix.lower() != ".wav" or not args.audio.is_file():
            raise InvalidAudioError("Choose the WAV used for transcription")
        frames, rate = _wav_duration(args.audio)
        duration_ms = frames * 1000 // rate
        transcript = _TranscriptData.model_validate_json(
            args.transcript.read_text(encoding="utf-8")
        )
        words = tuple(TranscriptWord(**word.model_dump()) for word in transcript.words)
        candidates = select_candidates(words, duration_ms)
    except InvalidAudioError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except CandidateSelectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError, UnicodeError, ValidationError, ValueError:
        print("Error: could not read a valid transcript JSON file", file=sys.stderr)
        return 1

    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "audio_duration_ms": duration_ms,
                        "min_duration_ms": 20_000,
                        "max_duration_ms": 60_000,
                        "candidates": [asdict(candidate) for candidate in candidates],
                    },
                    output,
                    ensure_ascii=False,
                    indent=2,
                )
                output.write("\n")
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
    except OSError:
        print("Error: could not create the candidate file", file=sys.stderr)
        return 1
    print(f"Found {len(candidates)} unranked candidates in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
