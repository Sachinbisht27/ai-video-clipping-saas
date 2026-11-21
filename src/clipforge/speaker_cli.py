"""Create a corrected transcript using explicit, local speaker-label mappings."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from clipforge.candidates_cli import _TranscriptData
from clipforge.services.speaker_corrections import SpeakerCorrectionError, correct_speakers
from clipforge.services.transcription import TranscriptWord


def _parse_mapping(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)=([0-9]+)", value)
    if match is None:
        raise argparse.ArgumentTypeError("Use SOURCE=TARGET with non-negative speaker labels")
    try:
        return int(match.group(1)), int(match.group(2))
    except ValueError:
        raise argparse.ArgumentTypeError("Speaker labels are too large") from None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a transcript with manually corrected speaker labels"
    )
    parser.add_argument("transcript", type=Path, help="Existing transcript JSON")
    parser.add_argument("output", type=Path, help="Path for a new corrected transcript JSON")
    parser.add_argument(
        "--map",
        action="append",
        required=True,
        type=_parse_mapping,
        metavar="SOURCE=TARGET",
        help="Map every occurrence of a current speaker label to another label; repeat as needed",
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        print("Error: output must be a new file in an existing directory", file=sys.stderr)
        return 1
    mapping: dict[int, int] = {}
    for source, target in args.map:
        if source in mapping:
            print("Error: specify each source speaker label only once", file=sys.stderr)
            return 1
        mapping[source] = target
    try:
        data: object = json.loads(args.transcript.read_text(encoding="utf-8"))
        parsed = _TranscriptData.model_validate(data)
        if not isinstance(data, dict) or not isinstance(data.get("words"), list):
            raise ValueError
        words = tuple(TranscriptWord(**word.model_dump()) for word in parsed.words)
        corrected = correct_speakers(words, mapping)
        changed = 0
        for original_data, before, after in zip(data["words"], words, corrected, strict=True):
            if before.speaker != after.speaker:
                original_data["speaker"] = after.speaker
                original_data["original_speaker"] = after.original_speaker
                changed += 1
    except SpeakerCorrectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError, UnicodeError, ValidationError, ValueError, TypeError:
        print("Error: could not read a valid transcript JSON file", file=sys.stderr)
        return 1
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(data, output, ensure_ascii=False, indent=2)
                output.write("\n")
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
    except OSError:
        print("Error: could not create the corrected transcript", file=sys.stderr)
        return 1
    print(f"Corrected {changed} word labels in {args.output}; original transcript unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
