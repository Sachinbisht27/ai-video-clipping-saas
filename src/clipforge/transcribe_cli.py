"""Explicit local command for producing a timestamped transcript JSON file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from clipforge.services.transcription import DeepgramTranscriber, TranscriptionError


class Settings(BaseSettings):
    """Load the local key without putting it in printable settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepgram_api_key: SecretStr = SecretStr("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe a local WAV using Deepgram")
    parser.add_argument("audio", type=Path, help="Path to the source WAV (sent to Deepgram)")
    parser.add_argument("output", type=Path, help="Path for a new transcript JSON file")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        print("Error: output must be a new file in an existing directory", file=sys.stderr)
        return 1
    try:
        settings = Settings()
        transcript = DeepgramTranscriber(settings.deepgram_api_key.get_secret_value()).transcribe(
            args.audio
        )
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        # Private creation and exclusive mode prevent accidental replacement.
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "provider": transcript.provider,
                        "language": transcript.language,
                        "full_text": transcript.full_text,
                        "words": [asdict(word) for word in transcript.words],
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
        print("Error: could not create the transcript file", file=sys.stderr)
        return 1
    print(f"Transcribed {len(transcript.words)} words to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
