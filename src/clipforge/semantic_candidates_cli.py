"""Experimental command to propose clip boundaries using transcript text."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from clipforge.candidates_cli import _TranscriptData
from clipforge.services.groq_boundaries import GroqBoundaryError, GroqBoundaryProposer
from clipforge.services.openai_boundaries import OpenAIBoundaryError, OpenAIBoundaryProposer
from clipforge.services.semantic_candidates import SemanticCandidateError, validate_proposals
from clipforge.services.semantic_windowed import WindowedResult, analyze_windows
from clipforge.services.transcript_windows import TranscriptWindowError
from clipforge.services.transcription import InvalidAudioError, TranscriptWord, _wav_duration

_MODELS = {"openai": "gpt-6.2-sol", "groq": "openai/gpt-oss-120b"}


class Settings(BaseSettings):
    """Read the local API key without exposing it in the settings representation."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental: send transcript text to a selected provider to propose clip boundaries. "
            "Requests may incur API charges."
        )
    )
    parser.add_argument("transcript", type=Path, help="Transcript JSON from transcribe_cli")
    parser.add_argument("audio", type=Path, help="The WAV used to produce the transcript")
    parser.add_argument("output", type=Path, help="Path for a new candidate JSON file")
    parser.add_argument(
        "--provider",
        choices=("openai", "groq"),
        default="openai",
        help="AI service receiving the transcript (default: openai)",
    )
    parser.add_argument(
        "--length", choices=("short", "long", "both"), help="Preferred clip lengths (Groq only)"
    )
    parser.add_argument(
        "--windowed", action="store_true", help="Process overlapping sections (Groq only)"
    )
    parser.add_argument("--start-window", type=int, default=0, help="First window index to process")
    parser.add_argument("--max-windows", type=int, help="Maximum windows to process in this run")
    args = parser.parse_args()
    if (
        (args.provider != "groq" and (args.windowed or args.length is not None))
        or (not args.windowed and (args.start_window != 0 or args.max_windows is not None))
        or args.start_window < 0
        or (args.max_windows is not None and args.max_windows <= 0)
    ):
        print("Error: invalid provider or window options", file=sys.stderr)
        return 1
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
        # Fail locally before sending any transcript text to a provider.
        validate_proposals(words, duration_ms, ())
    except InvalidAudioError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except SemanticCandidateError:
        print("Error: invalid transcript word or timing", file=sys.stderr)
        return 1
    except OSError, UnicodeError, ValidationError, ValueError:
        print("Error: could not read a valid transcript JSON file", file=sys.stderr)
        return 1

    windowed_result: WindowedResult | None = None
    rejected_proposals = 0
    duplicate_proposals = 0
    if not words and args.windowed and args.start_window:
        print("Error: starting window is beyond the transcript", file=sys.stderr)
        return 1
    if words:
        try:
            settings = Settings()
            key_name = "OPENAI_API_KEY" if args.provider == "openai" else "GROQ_API_KEY"
            api_key = (
                settings.openai_api_key.get_secret_value()
                if args.provider == "openai"
                else settings.groq_api_key.get_secret_value()
            )
            if not api_key.strip():
                print(f"Error: set {key_name} before proposing boundaries", file=sys.stderr)
                return 1
            if args.windowed:
                windowed_result = analyze_windows(
                    words,
                    duration_ms,
                    GroqBoundaryProposer(api_key),
                    duration_mode=args.length or "both",
                    start_window=args.start_window,
                    max_windows=args.max_windows,
                )
                candidates = windowed_result.candidates
                rejected_proposals = sum(w.rejected for w in windowed_result.windows)
                duplicate_proposals = sum(w.duplicates for w in windowed_result.windows)
                usage = None
            else:
                if args.provider == "openai":
                    result = OpenAIBoundaryProposer(api_key, model=_MODELS["openai"]).propose(words)
                elif args.length is None:
                    result = GroqBoundaryProposer(api_key).propose(words)
                else:
                    result = GroqBoundaryProposer(api_key).propose(
                        words, duration_mode=args.length, input_format="segments"
                    )
                candidates = validate_proposals(
                    words,
                    duration_ms,
                    result.proposals,
                    target_min_ms=60_000 if args.length == "long" else 20_000,
                    target_max_ms=(
                        180_000 if args.provider == "groq" and args.length != "short" else 60_000
                    ),
                )
                rejected_proposals = result.rejected_proposals
                duplicate_proposals = result.duplicate_proposals
                usage = None if result.usage is None else asdict(result.usage)
        except SemanticCandidateError:
            print("Error: the proposed boundaries failed validation", file=sys.stderr)
            return 1
        except (OpenAIBoundaryError, GroqBoundaryError) as exc:
            if str(exc) == "Transcript is too large for one model request":
                message = "transcript is too large for one request; use a shorter excerpt"
            elif str(exc) == "Model provider rejected the request (HTTP 429)":
                message = "provider rate limit reached; check your account quota"
            elif str(exc) == "Model could not return structured boundaries":
                message = "provider could not return structured boundaries; try a shorter excerpt"
            else:
                message = "the boundary provider could not produce valid proposals"
            print(f"Error: {message}", file=sys.stderr)
            return 1
        except ValidationError, SettingsError, OSError, UnicodeError:
            print("Error: could not load boundary selection settings", file=sys.stderr)
            return 1
        except TranscriptWindowError, ValueError:
            print("Error: invalid window selection or transcript", file=sys.stderr)
            return 1
    else:
        candidates = ()
        usage = None

    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "provider": args.provider if words else None,
                        "model": _MODELS[args.provider] if words else None,
                        "audio_duration_ms": duration_ms,
                        "target_min_ms": 60_000 if args.length == "long" else 20_000,
                        "target_max_ms": (
                            180_000
                            if args.provider == "groq" and args.length != "short"
                            else 60_000
                        ),
                        "usage": usage,
                        "rejected_proposals": rejected_proposals,
                        "duplicate_proposals": duplicate_proposals,
                        "duration_mode": (args.length or "both")
                        if args.provider == "groq"
                        else None,
                        "windowing": (
                            {
                                "total_windows": windowed_result.total_windows,
                                "next_window_index": windowed_result.next_window_index,
                                "processing_complete": windowed_result.processing_complete,
                                "windows": [asdict(window) for window in windowed_result.windows],
                            }
                            if windowed_result is not None
                            else None
                        ),
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
    print(f"Found {len(candidates)} unranked candidates in {args.output}; review them before use")
    if rejected_proposals or (
        windowed_result is not None and not windowed_result.processing_complete
    ):
        print(
            "Some windows were not processed or proposals were rejected; inspect output",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
