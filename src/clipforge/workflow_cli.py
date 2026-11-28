"""Run the experimental local clipping workflow with an explicit review step."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import SettingsError

from clipforge.semantic_candidates_cli import Settings as BoundarySettings
from clipforge.transcribe_cli import Settings as TranscriptionSettings


def _run(module: str, *arguments: str) -> int:
    """Run existing commands without a shell and preserve their exit status."""
    try:
        return subprocess.run([sys.executable, "-m", module, *arguments], check=False).returncode
    except OSError:
        print("Error: could not start a workflow step", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimental local workflow. Review suggestions before exporting."
    )
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare", help="Extract, transcribe, and propose clip boundaries")
    prepare.add_argument("video", type=Path, help="Source MP4; audio is sent to Deepgram")
    prepare.add_argument("job", type=Path, help="New private output directory")
    prepare.add_argument("--language", choices=("en", "multi"), default="en")
    prepare.add_argument("--diarizer", choices=("v1", "v2"), default="v2")
    prepare.add_argument(
        "--max-windows", type=int, default=1, help="Groq windows to process (default: 1)"
    )
    resume = actions.add_parser(
        "resume", help="Continue windowed suggestions without retranscribing"
    )
    resume.add_argument("job", type=Path, help="Prepared output directory")
    resume.add_argument("output", type=Path, help="New candidate JSON file")
    resume.add_argument("--start-window", type=int, required=True)
    resume.add_argument("--max-windows", type=int, default=1)
    export = actions.add_parser("export", help="Export a manually reviewed time range")
    export.add_argument("video", type=Path, help="Original MP4 used in preparation")
    export.add_argument("job", type=Path, help="Prepared output directory")
    export.add_argument("output", type=Path, help="New output MP4")
    export.add_argument("start_ms", type=int, help="Reviewed start in WAV-relative milliseconds")
    export.add_argument("end_ms", type=int, help="Reviewed end in WAV-relative milliseconds")
    export.add_argument("--allow-video-tail-hold", action="store_true")
    args = parser.parse_args()
    if args.action == "resume":
        if not args.job.is_dir() or args.job.is_symlink():
            print("Error: choose an existing workflow directory", file=sys.stderr)
            return 1
        if args.start_window < 0 or args.max_windows < 1:
            print("Error: invalid window selection", file=sys.stderr)
            return 1
        print("Transcript text will be sent to Groq; requests may incur charges.", flush=True)
        return _run(
            "clipforge.semantic_candidates_cli",
            str(args.job / "transcript.json"),
            str(args.job / "audio.wav"),
            str(args.output),
            "--provider",
            "groq",
            "--windowed",
            "--start-window",
            str(args.start_window),
            "--max-windows",
            str(args.max_windows),
        )
    if args.action == "export":
        if not args.job.is_dir() or args.job.is_symlink():
            print("Error: choose an existing workflow directory", file=sys.stderr)
            return 1
        extra = ("--allow-video-tail-hold",) if args.allow_video_tail_hold else ()
        return _run(
            "clipforge.video_export_cli",
            str(args.video),
            str(args.job / "audio.wav"),
            str(args.output),
            str(args.start_ms),
            str(args.end_ms),
            *extra,
        )

    if args.max_windows < 1:
        print("Error: --max-windows must be positive", file=sys.stderr)
        return 1
    if not args.video.is_file() or args.video.suffix.lower() != ".mp4":
        print("Error: choose an existing MP4", file=sys.stderr)
        return 1
    try:
        if not TranscriptionSettings().deepgram_api_key.get_secret_value().strip():
            print("Error: set DEEPGRAM_API_KEY before starting", file=sys.stderr)
            return 1
        if not BoundarySettings().groq_api_key.get_secret_value().strip():
            print("Error: set GROQ_API_KEY before starting", file=sys.stderr)
            return 1
    except ValidationError, SettingsError, OSError, UnicodeError:
        print("Error: could not load workflow settings", file=sys.stderr)
        return 1
    try:
        args.job.mkdir(mode=0o700)
    except OSError:
        print("Error: job must be a new directory with an existing parent", file=sys.stderr)
        return 1

    audio, transcript, candidates = (
        args.job / name for name in ("audio.wav", "transcript.json", "candidates.json")
    )
    print(
        "Audio will be sent to Deepgram; transcript text will be sent to Groq. "
        "Requests may incur charges.",
        flush=True,
    )
    steps = (
        ("clipforge.cli", (str(args.video), str(audio))),
        (
            "clipforge.transcribe_cli",
            (str(audio), str(transcript), "--language", args.language, "--diarizer", args.diarizer),
        ),
        (
            "clipforge.semantic_candidates_cli",
            (
                str(transcript),
                str(audio),
                str(candidates),
                "--provider",
                "groq",
                "--windowed",
                "--max-windows",
                str(args.max_windows),
            ),
        ),
    )
    for module, arguments in steps:
        status = _run(module, *arguments)
        if status:
            print(
                f"Workflow stopped at {module}; inspect {args.job}. Existing files were kept.",
                file=sys.stderr,
            )
            return status
    print(f"Inspect {candidates}; it may contain no suggestions. Listen before export.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
