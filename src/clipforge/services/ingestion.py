"""Validate local videos and prepare audio for transcription."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


class IngestionError(Exception):
    """Base class for expected local ingestion failures."""


class InvalidVideoError(IngestionError):
    """The input is missing or is not a supported video with audio."""


class MediaToolError(IngestionError):
    """A required media tool is missing or failed."""


class OutputError(IngestionError):
    """The output cannot be safely created."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    duration_seconds: float
    size_bytes: int


def _run_tool(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise MediaToolError(f"{args[0]} is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaToolError(f"{args[0]} timed out") from exc
    except OSError as exc:
        raise MediaToolError(f"Could not run {args[0]}") from exc


def probe_video(path: Path) -> VideoMetadata:
    """Check a local MP4 and return its duration and file size.

    A usable input must contain both video and audio streams. This function is
    intended for trusted, locally selected files, not arbitrary API paths.
    """
    path = Path(path)
    if path.suffix.lower() != ".mp4":
        raise InvalidVideoError("Input must be an MP4 file")
    try:
        if not path.is_file():
            raise InvalidVideoError("Input file does not exist or is not a regular file")
        source = path.resolve(strict=True)
        size = source.stat().st_size
    except OSError as exc:
        raise InvalidVideoError("Cannot access input file") from exc
    if size <= 0:
        raise InvalidVideoError("Input file is empty")

    result = _run_tool(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-show_entries",
            "format=format_name,duration:stream=codec_type",
            "-of",
            "json",
            str(source),
        ],
        timeout=30,
    )
    if result.returncode != 0:
        raise InvalidVideoError("Could not read the video; it may be invalid or corrupted")

    try:
        data = json.loads(result.stdout)
        formats = data["format"]["format_name"].split(",")
        duration = float(data["format"]["duration"])
        streams = {stream["codec_type"] for stream in data["streams"]}
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise InvalidVideoError("Video metadata is missing or invalid") from exc
    if "mp4" not in formats or not math.isfinite(duration) or duration <= 0:
        raise InvalidVideoError("Input is not a valid MP4 with a positive duration")
    if not {"video", "audio"}.issubset(streams):
        raise InvalidVideoError("Video must contain both video and audio streams")
    return VideoMetadata(duration_seconds=duration, size_bytes=size)


def extract_audio(source: Path, destination: Path) -> VideoMetadata:
    """Extract mono 16 kHz PCM WAV without overwriting an existing file.

    Output is staged in the destination directory and published only after
    FFmpeg succeeds, so an interrupted conversion does not leave a final file.
    """
    metadata = probe_video(source)
    destination = Path(destination)
    if destination.suffix.lower() != ".wav":
        raise OutputError("Output must have a .wav extension")
    if destination.exists() or destination.is_symlink():
        raise OutputError("Output already exists")
    if not destination.parent.is_dir():
        raise OutputError("Output directory does not exist")

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".clipforge-audio-", suffix=".wav", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        os.close(descriptor)
        result = _run_tool(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(Path(source).resolve(strict=True)),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                "-y",
                str(temporary_path.resolve()),
            ],
            timeout=3600,
        )
        if result.returncode != 0:
            raise MediaToolError("FFmpeg could not extract audio from the video")
        if temporary_path.stat().st_size <= 44:
            raise MediaToolError("FFmpeg produced an empty audio file")
        # Linking is atomic and fails if another process creates the output.
        os.link(temporary_path, destination)
    except FileExistsError as exc:
        raise OutputError("Output already exists") from exc
    except OSError as exc:
        raise OutputError("Could not create the output file") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return metadata
