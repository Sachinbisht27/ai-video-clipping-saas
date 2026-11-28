"""Export a locally reviewed interval from an aligned MP4 and reference WAV.

The WAV is checked against a fresh decode of the supplied video audio. The
caller must still use the transcript derived from that WAV and choose the
intended visual source; matching audio cannot prove transcript or video origin.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from clipforge.services.ingestion import IngestionError, _run_tool, probe_video


class VideoExportError(Exception):
    """A source, timing, or export operation is not supported."""


@dataclass(frozen=True, slots=True)
class ExportMetadata:
    """The requested WAV-relative range; video frames are quantized to source frames."""

    start_ms: int
    end_ms: int
    duration_ms: int


_TIMING_TOLERANCE_SECONDS = 0.002
_ZERO_TOLERANCE_SECONDS = 0.0001


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _float(value: object) -> float:
    try:
        result = float(str(value))
    except (ValueError, TypeError) as exc:
        raise VideoExportError("Source timestamp metadata is missing or invalid") from exc
    if not math.isfinite(result):
        raise VideoExportError("Source timestamp metadata is missing or invalid")
    return result


def _streams(video: Path) -> tuple[Fraction, int]:
    """Require one zero-aligned CFR video stream and one audio stream."""
    try:
        probe_video(video)
        result = _run_tool(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-show_entries",
                "format=start_time:stream=codec_type,start_time,sample_rate,avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(video.resolve(strict=True)),
            ],
            timeout=30,
        )
    except (IngestionError, OSError) as exc:
        raise VideoExportError("Could not inspect source video") from exc
    if result.returncode:
        raise VideoExportError("Could not inspect source video")
    try:
        data = json.loads(result.stdout)
        if abs(_float(data["format"]["start_time"])) > _ZERO_TOLERANCE_SECONDS:
            raise VideoExportError("Source timeline is not zero-aligned")
        videos = [s for s in data["streams"] if s.get("codec_type") == "video"]
        audios = [s for s in data["streams"] if s.get("codec_type") == "audio"]
        if len(videos) != 1 or len(audios) != 1:
            raise VideoExportError("Source must have exactly one video and one audio stream")
        if any(
            abs(_float(s["start_time"])) > _ZERO_TOLERANCE_SECONDS for s in (videos[0], audios[0])
        ):
            raise VideoExportError("Source streams are not zero-aligned")
        video_rate = Fraction(videos[0]["avg_frame_rate"])
        nominal_rate = Fraction(videos[0]["r_frame_rate"])
        audio_rate = int(audios[0]["sample_rate"])
        if video_rate != nominal_rate or not 1 <= video_rate <= 240 or audio_rate <= 0:
            raise VideoExportError("Source frame or sample rate is unsupported")
    except VideoExportError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError) as exc:
        raise VideoExportError("Source timestamp metadata is missing or invalid") from exc
    return video_rate, audio_rate


def _continuous_end(video: Path, selector: str, rate: Fraction | int) -> Fraction:
    """Check every decoded presentation timestamp and return the stream end."""
    audio = selector == "a:0"
    entries = (
        "frame=best_effort_timestamp_time,nb_samples"
        if audio
        else "frame=best_effort_timestamp_time"
    )
    try:
        result = _run_tool(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-select_streams",
                selector,
                "-show_frames",
                "-show_entries",
                entries,
                "-of",
                "csv=p=0",
                str(video.resolve(strict=True)),
            ],
            timeout=3600,
        )
    except (IngestionError, OSError) as exc:
        raise VideoExportError("Could not verify source timestamps") from exc
    if result.returncode:
        raise VideoExportError("Could not verify source timestamps")
    elapsed = Fraction(0)
    frames = 0
    try:
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            actual = _float(parts[0])
            tolerance = _ZERO_TOLERANCE_SECONDS if frames == 0 else _TIMING_TOLERANCE_SECONDS
            if abs(actual - float(elapsed)) > tolerance:
                raise VideoExportError("Source timestamps are discontinuous or not zero-aligned")
            if audio:
                samples = int(parts[1])
                if samples <= 0:
                    raise ValueError
                elapsed += Fraction(samples, int(rate))
            else:
                elapsed += 1 / Fraction(rate)
            frames += 1
    except VideoExportError:
        raise
    except (ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
        raise VideoExportError("Source frame timestamps are missing or invalid") from exc
    if not frames:
        raise VideoExportError("Source has no decodable frames")
    return elapsed


def _verify_wav(video: Path, audio: Path) -> Fraction:
    """Compare every WAV sample with a fresh decode using ingestion settings."""
    try:
        if audio.suffix.lower() != ".wav" or not audio.is_file():
            raise VideoExportError("Reference audio must be an existing WAV file")
        with wave.open(str(audio), "rb") as reference, tempfile.TemporaryFile() as decoded:
            if (
                reference.getnchannels() != 1
                or reference.getsampwidth() != 2
                or reference.getframerate() != 16000
                or reference.getcomptype() != "NONE"
                or reference.getnframes() <= 0
            ):
                raise VideoExportError("Reference audio must be mono 16-bit 16 kHz PCM WAV")
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-i",
                    str(video.resolve(strict=True)),
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
                    "s16le",
                    "pipe:1",
                ],
                stdout=decoded,
                stderr=subprocess.DEVNULL,
                timeout=3600,
                check=False,
            )
            if result.returncode:
                raise VideoExportError("Could not decode source audio for comparison")
            decoded.seek(0)
            remaining = reference.getnframes()
            total = remaining
            while remaining:
                frames = min(remaining, 16000)
                chunk = reference.readframes(frames)
                if len(chunk) != frames * 2 or decoded.read(frames * 2) != chunk:
                    raise VideoExportError("Reference WAV does not match the source audio")
                remaining -= frames
            if decoded.read(1):
                raise VideoExportError("Reference WAV does not match the source audio")
            return Fraction(total, 16000)
    except VideoExportError:
        raise
    except (OSError, EOFError, wave.Error, ValueError, subprocess.TimeoutExpired) as exc:
        raise VideoExportError("Could not verify reference WAV audio") from exc


def _verify_exported_coverage(path: Path, duration_ms: int, frame_rate: Fraction) -> None:
    """Reject output whose audio or video does not cover the requested interval.

    Re-encoding can quantize video to a source frame and add AAC encoder
    framing. These checks cannot prove visual content or subjective quality.
    """
    try:
        probe_video(path)
        result = _run_tool(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-show_entries",
                "stream=codec_type,start_time,duration",
                "-of",
                "json",
                str(path),
            ],
            timeout=30,
        )
    except IngestionError as exc:
        raise VideoExportError("Exported clip is invalid or incomplete") from exc
    if result.returncode:
        raise VideoExportError("Could not verify exported clip timing")
    try:
        data = json.loads(result.stdout)
        videos = [s for s in data["streams"] if s.get("codec_type") == "video"]
        audios = [s for s in data["streams"] if s.get("codec_type") == "audio"]
        if len(videos) != 1 or len(audios) != 1:
            raise VideoExportError("Exported clip is missing audio or video")
        video_start = _float(videos[0]["start_time"])
        video_duration = _float(videos[0]["duration"])
        audio_start = _float(audios[0]["start_time"])
        audio_duration = _float(audios[0]["duration"])
        requested = duration_ms / 1000
        video_tolerance = float(1 / frame_rate) + _TIMING_TOLERANCE_SECONDS
        audio_tolerance = 0.05
        if (
            video_duration <= 0
            or audio_duration <= 0
            or abs(video_start) > video_tolerance
            or abs(audio_start) > audio_tolerance
            or abs(video_start + video_duration - requested) > video_tolerance
            or abs(audio_start + audio_duration - requested) > audio_tolerance
        ):
            raise VideoExportError("Exported clip does not cover the requested range")
    except VideoExportError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise VideoExportError("Exported clip timing is missing or invalid") from exc


def _verify_exported_decode(path: Path) -> None:
    """Reject outputs with unreadable audio or video before publication."""
    try:
        result = _run_tool(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            timeout=3600,
        )
    except IngestionError as exc:
        raise VideoExportError("Could not verify exported clip playback") from exc
    if result.returncode:
        raise VideoExportError("Exported clip cannot be decoded")


def export_video_clip(
    video: Path, audio: Path, output: Path, start_ms: int, end_ms: int
) -> ExportMetadata:
    """Export one audio-relative range after checking supported source timing.

    Media is local and must be supplied by the caller. The requested interval is
    in WAV-relative milliseconds; actual video boundaries are frame-quantized.
    The output is a new private file; source and reference are not modified.
    """
    video, audio, output = Path(video), Path(audio), Path(output)
    if not _integer(start_ms) or not _integer(end_ms) or start_ms < 0 or end_ms <= start_ms:
        raise VideoExportError("Invalid clip timing")
    if output.suffix.lower() != ".mp4" or not output.parent.is_dir():
        raise VideoExportError("Output must be a new MP4 file in an existing directory")
    if output.exists() or output.is_symlink():
        raise VideoExportError("Output already exists")

    frame_rate, audio_rate = _streams(video)
    video_end = _continuous_end(video, "v:0", frame_rate)
    audio_end = _continuous_end(video, "a:0", audio_rate)
    wav_end = _verify_wav(video, audio)
    if Fraction(end_ms, 1000) > min(video_end, audio_end, wav_end):
        raise VideoExportError("Clip ends beyond the source video or audio")

    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=".clipforge-video-", suffix=".mp4", dir=output.parent
        )
        temporary = Path(name)
        os.close(descriptor)
        duration = end_ms - start_ms
        result = _run_tool(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-ss",
                f"{start_ms / 1000:.3f}",
                "-i",
                str(video.resolve(strict=True)),
                "-t",
                f"{duration / 1000:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-map_metadata",
                "-1",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                "-y",
                str(temporary),
            ],
            timeout=3600,
        )
        if result.returncode or temporary.stat().st_size <= 0:
            raise VideoExportError("FFmpeg could not export the clip")
        _verify_exported_coverage(temporary, duration, frame_rate)
        _verify_exported_decode(temporary)
        os.link(temporary, output)
    except VideoExportError:
        raise
    except IngestionError as exc:
        raise VideoExportError("Could not run FFmpeg") from exc
    except FileExistsError as exc:
        raise VideoExportError("Output already exists") from exc
    except OSError as exc:
        raise VideoExportError("Could not create the output file") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return ExportMetadata(start_ms=start_ms, end_ms=end_ms, duration_ms=end_ms - start_ms)
