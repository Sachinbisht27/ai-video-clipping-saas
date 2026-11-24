"""Create a local preview with extra source audio after an existing ending.

Additional audio is only context for a listener. This does not detect laughter,
choose an ending, or establish whether a clip is complete.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from pathlib import Path


class AudioContextError(ValueError):
    """The source, boundaries, or preview destination are invalid."""


@dataclass(frozen=True, slots=True)
class ContextPreview:
    """Source-relative times for an additional-context preview."""

    start_ms: int
    original_end_ms: int
    preview_end_ms: int
    requested_context_ms: int
    included_context_ms: int
    truncated_at_eof: bool


_RATE = 16_000
_FRAMES_PER_MS = 16
_MAX_CONTEXT_MS = 5_000
_CHUNK_FRAMES = 16_000
_MAX_WAV_DATA_BYTES = 0xFFFFFFFF - 36


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def create_context_preview(
    audio: Path,
    start_ms: int,
    end_ms: int,
    context_ms: int,
    output: Path,
) -> ContextPreview:
    """Copy the candidate and bounded following audio to a new private WAV.

    The caller must provide the same source WAV used for the candidate times.
    A new file is created exclusively, preserving the original PCM samples;
    no fade or audio processing is applied. Audio beyond the source is omitted.
    An unverified extra-context preview can include another sentence, noise, or
    any other sound; it is never a claim that a reaction or a safe ending exists.
    """
    if (
        not _is_int(start_ms)
        or not _is_int(end_ms)
        or not _is_int(context_ms)
        or start_ms < 0
        or end_ms <= start_ms
        or not 0 <= context_ms <= _MAX_CONTEXT_MS
    ):
        raise AudioContextError("Invalid clip timing or context length")

    try:
        destination = Path(output)
        with wave.open(str(audio), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != _RATE
                or source.getcomptype() != "NONE"
            ):
                raise AudioContextError("Audio must be mono 16-bit 16 kHz PCM WAV")
            total_frames = source.getnframes()
            duration_ms = total_frames // _FRAMES_PER_MS
            if total_frames <= 0 or end_ms > duration_ms:
                raise AudioContextError("Clip timing is outside the audio")

            # Headers can advertise frames that do not exist in a truncated file.
            source.setpos(total_frames - 1)
            if len(source.readframes(1)) != 2:
                raise AudioContextError("Audio is truncated")

            preview_end_ms = min(duration_ms, end_ms + context_ms)
            preview_frames = (preview_end_ms - start_ms) * _FRAMES_PER_MS
            if preview_frames * 2 > _MAX_WAV_DATA_BYTES:
                raise AudioContextError("Preview exceeds the WAV file size limit")
            source.setpos(start_ms * _FRAMES_PER_MS)

            # O_EXCL also refuses existing symlinks, including dangling ones.
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
            try:
                with os.fdopen(descriptor, "wb") as raw:
                    descriptor = -1
                    with wave.open(raw, "wb") as preview:
                        preview.setnchannels(1)
                        preview.setsampwidth(2)
                        preview.setframerate(_RATE)
                        remaining = preview_frames
                        while remaining:
                            frames = min(remaining, _CHUNK_FRAMES)
                            data = source.readframes(frames)
                            if len(data) != frames * 2:
                                raise AudioContextError("Audio is truncated")
                            preview.writeframesraw(data)
                            remaining -= frames
            except BaseException:
                if descriptor != -1:
                    os.close(descriptor)
                if created:
                    destination.unlink(missing_ok=True)
                raise

    except AudioContextError:
        raise
    except OSError, EOFError, wave.Error, ValueError, TypeError, OverflowError:
        raise AudioContextError("Could not read the WAV or create a new preview file") from None

    included_ms = preview_end_ms - end_ms
    return ContextPreview(
        start_ms=start_ms,
        original_end_ms=end_ms,
        preview_end_ms=preview_end_ms,
        requested_context_ms=context_ms,
        included_context_ms=included_ms,
        truncated_at_eof=included_ms < context_ms,
    )
