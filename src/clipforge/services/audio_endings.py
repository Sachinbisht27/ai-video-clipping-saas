"""Find low-energy ending alternatives for human review.

These are waveform observations, not speech detection or an assessment of
whether a sentence is complete. No boundary is selected or changed here.
"""

from __future__ import annotations

import math
import struct
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class AudioEndingError(ValueError):
    """Audio or boundary input cannot be inspected safely."""


@dataclass(frozen=True, slots=True)
class EndingSuggestion:
    """A low-energy position, not a verified safe cut."""

    end_ms: int
    gap_start_ms: int
    gap_end_ms: int
    # None means transcript timing was not supplied for this check.
    overlaps_transcript_word: bool | None


@dataclass(frozen=True, slots=True)
class EndingReview:
    """The original boundary and optional, unverified alternatives."""

    start_ms: int
    end_ms: int
    analysis_start_ms: int
    analysis_end_ms: int
    status: Literal["suggestions", "uncertain"]
    reason: Literal["review_required", "no_clear_gap", "quiet_audio"]
    suggestions: tuple[EndingSuggestion, ...]
    suggestions_truncated: bool


_RATE = 16_000
_FRAME_MS = 20
_FRAME_SAMPLES = _RATE * _FRAME_MS // 1000
_LOOKBACK_MS = 2_000
_LOOKAHEAD_MS = 250
_MAX_SUGGESTIONS = 8
_MIN_ACTIVE_RMS = 128
_RELATIVE_GAP_THRESHOLD = 0.12


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ranges(
    protected_ranges: Sequence[tuple[int, int]] | None,
    duration_ms: int,
) -> tuple[tuple[int, int], ...] | None:
    if protected_ranges is None:
        return None
    ranges = tuple(protected_ranges)
    for item in ranges:
        if (
            not isinstance(item, (tuple, list))
            or len(item) != 2
            or not _is_int(item[0])
            or not _is_int(item[1])
            or item[0] < 0
            or item[1] < item[0]
            or item[1] > duration_ms
        ):
            raise AudioEndingError("Invalid transcript timing range")
    return tuple((item[0], item[1]) for item in ranges)


def propose_audio_endings(
    audio: Path,
    start_ms: int,
    end_ms: int,
    *,
    protected_ranges: Sequence[tuple[int, int]] | None = None,
) -> EndingReview:
    """Inspect up to 2.25 seconds around an existing candidate's end.

    Input is mono, 16-bit, 16 kHz uncompressed PCM WAV, as produced by local
    ingestion. Report up to eight recent low-energy gaps with activity on both
    sides, ordered chronologically. Transcript timing ranges only add warnings;
    no range can prove that cutting a particular sample preserves a word. Every
    suggestion and the original boundary require human listening review.
    """
    if not _is_int(start_ms) or not _is_int(end_ms) or start_ms < 0 or end_ms <= start_ms:
        raise AudioEndingError("Invalid candidate timing")

    try:
        with wave.open(str(audio), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != _RATE
                or source.getcomptype() != "NONE"
            ):
                raise AudioEndingError("Audio must be mono 16-bit 16 kHz PCM WAV")
            total_frames = source.getnframes()
            duration_ms = total_frames * 1000 // _RATE
            if total_frames <= 0 or end_ms > duration_ms:
                raise AudioEndingError("Candidate timing is outside the audio")
            # A WAV header can claim more frames than the file actually holds.
            source.setpos(total_frames - 1)
            if len(source.readframes(1)) != 2:
                raise AudioEndingError("Audio is truncated")
            checked_ranges = _ranges(protected_ranges, duration_ms)
            analysis_start_ms = max(start_ms, end_ms - _LOOKBACK_MS)
            analysis_end_ms = min(duration_ms, end_ms + _LOOKAHEAD_MS)
            first_frame = analysis_start_ms * _RATE // 1000
            frame_count = (analysis_end_ms - analysis_start_ms) * _RATE // 1000
            source.setpos(first_frame)
            raw = source.readframes(frame_count)
            if len(raw) != frame_count * 2:
                raise AudioEndingError("Audio is truncated")
    except AudioEndingError:
        raise
    except OSError, EOFError, wave.Error, ValueError, TypeError:
        raise AudioEndingError("Could not read PCM WAV audio") from None

    samples = tuple(value[0] for value in struct.iter_unpack("<h", raw))
    levels = tuple(
        math.sqrt(sum(value * value for value in samples[i : i + _FRAME_SAMPLES]) / _FRAME_SAMPLES)
        for i in range(0, len(samples) - _FRAME_SAMPLES + 1, _FRAME_SAMPLES)
    )
    peak = max(levels, default=0.0)
    if peak < _MIN_ACTIVE_RMS:
        return EndingReview(
            start_ms,
            end_ms,
            analysis_start_ms,
            analysis_end_ms,
            "uncertain",
            "quiet_audio",
            (),
            False,
        )

    threshold = peak * _RELATIVE_GAP_THRESHOLD
    options: list[EndingSuggestion] = []
    low_start: int | None = None
    # The final sentinel flushes the last run; runs at either edge are omitted.
    for index in range(len(levels) + 1):
        quiet = index < len(levels) and levels[index] <= threshold
        if quiet and low_start is None:
            low_start = index
        elif not quiet and low_start is not None:
            if index - low_start >= 2 and low_start > 0 and index < len(levels):
                gap_start = analysis_start_ms + low_start * _FRAME_MS
                gap_end = analysis_start_ms + index * _FRAME_MS
                # Do not silently extend the original candidate.
                latest = min(gap_end, end_ms - 1)
                if gap_start < latest and gap_start > start_ms:
                    candidate_end = (gap_start + latest) // 2
                    overlap = (
                        None
                        if checked_ranges is None
                        else any(a < candidate_end < b for a, b in checked_ranges)
                    )
                    options.append(EndingSuggestion(candidate_end, gap_start, gap_end, overlap))
            low_start = None

    truncated = len(options) > _MAX_SUGGESTIONS
    suggestions = tuple(options[-_MAX_SUGGESTIONS:])
    if suggestions:
        return EndingReview(
            start_ms,
            end_ms,
            analysis_start_ms,
            analysis_end_ms,
            "suggestions",
            "review_required",
            suggestions,
            truncated,
        )
    return EndingReview(
        start_ms, end_ms, analysis_start_ms, analysis_end_ms, "uncertain", "no_clear_gap", (), False
    )
