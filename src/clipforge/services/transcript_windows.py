"""Plan overlapping transcript windows without contacting a provider."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

from clipforge.services.transcription import TranscriptWord


class TranscriptWindowError(ValueError):
    """The transcript or window settings are invalid."""


@dataclass(frozen=True, slots=True)
class TranscriptWindow:
    """A contiguous slice, with an exclusive end and audio-relative times.

    ``word_limit_reached`` means the word cap excluded at least one word that
    began within the requested time span. The flag says nothing about whether
    a semantically complete clip is available in the window.
    """

    start_word_index: int
    end_word_index: int
    start_ms: int
    end_ms: int
    word_limit_reached: bool


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def make_windows(
    words: Sequence[TranscriptWord],
    *,
    window_ms: int = 240_000,
    overlap_ms: int = 180_000,
    max_words: int = 300,
) -> tuple[TranscriptWindow, ...]:
    """Cover all words with bounded slices and time or count-based overlap.

    The time overlap is a target; the word cap, sparse speech, and duplicate
    timestamps can reduce it. When the word cap is hit, advance by half the
    window's words to ensure useful progress. No duration or content quality
    is guaranteed. Original indices and audio-relative timestamps are retained.
    """
    if (
        not _integer(window_ms)
        or not _integer(overlap_ms)
        or not _integer(max_words)
        or window_ms <= 0
        or overlap_ms < 0
        or overlap_ms >= window_ms
        or max_words <= 0
    ):
        raise TranscriptWindowError("Invalid window settings")

    items = tuple(words)
    starts: list[int] = []
    previous_start = -1
    for word in items:
        if (
            not isinstance(word, TranscriptWord)
            or not isinstance(word.text, str)
            or not word.text.strip()
            or not _integer(word.start_ms)
            or not _integer(word.end_ms)
            or word.start_ms < 0
            or word.start_ms < previous_start
            or word.end_ms < word.start_ms
        ):
            raise TranscriptWindowError("Invalid transcript word or timing")
        starts.append(word.start_ms)
        previous_start = word.start_ms

    windows: list[TranscriptWindow] = []
    first = 0
    while first < len(items):
        start_time = starts[first]
        by_time = bisect_left(starts, start_time + window_ms, lo=first + 1)
        end = min(by_time, first + max_words)
        capped = end < by_time
        windows.append(
            TranscriptWindow(
                start_word_index=first,
                end_word_index=end,
                start_ms=start_time,
                end_ms=max(word.end_ms for word in items[first:end]),
                word_limit_reached=capped,
            )
        )
        if end == len(items):
            break
        if capped:
            first += max(1, (end - first) // 2)
        else:
            time_advance = start_time + window_ms - overlap_ms
            first = min(end, bisect_left(starts, time_advance, lo=first + 1))
    return tuple(windows)
