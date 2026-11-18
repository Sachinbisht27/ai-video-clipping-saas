"""Choose unranked, audio-relative clip candidates from timed transcript words."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from clipforge.services.transcription import TranscriptWord


class CandidateSelectionError(ValueError):
    """The transcript timings or selection settings are invalid."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One candidate; word indices are inclusive and times are audio-relative."""

    start_ms: int
    end_ms: int
    start_word_index: int
    end_word_index: int
    words: tuple[TranscriptWord, ...]
    start_boundary: str
    end_boundary: str


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sentence_end(text: str) -> bool:
    # Punctuation often precedes a closing quote or bracket.
    return text.rstrip().rstrip("\"'”’)]}»").endswith((".", "?", "!", "।", "。", "？", "！"))


def select_candidates(
    words: Sequence[TranscriptWord],
    duration_ms: int,
    *,
    min_duration_ms: int = 20_000,
    max_duration_ms: int = 60_000,
    pause_ms: int = 800,
    max_gap_ms: int = 5_000,
) -> tuple[Candidate, ...]:
    """Return chronological, non-overlapping candidates without ranking them.

    Within each window prefer the latest safe sentence or pause boundary, then
    the latest safe word boundary. Overlapping words are never split. Short
    remainders and indivisible spans longer than the maximum are omitted.
    Speech separated by a gap of at least ``max_gap_ms`` is never combined.
    The WAV duration must describe the same audio as the supplied transcript.
    """
    if (
        not _integer(duration_ms)
        or duration_ms <= 0
        or not _integer(min_duration_ms)
        or not _integer(max_duration_ms)
        or min_duration_ms <= 0
        or max_duration_ms < min_duration_ms
        or not _integer(pause_ms)
        or pause_ms < 0
        or not _integer(max_gap_ms)
        or max_gap_ms <= 0
    ):
        raise CandidateSelectionError("Invalid audio duration or candidate selection settings")

    items = tuple(words)
    prefix_ends: list[int] = []
    previous_start = -1
    furthest_end = 0
    for word in items:
        if (
            not isinstance(word, TranscriptWord)
            or not isinstance(word.text, str)
            or not word.text.strip()
            or not _integer(word.start_ms)
            or not _integer(word.end_ms)
            or word.start_ms < 0
            or word.end_ms < word.start_ms
            or word.end_ms > duration_ms
            or word.start_ms < previous_start
            or (word.speaker is not None and (not _integer(word.speaker) or word.speaker < 0))
        ):
            raise CandidateSelectionError("Invalid transcript word or timing")
        previous_start = word.start_ms
        furthest_end = max(furthest_end, word.end_ms)
        prefix_ends.append(furthest_end)

    if not items:
        return ()

    # A cut is safe only when every earlier word has ended before the next
    # starts. Using prefix maxima also handles overlapping and nested words.
    safe_cuts: list[int] = []
    natural_cuts: list[int] = []
    gap_cuts: list[int] = []
    reasons: dict[int, str] = {}
    for index, word in enumerate(items):
        is_last = index == len(items) - 1
        if not is_last and prefix_ends[index] > items[index + 1].start_ms:
            continue
        safe_cuts.append(index)
        gap = None if is_last else items[index + 1].start_ms - prefix_ends[index]
        if gap is not None and gap >= max_gap_ms:
            gap_cuts.append(index)
        if is_last:
            reason = "end_of_transcript"
        elif _sentence_end(word.text):
            reason = "sentence"
        elif gap is not None and (gap >= pause_ms or gap >= max_gap_ms):
            reason = "pause"
        else:
            reason = "word"
        reasons[index] = reason
        if reason != "word":
            natural_cuts.append(index)

    selected: list[Candidate] = []
    first = 0
    while first < len(items):
        start = items[first].start_ms
        # Prefix ends and safe-cut indices are both sorted, so even long
        # recordings and dense overlapping speech avoid quadratic rescans.
        last_allowed = bisect_right(prefix_ends, start + max_duration_ms) - 1
        next_gap = bisect_left(gap_cuts, first)
        if next_gap < len(gap_cuts):
            last_allowed = min(last_allowed, gap_cuts[next_gap])
        first_long_enough = max(first, bisect_left(prefix_ends, start + min_duration_ms))
        natural_pos = bisect_right(natural_cuts, last_allowed) - 1
        safe_pos = bisect_right(safe_cuts, last_allowed) - 1
        if natural_pos >= 0 and natural_cuts[natural_pos] >= first_long_enough:
            last = natural_cuts[natural_pos]
        elif safe_pos >= 0 and safe_cuts[safe_pos] >= first_long_enough:
            last = safe_cuts[safe_pos]
        else:
            # No eligible end: skip only the first indivisible run and try
            # again, allowing later speech after a long pause to qualify.
            first_safe = bisect_left(safe_cuts, first)
            first = safe_cuts[first_safe] + 1
            continue
        selected.append(
            Candidate(
                start_ms=start,
                end_ms=prefix_ends[last],
                start_word_index=first,
                end_word_index=last,
                words=items[first : last + 1],
                start_boundary="start_of_transcript" if first == 0 else reasons[first - 1],
                end_boundary=reasons[last],
            )
        )
        first = last + 1
    return tuple(selected)
