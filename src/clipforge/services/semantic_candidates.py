"""Validate proposed clip boundaries against the source transcript and audio.

This module does not judge whether speech is meaningful or call a model. Word
indices always refer to the supplied full transcript, not to a context slice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from clipforge.services.transcription import TranscriptWord


class SemanticCandidateError(ValueError):
    """A proposed boundary, transcript or duration setting is invalid."""


@dataclass(frozen=True, slots=True)
class BoundaryProposal:
    """A proposed inclusive range of word indices in the full transcript."""

    start_word_index: int
    end_word_index: int


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    """A structurally valid proposal; its meaning still needs review."""

    start_word_index: int
    end_word_index: int
    start_ms: int
    end_ms: int
    words: tuple[TranscriptWord, ...]
    duration_status: Literal["shorter_than_target", "within_target", "longer_than_target"]


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_proposals(
    words: Sequence[TranscriptWord],
    duration_ms: int,
    proposals: Sequence[BoundaryProposal],
    *,
    target_min_ms: int = 20_000,
    target_max_ms: int = 60_000,
    hard_min_ms: int | None = None,
    hard_max_ms: int | None = None,
) -> tuple[SemanticCandidate, ...]:
    """Resolve proposed word indices to bounded audio-relative timestamps.

    Target lengths are preferences and recorded as a status. Explicit hard
    limits are enforced. Proposals may overlap each other, but no proposal
    may cut through a word outside its inclusive range. Duplicate index ranges
    and malformed inputs reject the entire response. A valid empty response is
    distinct from an invalid one. Matching WAV and transcript and source-video
    alignment remain the caller's responsibility.
    """
    if (
        not _integer(duration_ms)
        or duration_ms <= 0
        or not _integer(target_min_ms)
        or not _integer(target_max_ms)
        or target_min_ms <= 0
        or target_max_ms < target_min_ms
        or (hard_min_ms is not None and (not _integer(hard_min_ms) or hard_min_ms <= 0))
        or (hard_max_ms is not None and (not _integer(hard_max_ms) or hard_max_ms <= 0))
        or (hard_min_ms is not None and hard_max_ms is not None and hard_min_ms > hard_max_ms)
    ):
        raise SemanticCandidateError("Invalid audio duration or duration settings")

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
            raise SemanticCandidateError("Invalid transcript word or timing")
        previous_start = word.start_ms
        furthest_end = max(furthest_end, word.end_ms)
        prefix_ends.append(furthest_end)

    selected: list[SemanticCandidate] = []
    seen: set[tuple[int, int]] = set()
    for proposal in proposals:
        if (
            not isinstance(proposal, BoundaryProposal)
            or not _integer(proposal.start_word_index)
            or not _integer(proposal.end_word_index)
            or proposal.start_word_index < 0
            or proposal.end_word_index < proposal.start_word_index
            or proposal.end_word_index >= len(items)
        ):
            raise SemanticCandidateError("Invalid proposed word range")
        first, last = proposal.start_word_index, proposal.end_word_index
        if (first, last) in seen:
            raise SemanticCandidateError("Duplicate proposed word range")
        seen.add((first, last))
        start = items[first].start_ms
        # Prefix maxima preserve the longest word in overlap and nesting chains.
        if first and prefix_ends[first - 1] > start:
            raise SemanticCandidateError("Proposed start cuts through a word")
        end = prefix_ends[last]
        if last + 1 < len(items) and items[last + 1].start_ms < end:
            raise SemanticCandidateError("Proposed end cuts through a word")
        length = end - start
        if length <= 0:
            raise SemanticCandidateError("A proposed candidate must have positive duration")
        if (hard_min_ms is not None and length < hard_min_ms) or (
            hard_max_ms is not None and length > hard_max_ms
        ):
            raise SemanticCandidateError("Proposed candidate violates a hard duration limit")
        status: Literal["shorter_than_target", "within_target", "longer_than_target"]
        if length < target_min_ms:
            status = "shorter_than_target"
        elif length > target_max_ms:
            status = "longer_than_target"
        else:
            status = "within_target"
        selected.append(
            SemanticCandidate(
                start_word_index=first,
                end_word_index=last,
                start_ms=start,
                end_ms=end,
                words=items[first : last + 1],
                duration_status=status,
            )
        )
    return tuple(selected)
