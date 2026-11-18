"""Validate proposed transcript boundaries without making model calls."""

import pytest

from clipforge.services.semantic_candidates import (
    BoundaryProposal,
    SemanticCandidateError,
    validate_proposals,
)
from clipforge.services.transcription import TranscriptWord


@pytest.fixture
def words() -> tuple[TranscriptWord, ...]:
    return (
        TranscriptWord("Hello.", 0, 1000, speaker=0),
        TranscriptWord("Why?", 1500, 3000, speaker=1, confidence=0.91),
        TranscriptWord(
            "Because", 3000, 25000, speaker=0, timing_adjusted=True, original_end_ms=26000
        ),
        TranscriptWord("done.", 26000, 62000, speaker=1),
    )


def test_empty_proposals_return_no_candidates(words: tuple[TranscriptWord, ...]) -> None:
    assert validate_proposals(words, 65000, ()) == ()


def test_preserves_exact_words_speakers_and_timing_metadata(
    words: tuple[TranscriptWord, ...],
) -> None:
    result = validate_proposals(words, 65000, [BoundaryProposal(1, 2)])
    assert len(result) == 1
    candidate = result[0]
    assert (candidate.start_word_index, candidate.end_word_index) == (1, 2)
    assert (candidate.start_ms, candidate.end_ms) == (1500, 25000)
    assert candidate.words == words[1:3]
    assert [w.speaker for w in candidate.words] == [1, 0]
    assert candidate.words[1].timing_adjusted
    assert candidate.words[1].original_end_ms == 26000
    assert candidate.duration_status == "within_target"


def test_overlapping_distinct_proposals_are_allowed(words: tuple[TranscriptWord, ...]) -> None:
    result = validate_proposals(words, 65000, [BoundaryProposal(0, 2), BoundaryProposal(1, 3)])
    assert len(result) == 2
    assert (result[0].start_word_index, result[1].start_word_index) == (0, 1)


def test_exact_duplicates_are_rejected(words: tuple[TranscriptWord, ...]) -> None:
    with pytest.raises(SemanticCandidateError):
        validate_proposals(words, 65000, [BoundaryProposal(0, 2), BoundaryProposal(0, 2)])


@pytest.mark.parametrize(
    "first,last", [(-1, 2), (0, 4), (3, 2), (True, 2), (0, False), (0.0, 2), (0, "2")]
)
def test_invalid_indices_are_rejected(
    words: tuple[TranscriptWord, ...], first: object, last: object
) -> None:
    with pytest.raises(SemanticCandidateError):
        validate_proposals(words, 65000, [BoundaryProposal(first, last)])  # type: ignore[arg-type]


def test_rejects_word_timing_outside_audio(words: tuple[TranscriptWord, ...]) -> None:
    with pytest.raises(SemanticCandidateError):
        validate_proposals(words, 61000, [BoundaryProposal(0, 2)])


@pytest.mark.parametrize(
    "proposal,status",
    [
        (BoundaryProposal(0, 1), "shorter_than_target"),
        (BoundaryProposal(0, 2), "within_target"),
        (BoundaryProposal(0, 3), "longer_than_target"),
    ],
)
def test_target_is_preference_not_a_hard_limit(
    words: tuple[TranscriptWord, ...], proposal: BoundaryProposal, status: str
) -> None:
    assert validate_proposals(words, 65000, [proposal])[0].duration_status == status


@pytest.mark.parametrize(
    "proposal,kwargs",
    [
        (BoundaryProposal(0, 1), {"hard_min_ms": 20_000}),
        (BoundaryProposal(0, 3), {"hard_max_ms": 60_000}),
    ],
)
def test_explicit_hard_limits_are_enforced(
    words: tuple[TranscriptWord, ...], proposal: BoundaryProposal, kwargs: dict[str, int]
) -> None:
    with pytest.raises(SemanticCandidateError):
        validate_proposals(words, 65000, [proposal], **kwargs)


def test_overlapping_speech_must_not_be_cut_at_either_boundary() -> None:
    words = (
        TranscriptWord("first", 0, 2000, speaker=0),
        TranscriptWord("overlap", 1000, 3000, speaker=1),
        TranscriptWord("after", 3000, 4000, speaker=0),
    )
    for proposal in (BoundaryProposal(1, 2), BoundaryProposal(0, 0)):
        with pytest.raises(SemanticCandidateError):
            validate_proposals(words, 5000, [proposal])
    candidate = validate_proposals(words, 5000, [BoundaryProposal(0, 1)])[0]
    assert (candidate.start_ms, candidate.end_ms) == (0, 3000)
    assert [w.speaker for w in candidate.words] == [0, 1]


def test_nested_overlapping_words_use_maximum_end() -> None:
    words = (
        TranscriptWord("long", 0, 4000, speaker=0),
        TranscriptWord("short", 1000, 2000, speaker=1),
        TranscriptWord("next", 4000, 5000, speaker=0),
    )
    result = validate_proposals(words, 6000, [BoundaryProposal(0, 1)])
    assert result[0].end_ms == 4000


@pytest.mark.parametrize("value", [None, {}, "not a proposal", (0, 1)])
def test_malformed_proposal_is_rejected(words: tuple[TranscriptWord, ...], value: object) -> None:
    with pytest.raises(SemanticCandidateError):
        validate_proposals(words, 65000, [value])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_min_ms": 0},
        {"target_min_ms": 60001},
        {"target_max_ms": True},
        {"hard_min_ms": -1},
        {"hard_min_ms": 100, "hard_max_ms": 99},
    ],
)
def test_invalid_settings_are_rejected(
    words: tuple[TranscriptWord, ...], kwargs: dict[str, int]
) -> None:
    with pytest.raises(SemanticCandidateError):
        validate_proposals(words, 65000, [BoundaryProposal(0, 2)], **kwargs)


@pytest.mark.parametrize("duration", [20000, 60000])
def test_target_limits_are_inclusive(duration: int) -> None:
    words = (TranscriptWord("thought", 0, duration),)
    result = validate_proposals(words, duration, [BoundaryProposal(0, 0)])
    assert result[0].duration_status == "within_target"


def test_zero_duration_candidate_is_rejected() -> None:
    words = (TranscriptWord("word", 500, 500),)
    with pytest.raises(SemanticCandidateError):
        validate_proposals(words, 1000, [BoundaryProposal(0, 0)])
