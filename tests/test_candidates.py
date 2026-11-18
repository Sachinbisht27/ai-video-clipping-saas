"""Candidate selection behavior using synthetic, local transcript words."""

from dataclasses import replace

import pytest

from clipforge.services.candidates import CandidateSelectionError, select_candidates
from clipforge.services.transcription import TranscriptWord


def word(text: str, start: int, end: int, speaker: int | None = None) -> TranscriptWord:
    return TranscriptWord(text=text, start_ms=start, end_ms=end, speaker=speaker)


def test_prefers_latest_natural_sentence_boundary_within_maximum() -> None:
    words = (
        word("First.", 0, 10_000),
        word("Second.", 10_100, 24_000),
        word("Third.", 24_100, 41_000),
        word("Fourth.", 41_100, 70_000),
    )
    candidates = select_candidates(words, 70_000)
    assert len(candidates) == 2
    assert (candidates[0].start_ms, candidates[0].end_ms) == (0, 41_000)
    assert candidates[0].end_boundary == "sentence"
    assert candidates[0].words == words[:3]
    assert (candidates[1].start_ms, candidates[1].end_ms) == (41_100, 70_000)
    assert candidates[1].words == words[3:]


def test_pause_is_a_natural_boundary() -> None:
    words = (
        word("One", 0, 24_000),
        word("Two", 25_000, 50_000),
        word("Three", 50_100, 75_000),
    )
    candidates = select_candidates(words, 75_000)
    assert candidates[0].end_ms == 24_000
    assert candidates[0].end_boundary == "pause"
    assert candidates[1].words == words[1:]


@pytest.mark.parametrize("duration", [20_000, 60_000])
def test_minimum_and_maximum_durations_are_inclusive(duration: int) -> None:
    candidates = select_candidates((word("Complete.", 0, duration),), duration)
    assert len(candidates) == 1
    assert (candidates[0].start_ms, candidates[0].end_ms) == (0, duration)


def test_word_boundaries_fall_back_when_no_natural_boundary_fits() -> None:
    words = (
        word("One", 0, 19_000),
        word("two", 19_000, 40_000),
        word("three", 40_000, 70_000),
    )
    candidates = select_candidates(words, 70_000)
    assert [(c.start_ms, c.end_ms) for c in candidates] == [(0, 40_000), (40_000, 70_000)]
    assert candidates[0].end_boundary == "word"
    assert candidates[1].start_boundary == "word"


def test_no_candidates_for_silence_or_short_recording() -> None:
    assert select_candidates((), 70_000) == ()
    assert select_candidates((word("Short.", 0, 19_999),), 19_999) == ()


def test_long_silence_does_not_join_separate_sections() -> None:
    words = (word("First.", 0, 21_000), word("Second.", 100_000, 121_000))
    candidates = select_candidates(words, 121_000)
    assert [(c.start_ms, c.end_ms) for c in candidates] == [(0, 21_000), (100_000, 121_000)]


def test_long_silence_does_not_make_two_short_parts_a_candidate() -> None:
    words = (word("One", 0, 1_000), word("Two.", 59_000, 60_000))
    assert select_candidates(words, 60_000) == ()


def test_gap_at_threshold_separates_long_enough_parts() -> None:
    words = (word("First.", 0, 21_000), word("Second.", 26_000, 47_000))
    candidates = select_candidates(words, 47_000)
    assert [(c.start_ms, c.end_ms) for c in candidates] == [(0, 21_000), (26_000, 47_000)]


def test_gap_just_below_threshold_can_be_in_candidate() -> None:
    words = (word("First", 0, 10_000), word("second.", 14_999, 24_999))
    candidates = select_candidates(words, 24_999)
    assert len(candidates) == 1
    assert candidates[0].words == words


def test_short_section_after_long_pause_does_not_hide_a_later_candidate() -> None:
    words = (word("Short.", 0, 5_000), word("Long.", 100_000, 121_000))
    candidates = select_candidates(words, 121_000)
    assert [(c.start_ms, c.end_ms) for c in candidates] == [(100_000, 121_000)]
    assert candidates[0].words == words[1:]


def test_long_indivisible_word_is_not_truncated_to_fit() -> None:
    assert select_candidates((word("long", 0, 60_001),), 60_001) == ()


def test_overlapping_speech_preserves_speakers_and_original_timing_metadata() -> None:
    first = replace(
        word("Hello", 0, 18_000, 0),
        timing_adjusted=True,
        original_end_ms=18_010,
    )
    words = (first, word("Hi", 17_000, 22_000, 1), word("again.", 22_000, 40_000, 0))
    candidates = select_candidates(words, 40_000)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.words == words
    assert (candidate.start_word_index, candidate.end_word_index) == (0, 2)
    assert [item.speaker for item in candidate.words] == [0, 1, 0]
    assert candidate.words[0].timing_adjusted
    assert candidate.words[0].original_end_ms == 18_010


def test_overlap_chain_is_not_split_at_a_crossing_word() -> None:
    words = (
        word("A", 0, 25_000, 0),
        word("B.", 20_000, 40_000, 1),
        word("C.", 40_100, 70_000, 0),
    )
    candidates = select_candidates(words, 70_000)
    assert [(c.start_word_index, c.end_word_index) for c in candidates] == [(0, 1), (2, 2)]
    assert candidates[0].end_ms == 40_000


def test_zero_duration_word_at_audio_end_is_accepted() -> None:
    words = (word("Speech.", 0, 20_000), word("end", 20_000, 20_000))
    assert select_candidates(words, 20_000)


@pytest.mark.parametrize(
    "settings",
    [
        {"min_duration_ms": 0},
        {"min_duration_ms": 60_001, "max_duration_ms": 60_000},
        {"pause_ms": -1},
        {"max_gap_ms": 0},
        {"min_duration_ms": True},
    ],
)
def test_invalid_selection_settings_are_rejected(settings: dict[str, int]) -> None:
    with pytest.raises(CandidateSelectionError):
        select_candidates((), 1, **settings)


@pytest.mark.parametrize(
    "words,duration",
    [
        ((word("bad", -1, 10),), 100),
        ((word("bad", 20, 10),), 100),
        ((word("bad", 0, 101),), 100),
        ((word("bad", 101, 101),), 100),
        ((word("one", 30, 40), word("two", 20, 25)), 100),
        ((word("bad", True, 10),), 100),
        ((word("bad", 0, 10),), -1),
        ((word("bad", 0, 10),), True),
    ],
)
def test_invalid_timestamps_are_rejected(words: tuple[TranscriptWord, ...], duration: int) -> None:
    with pytest.raises(CandidateSelectionError):
        select_candidates(words, duration)


def test_selection_is_deterministic_and_does_not_duplicate_windows() -> None:
    words = tuple(word(f"{index}.", index * 10_000, (index + 1) * 10_000) for index in range(12))
    first = select_candidates(words, 120_000)
    assert first == select_candidates(words, 120_000)
    windows = [(candidate.start_ms, candidate.end_ms) for candidate in first]
    assert len(windows) == len(set(windows))
    assert all(20_000 <= end - start <= 60_000 for start, end in windows)
