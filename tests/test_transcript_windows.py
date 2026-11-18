"""Overlapping transcript windows preserve source coverage without provider calls."""

import pytest

from clipforge.services.transcript_windows import TranscriptWindowError, make_windows
from clipforge.services.transcription import TranscriptWord


def _words(count: int, *, spacing_ms: int = 1000) -> tuple[TranscriptWord, ...]:
    return tuple(
        TranscriptWord(f"word-{index}", index * spacing_ms, index * spacing_ms + 500)
        for index in range(count)
    )


def _assert_coverage(windows: object, count: int) -> None:
    assert isinstance(windows, tuple)
    if count == 0:
        assert windows == ()
        return
    assert windows[0].start_word_index == 0
    assert windows[-1].end_word_index == count
    assert all(0 <= w.start_word_index < w.end_word_index <= count for w in windows)
    assert all(
        a.start_word_index < b.start_word_index <= a.end_word_index
        for a, b in zip(windows, windows[1:], strict=False)
    )


def test_empty_and_single_word() -> None:
    assert make_windows(()) == ()
    windows = make_windows((TranscriptWord("hello", 0, 500),))
    _assert_coverage(windows, 1)
    assert (windows[0].start_ms, windows[0].end_ms) == (0, 500)


def test_windows_overlap_chronologically_and_map_to_global_indices() -> None:
    words = _words(10)
    windows = make_windows(words, window_ms=4000, overlap_ms=2000, max_words=20)
    _assert_coverage(windows, len(words))
    assert len(windows) > 1
    assert all(
        a.end_word_index > b.start_word_index for a, b in zip(windows, windows[1:], strict=False)
    )
    for w in windows:
        assert w.start_ms == words[w.start_word_index].start_ms
        assert w.end_ms == max(x.end_ms for x in words[w.start_word_index : w.end_word_index])
    assert make_windows(words, window_ms=4000, overlap_ms=2000, max_words=20) == windows


def test_zero_overlap_still_covers_all_words() -> None:
    words = _words(15)
    windows = make_windows(words, window_ms=3000, overlap_ms=0, max_words=20)
    _assert_coverage(windows, len(words))


def test_word_limit_forces_progress_and_reports_incomplete_long_context() -> None:
    words = _words(100, spacing_ms=100)
    windows = make_windows(words, window_ms=240_000, overlap_ms=180_000, max_words=20)
    _assert_coverage(windows, len(words))
    assert all(w.end_word_index - w.start_word_index <= 20 for w in windows)
    assert any(w.word_limit_reached for w in windows)
    assert all(w.end_ms - w.start_ms < 180_000 for w in windows)
    assert len(windows) <= len(words)


def test_long_window_can_contain_three_minute_thought() -> None:
    words = _words(270, spacing_ms=1000)
    windows = make_windows(words, window_ms=240_000, overlap_ms=180_000, max_words=300)
    _assert_coverage(windows, len(words))
    assert windows[0].end_ms - windows[0].start_ms >= 180_000
    assert not windows[0].word_limit_reached


def test_sparse_speech_and_long_silence_do_not_skip_words() -> None:
    words = (
        TranscriptWord("a", 0, 500),
        TranscriptWord("b", 1_000_000, 1_000_500),
        TranscriptWord("c", 1_000_600, 1_001_000),
        TranscriptWord("d", 2_000_000, 2_000_100),
    )
    windows = make_windows(words, window_ms=3000, overlap_ms=1000, max_words=10)
    _assert_coverage(windows, len(words))
    assert {i for w in windows for i in range(w.start_word_index, w.end_word_index)} == set(
        range(4)
    )


def test_simultaneous_words_and_zero_duration_make_strict_progress() -> None:
    words = tuple(TranscriptWord(f"w{i}", 1000, 1000, speaker=i % 2) for i in range(101))
    windows = make_windows(words, window_ms=3000, overlap_ms=1000, max_words=10)
    _assert_coverage(windows, len(words))
    assert all(w.end_word_index - w.start_word_index <= 10 for w in windows)
    assert len(windows) <= len(words)


def test_nested_overlapping_speakers_preserve_furthest_end() -> None:
    words = (
        TranscriptWord("long", 0, 5000, speaker=0),
        TranscriptWord("short", 1000, 1500, speaker=1),
        TranscriptWord("next", 2000, 3000, speaker=1),
        TranscriptWord("last", 5000, 6000, speaker=0),
    )
    windows = make_windows(words, window_ms=4000, overlap_ms=1000, max_words=10)
    _assert_coverage(windows, len(words))
    assert windows[0].end_ms == 5000


def test_large_dense_input_does_not_create_quadratic_window_count() -> None:
    words = _words(10_000, spacing_ms=100)
    windows = make_windows(words, window_ms=240_000, overlap_ms=180_000, max_words=100)
    _assert_coverage(windows, len(words))
    assert len(windows) <= len(words)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_ms": 0},
        {"window_ms": -1},
        {"window_ms": True},
        {"window_ms": 1000, "overlap_ms": 1000},
        {"window_ms": 1000, "overlap_ms": -1},
        {"overlap_ms": False},
        {"max_words": 0},
        {"max_words": True},
    ],
)
def test_invalid_settings_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(TranscriptWindowError):
        make_windows(_words(2), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "words",
    [
        (TranscriptWord("negative", -1, 0),),
        (TranscriptWord("backward", 20, 10),),
        (TranscriptWord("bool", True, 10),),
        (TranscriptWord("x", 20, 25), TranscriptWord("y", 10, 12)),
    ],
)
def test_invalid_word_times_are_rejected(words: tuple[TranscriptWord, ...]) -> None:
    with pytest.raises(TranscriptWindowError):
        make_windows(words)
