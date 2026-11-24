"""Synthetic signal and input-safety checks for audio ending review hints.

These signals check timing and validation, not speech intelligibility.
"""

import struct
import wave
from pathlib import Path

import pytest

from clipforge.services.audio_endings import AudioEndingError, propose_audio_endings

RATE = 16_000


def make_wav(path: Path, segments: list[tuple[int, int]], *, rate: int = RATE) -> None:
    """Create alternating-amplitude PCM segments with durations in milliseconds."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for milliseconds, amplitude in segments:
            frames = milliseconds * rate // 1000
            sample_pair = struct.pack("<hh", amplitude, -amplitude)
            wav.writeframesraw(sample_pair * (frames // 2))
            if frames % 2:
                wav.writeframesraw(struct.pack("<h", amplitude))


@pytest.fixture
def clear_gap(tmp_path: Path) -> Path:
    audio = tmp_path / "source.wav"
    make_wav(audio, [(2000, 10_000), (100, 0), (1900, 10_000)])
    return audio


def test_gap_options_remain_within_original_clip_and_are_review_only(clear_gap: Path) -> None:
    original = clear_gap.read_bytes()
    result = propose_audio_endings(clear_gap, 0, 3000)
    assert result.start_ms == 0
    assert result.end_ms == 3000
    assert result.status == "suggestions"
    assert result.reason == "review_required"
    assert 0 < len(result.suggestions) <= 8
    assert tuple(item.end_ms for item in result.suggestions) == tuple(
        sorted(item.end_ms for item in result.suggestions)
    )
    assert any(2000 <= item.end_ms <= 2100 for item in result.suggestions)
    for item in result.suggestions:
        assert 0 < item.end_ms < result.end_ms
        assert result.analysis_start_ms <= item.gap_start_ms <= item.end_ms
        assert item.end_ms <= item.gap_end_ms <= result.analysis_end_ms
        assert item.overlaps_transcript_word is None
    assert clear_gap.read_bytes() == original


def test_checked_transcript_overlap_is_distinct_from_not_checked(clear_gap: Path) -> None:
    overlapping = propose_audio_endings(clear_gap, 0, 3000, protected_ranges=[(1900, 2200)])
    checked_empty = propose_audio_endings(clear_gap, 0, 3000, protected_ranges=[])
    assert overlapping.suggestions
    assert all(item.overlaps_transcript_word is True for item in overlapping.suggestions)
    assert checked_empty.suggestions
    assert all(item.overlaps_transcript_word is False for item in checked_empty.suggestions)


def test_protected_interval_boundaries_are_not_inside_word(clear_gap: Path) -> None:
    baseline = propose_audio_endings(clear_gap, 0, 3000)
    boundary = baseline.suggestions[0].end_ms
    left = propose_audio_endings(clear_gap, 0, 3000, protected_ranges=[(1000, boundary)])
    right = propose_audio_endings(clear_gap, 0, 3000, protected_ranges=[(boundary, 2500)])
    assert left.suggestions[0].overlaps_transcript_word is False
    assert right.suggestions[0].overlaps_transcript_word is False


@pytest.mark.parametrize("amplitude", [0, 10_000])
def test_silent_and_continuous_audio_abstain(tmp_path: Path, amplitude: int) -> None:
    audio = tmp_path / "source.wav"
    make_wav(audio, [(4000, amplitude)])
    result = propose_audio_endings(audio, 0, 3000)
    assert result.status == "uncertain"
    assert result.reason == ("quiet_audio" if amplitude == 0 else "no_clear_gap")
    assert not result.suggestions


def test_persistent_background_signal_does_not_become_a_silence_gap(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    make_wav(audio, [(2000, 10_000), (100, 4000), (1900, 10_000)])
    result = propose_audio_endings(audio, 0, 3000)
    assert result.status == "uncertain"
    assert result.reason == "no_clear_gap"
    assert not result.suggestions


def test_brief_loud_transients_do_not_make_continuous_quiet_audio_a_gap(tmp_path: Path) -> None:
    audio = tmp_path / "transients.wav"
    make_wav(audio, [(1200, 500), (20, 20_000), (600, 500), (20, 20_000), (2160, 500)])
    result = propose_audio_endings(audio, 0, 3000)
    assert result.status == "uncertain"
    assert not result.suggestions


def test_finds_gap_between_short_active_sections(tmp_path: Path) -> None:
    audio = tmp_path / "sparse.wav"
    make_wav(audio, [(1200, 0), (100, 10_000), (600, 0), (100, 10_000), (2000, 0)])
    result = propose_audio_endings(audio, 0, 3000)
    assert result.status == "suggestions"
    assert any(1300 <= item.end_ms <= 1900 for item in result.suggestions)


def test_analysis_window_is_clipped_at_both_audio_ends(clear_gap: Path) -> None:
    near_start = propose_audio_endings(clear_gap, 0, 200)
    near_end = propose_audio_endings(clear_gap, 2000, 4000)
    assert near_start.analysis_start_ms == 0
    assert near_end.analysis_end_ms <= 4000
    assert all(item.end_ms < near_start.end_ms for item in near_start.suggestions)
    assert all(near_end.start_ms < item.end_ms < near_end.end_ms for item in near_end.suggestions)


def test_analysis_is_bounded_for_long_recordings(tmp_path: Path) -> None:
    audio = tmp_path / "long.wav"
    make_wav(audio, [(10_000, 10_000)])
    result = propose_audio_endings(audio, 0, 5000)
    assert result.analysis_start_ms == 3000
    assert result.analysis_end_ms == 5250
    assert result.status == "uncertain"


def test_candidates_are_limited_and_truncation_is_reported(tmp_path: Path) -> None:
    audio = tmp_path / "many-gaps.wav"
    segments = [(100, 10_000)]
    for _ in range(12):
        segments.extend([(60, 0), (100, 10_000)])
    make_wav(audio, segments)
    result = propose_audio_endings(audio, 0, 2000)
    assert result.status == "suggestions"
    assert len(result.suggestions) == 8
    assert result.suggestions_truncated is True
    assert result.suggestions[0].end_ms > 200


def test_low_energy_runs_touching_analysis_edges_are_not_proposed(tmp_path: Path) -> None:
    audio = tmp_path / "edge-gaps.wav"
    make_wav(audio, [(200, 0), (2800, 10_000), (250, 0)])
    result = propose_audio_endings(audio, 0, 3000)
    assert result.status == "uncertain"
    assert result.reason == "no_clear_gap"
    assert not result.suggestions


def test_short_gap_under_40ms_does_not_produce_suggestion(tmp_path: Path) -> None:
    audio = tmp_path / "short-gap.wav"
    make_wav(audio, [(2000, 10_000), (20, 0), (1980, 10_000)])
    result = propose_audio_endings(audio, 0, 3000)
    assert result.status == "uncertain"
    assert not result.suggestions


@pytest.mark.parametrize(
    "start,end", [(-1, 3000), (100, 100), (3000, 2000), (0, 4001), (True, 100)]
)
def test_rejects_out_of_range_or_empty_clip(clear_gap: Path, start: int, end: int) -> None:
    with pytest.raises(AudioEndingError):
        propose_audio_endings(clear_gap, start, end)


@pytest.mark.parametrize("ranges", [[(-1, 2)], [(100, 99)], [(0, 4001)], [(True, 100)]])
def test_rejects_invalid_transcript_timing_ranges(
    clear_gap: Path, ranges: list[tuple[int, int]]
) -> None:
    with pytest.raises(AudioEndingError):
        propose_audio_endings(clear_gap, 0, 3000, protected_ranges=ranges)


def test_rejects_invalid_and_truncated_wav(tmp_path: Path, clear_gap: Path) -> None:
    invalid = tmp_path / "invalid.wav"
    invalid.write_bytes(b"not a wav")
    truncated = tmp_path / "truncated.wav"
    truncated.write_bytes(clear_gap.read_bytes()[:-20])
    for audio in (invalid, truncated, tmp_path / "missing.wav"):
        with pytest.raises(AudioEndingError):
            propose_audio_endings(audio, 0, 1000)


def test_rejects_empty_wav(tmp_path: Path) -> None:
    audio = tmp_path / "empty.wav"
    make_wav(audio, [])
    with pytest.raises(AudioEndingError):
        propose_audio_endings(audio, 0, 1)


@pytest.mark.parametrize("channels,width,rate", [(2, 2, RATE), (1, 1, RATE), (1, 2, 8000)])
def test_rejects_unsupported_pcm_format(
    tmp_path: Path, channels: int, width: int, rate: int
) -> None:
    audio = tmp_path / "unsupported.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(bytes(rate * channels * width))
    with pytest.raises(AudioEndingError):
        propose_audio_endings(audio, 0, 500)
