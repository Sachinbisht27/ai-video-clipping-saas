"""Window orchestration tests use synthetic transcripts and fake providers."""

from collections.abc import Sequence

import pytest

from clipforge.services.groq_boundaries import GroqBoundaryError
from clipforge.services.openai_boundaries import BoundaryResult, ModelUsage
from clipforge.services.semantic_candidates import BoundaryProposal
from clipforge.services.semantic_windowed import analyze_windows
from clipforge.services.transcription import TranscriptWord


def _words(count: int) -> tuple[TranscriptWord, ...]:
    return tuple(
        TranscriptWord(f"word-{i}", i * 1000, (i + 1) * 1000, speaker=i % 2) for i in range(count)
    )


class FakeProposer:
    def __init__(self, responses: dict[int, tuple[BoundaryProposal, ...] | str]) -> None:
        self.responses = responses
        self.calls: list[tuple[int, int, str, str]] = []

    def propose(
        self,
        words: Sequence[TranscriptWord],
        *,
        first_word_index: int = 0,
        duration_mode: str = "both",
        input_format: str = "segments",
    ) -> BoundaryResult:
        self.calls.append((first_word_index, len(words), duration_mode, input_format))
        result = self.responses.get(first_word_index, ())
        if isinstance(result, str):
            raise GroqBoundaryError(result)
        return BoundaryResult(result, ModelUsage(50, 20, 10))


def test_offsets_map_to_full_transcript_and_modes_forwarded() -> None:
    words = _words(10)
    proposer = FakeProposer({3: (BoundaryProposal(4, 6),)})
    result = analyze_windows(
        words,
        10_000,
        proposer,
        duration_mode="long",
        start_window=1,
        max_windows=1,
        window_ms=6000,
        overlap_ms=3000,
        max_words=100,
    )
    assert proposer.calls == [(3, 6, "long", "segments")]
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert (candidate.start_word_index, candidate.end_word_index) == (4, 6)
    assert (candidate.start_ms, candidate.end_ms) == (4000, 7000)
    assert candidate.words == words[4:7]
    assert not result.processing_complete
    assert result.next_window_index == 2


def test_exact_duplicates_removed_but_near_and_distinct_overlaps_preserved() -> None:
    words = _words(110)
    proposer = FakeProposer(
        {
            0: (BoundaryProposal(30, 50),),
            30: (
                BoundaryProposal(30, 50),
                BoundaryProposal(30, 51),
                BoundaryProposal(40, 70),
            ),
        }
    )
    result = analyze_windows(
        words, 110_000, proposer, window_ms=60_000, overlap_ms=30_000, max_words=100
    )
    assert [(c.start_word_index, c.end_word_index) for c in result.candidates] == [
        (30, 50),
        (30, 51),
        (40, 70),
    ]
    assert result.windows[1].duplicates == 1
    assert result.windows[1].accepted == 2
    assert result.processing_complete


def test_near_duplicate_short_and_long_ranges_both_retained() -> None:
    words = _words(106)
    proposer = FakeProposer({0: (BoundaryProposal(5, 64),), 5: (BoundaryProposal(5, 65),)})
    result = analyze_windows(
        words,
        106_000,
        proposer,
        window_ms=100_000,
        overlap_ms=95_000,
        max_words=200,
        max_windows=2,
    )
    assert [(c.start_ms, c.end_ms) for c in result.candidates] == [(5000, 65_000), (5000, 66_000)]
    assert result.windows[1].duplicates == 0
    assert not result.processing_complete


def test_unsafe_speaker_boundary_rejected_while_valid_proposal_retained() -> None:
    words = (
        TranscriptWord("first", 0, 2000, speaker=0),
        TranscriptWord("overlap", 1000, 3000, speaker=1),
        TranscriptWord("later", 3000, 4000, speaker=0),
    )
    proposer = FakeProposer({0: (BoundaryProposal(0, 0), BoundaryProposal(0, 1))})
    result = analyze_windows(words, 4000, proposer, window_ms=5000, overlap_ms=1000, max_words=10)
    assert [(c.start_word_index, c.end_word_index) for c in result.candidates] == [(0, 1)]
    assert result.windows[0].rejected == 1
    assert result.windows[0].accepted == 1
    assert not result.processing_complete


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("Model provider rejected the request (HTTP 413)", "provider rejected request size"),
        ("Model provider rejected the request (HTTP 429)", "provider rate limit reached"),
        ("Transcript is too large for one model request", "window exceeds request size limit"),
        ("The model returned invalid boundary data", "provider returned invalid word indices"),
        (
            "The model response was incomplete or invalid",
            "provider returned an incomplete response",
        ),
        ("The model refused the request", "provider refused this section"),
        (
            "Model could not return structured boundaries",
            "provider could not return structured boundaries",
        ),
    ],
)
def test_provider_failure_stops_without_retry_and_reports_resume_index(
    status: str, reason: str
) -> None:
    words = _words(10)
    proposer = FakeProposer({0: (BoundaryProposal(0, 2),), 3: status})
    result = analyze_windows(
        words, 10_000, proposer, window_ms=6000, overlap_ms=3000, max_words=100
    )
    assert [call[0] for call in proposer.calls] == [0, 3]
    assert [w.status for w in result.windows] == ["processed", "failed"]
    assert result.windows[-1].error == reason
    assert result.next_window_index == 1
    assert not result.processing_complete
    assert len(result.candidates) == 1


def test_word_limit_and_partial_processing_are_exposed() -> None:
    words = _words(100)
    proposer = FakeProposer({})
    result = analyze_windows(
        words, 100_000, proposer, max_windows=1, window_ms=90_000, overlap_ms=60_000, max_words=20
    )
    assert result.windows[0].window.word_limit_reached
    assert result.next_window_index == 1
    assert not result.processing_complete
    assert len(proposer.calls) == 1


def test_empty_transcript_does_not_call_provider() -> None:
    proposer = FakeProposer({})
    result = analyze_windows((), 1000, proposer)
    assert proposer.calls == []
    assert result.candidates == ()
    assert result.processing_complete


def test_mixed_language_and_speaker_values_preserved() -> None:
    words = (
        TranscriptWord("Hello", 0, 1000, speaker=0),
        TranscriptWord("नमस्ते", 1000, 2000, speaker=1),
        TranscriptWord("friend.", 2000, 3000, speaker=0),
    )
    proposer = FakeProposer({0: (BoundaryProposal(0, 2),)})
    result = analyze_windows(words, 3000, proposer, duration_mode="short")
    assert result.candidates[0].words == words
    assert proposer.calls[0][2:] == ("short", "segments")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_mode": "unknown"},
        {"start_window": -1},
        {"start_window": True},
        {"max_windows": 0},
        {"max_windows": True},
        {"start_window": 99},
    ],
)
def test_invalid_settings_rejected_before_provider(kwargs: dict[str, object]) -> None:
    proposer = FakeProposer({})
    with pytest.raises(ValueError):
        analyze_windows(_words(10), 10_000, proposer, **kwargs)  # type: ignore[arg-type]
    assert proposer.calls == []


def test_proposal_outside_its_window_is_rejected_even_if_global_indices_exist() -> None:
    words = _words(10)
    proposer = FakeProposer({0: (BoundaryProposal(3, 9), BoundaryProposal(0, 2))})
    result = analyze_windows(
        words,
        10_000,
        proposer,
        window_ms=6000,
        overlap_ms=3000,
        max_words=100,
        max_windows=1,
    )
    assert [(c.start_word_index, c.end_word_index) for c in result.candidates] == [(0, 2)]
    assert result.windows[0].rejected == 1
    assert not result.processing_complete


def test_provider_rejected_proposals_are_counted_and_retry_window_identified() -> None:
    words = _words(10)

    class RejectingProposer:
        def propose(
            self,
            words: Sequence[TranscriptWord],
            *,
            first_word_index: int = 0,
            duration_mode: str = "both",
            input_format: str = "segments",
        ) -> BoundaryResult:
            return BoundaryResult(
                (BoundaryProposal(first_word_index, first_word_index),),
                ModelUsage(10, 10),
                rejected_proposals=2,
            )

    result = analyze_windows(
        words,
        10_000,
        RejectingProposer(),
        window_ms=6000,
        overlap_ms=3000,
        max_words=100,
        max_windows=1,
    )
    assert result.windows[0].accepted == 1
    assert result.windows[0].rejected == 2
    assert result.next_window_index == 0
    assert not result.processing_complete


def test_provider_duplicate_count_is_preserved() -> None:
    class DuplicateProposer:
        def propose(
            self,
            words: Sequence[TranscriptWord],
            *,
            first_word_index: int = 0,
            duration_mode: str = "both",
            input_format: str = "segments",
        ) -> BoundaryResult:
            return BoundaryResult(
                (BoundaryProposal(first_word_index, first_word_index),),
                None,
                duplicate_proposals=2,
            )

    result = analyze_windows(_words(2), 2000, DuplicateProposer())
    assert result.windows[0].accepted == 1
    assert result.windows[0].duplicates == 2
    assert result.processing_complete


def test_prior_rejected_window_takes_priority_over_later_provider_failure() -> None:
    words = _words(10)
    proposer = FakeProposer(
        {0: (BoundaryProposal(0, 9),), 3: "Model provider rejected the request (HTTP 429)"}
    )
    result = analyze_windows(
        words, 10_000, proposer, window_ms=6000, overlap_ms=3000, max_words=100
    )
    assert [w.status for w in result.windows] == ["processed", "failed"]
    assert result.windows[0].rejected == 1
    assert result.next_window_index == 0
    assert not result.processing_complete


def test_rejected_final_window_produces_retry_index_not_none() -> None:
    words = _words(3)
    proposer = FakeProposer({0: (BoundaryProposal(0, 9),)})
    result = analyze_windows(words, 3000, proposer)
    assert result.windows[0].rejected == 1
    assert result.next_window_index == 0
    assert not result.processing_complete


def test_unrecognized_provider_error_is_sanitized_without_transcript_or_key_leaks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "fixture-api-secret"
    private_words = "private-transcript-phrase"
    words = (TranscriptWord(private_words, 0, 1000),)
    proposer = FakeProposer({0: f"unexpected {secret} {private_words}"})
    result = analyze_windows(words, 1000, proposer)
    assert result.windows[0].error == "provider request or response failed"
    assert result.windows[0].status == "failed"
    assert result.next_window_index == 0
    assert not result.processing_complete
    rendered = repr(result)
    captured = capsys.readouterr()
    for output in (rendered, captured.out, captured.err):
        assert secret not in output
        assert private_words not in output
