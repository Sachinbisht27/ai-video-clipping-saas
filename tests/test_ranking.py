"""Ranking contract tests use synthetic transcripts and no external provider."""

from dataclasses import replace
from typing import Any

import pytest

from clipforge.services.ranking import (
    RankingAssessment,
    RankingCandidate,
    RankingError,
    build_ranking_candidates,
    validate_rankings,
)
from clipforge.services.semantic_candidates import BoundaryProposal
from clipforge.services.transcription import TranscriptWord


def words(count: int = 100) -> tuple[TranscriptWord, ...]:
    return tuple(
        TranscriptWord(text=f"word-{i}", start_ms=i * 1000, end_ms=(i + 1) * 1000, speaker=i % 2)
        for i in range(count)
    )


def assessment(
    candidate_id: str,
    decision: Any = "suitable",
    score: Any = 50,
    reason: Any = "Useful excerpt",
    evidence_start: Any = 0,
    evidence_end: Any = 1,
) -> RankingAssessment:
    return RankingAssessment(
        candidate_id=candidate_id,
        decision=decision,
        score=score,
        reason=reason,
        evidence_start_word_index=evidence_start,
        evidence_end_word_index=evidence_end,
    )


def test_ranking_groups_lengths_orders_scores_and_preserves_ties() -> None:
    source = words()
    candidates = build_ranking_candidates(
        source,
        100_000,
        (
            BoundaryProposal(0, 19),
            BoundaryProposal(10, 29),
            BoundaryProposal(0, 69),
            BoundaryProposal(10, 79),
            BoundaryProposal(0, 59),
        ),
    )
    assert [c.candidate_id for c in candidates] == ["w0_19", "w10_29", "w0_69", "w10_79", "w0_59"]
    result = validate_rankings(
        source,
        100_000,
        candidates,
        (
            assessment("w0_19", score=90, evidence_start=2, evidence_end=3),
            assessment("w10_29", score=90, evidence_start=10, evidence_end=11),
            assessment("w0_69", score=20),
            assessment("w10_79", score=80, evidence_start=11, evidence_end=12),
            assessment("w0_59", score=30),
        ),
    )
    assert [x.candidate.candidate_id for x in result.short] == ["w0_19", "w10_29", "w0_59"]
    assert [x.candidate.candidate_id for x in result.long] == ["w10_79", "w0_69"]
    assert result.short[0].evidence_words == source[2:4]
    assert result.short[0].candidate.candidate.words == source[:20]
    assert result.responses_complete
    assert not result.unsuitable and not result.uncertain and not result.missing


def test_unsuitable_uncertain_and_missing_are_not_promoted_to_ranked_results() -> None:
    source = words()
    candidates = build_ranking_candidates(
        source,
        100_000,
        (BoundaryProposal(0, 19), BoundaryProposal(20, 39), BoundaryProposal(40, 59)),
    )
    result = validate_rankings(
        source,
        100_000,
        candidates,
        (
            assessment("w0_19", decision="unsuitable", score=None, reason="Needs context"),
            assessment(
                "w20_39",
                decision="uncertain",
                score=None,
                reason="Cannot assess",
                evidence_start=None,
                evidence_end=None,
            ),
        ),
    )
    assert not result.short and not result.long
    assert result.unsuitable[0].candidate.candidate_id == "w0_19"
    assert result.unsuitable[0].evidence_words == source[:2]
    assert result.uncertain[0].candidate.candidate_id == "w20_39"
    assert result.uncertain[0].evidence_words == ()
    assert [c.candidate_id for c in result.missing] == ["w40_59"]
    assert not result.responses_complete


def test_uncertain_with_grounded_evidence_is_retained() -> None:
    source = words()
    candidates = build_ranking_candidates(source, 100_000, (BoundaryProposal(0, 19),))
    result = validate_rankings(
        source,
        100_000,
        candidates,
        (
            assessment(
                "w0_19", decision="uncertain", score=None, evidence_start=18, evidence_end=19
            ),
        ),
    )
    assert result.uncertain[0].evidence_words == source[18:20]
    assert result.responses_complete


def test_empty_input_is_valid_and_no_scores_are_invented() -> None:
    source = words(1)
    candidates = build_ranking_candidates(source, 1000, ())
    result = validate_rankings(source, 1000, candidates, ())
    assert result.responses_complete
    assert not result.short and not result.long and not result.missing


@pytest.mark.parametrize(
    "bad",
    [
        assessment("unknown"),
        assessment("w0_19", decision="unknown"),
        assessment("w0_19", score=True),
        assessment("w0_19", score=-1),
        assessment("w0_19", score=101),
        assessment("w0_19", score=1.5),
        assessment("w0_19", score=None),
        assessment("w0_19", decision="unsuitable", score=10),
        assessment("w0_19", decision="uncertain", score=10),
        assessment("w0_19", reason=" "),
        assessment("w0_19", reason="x" * 501),
        assessment("w0_19", evidence_start=-1),
        assessment("w0_19", evidence_start=10, evidence_end=9),
        assessment("w0_19", evidence_start=0, evidence_end=20),
        assessment("w0_19", evidence_start=True),
        assessment("w0_19", evidence_start=None, evidence_end=1),
        assessment("w0_19", evidence_start=None, evidence_end=None),
        assessment(
            "w0_19", decision="unsuitable", score=None, evidence_start=None, evidence_end=None
        ),
    ],
)
def test_invalid_assessment_is_rejected(bad: RankingAssessment) -> None:
    source = words()
    candidates = build_ranking_candidates(source, 100_000, (BoundaryProposal(0, 19),))
    with pytest.raises(RankingError):
        validate_rankings(source, 100_000, candidates, (bad,))


def test_duplicate_assessment_is_rejected_not_silently_overwritten() -> None:
    source = words()
    candidates = build_ranking_candidates(source, 100_000, (BoundaryProposal(0, 19),))
    entry = assessment("w0_19")
    with pytest.raises(RankingError, match="duplicate"):
        validate_rankings(source, 100_000, candidates, (entry, entry))


def test_candidate_identity_timing_and_words_must_match_the_source() -> None:
    source = words()
    candidate = build_ranking_candidates(source, 100_000, (BoundaryProposal(0, 19),))[0]
    for forged in (
        replace(candidate, candidate_id="w0_18"),
        replace(candidate, candidate=replace(candidate.candidate, end_ms=19_999)),
        replace(candidate, candidate=replace(candidate.candidate, words=source[1:20])),
    ):
        with pytest.raises(RankingError, match="source"):
            validate_rankings(source, 100_000, (forged,), ())


def test_duplicate_and_invalid_candidate_ranges_are_rejected() -> None:
    source = words()
    with pytest.raises(RankingError):
        build_ranking_candidates(source, 100_000, (BoundaryProposal(0, 5), BoundaryProposal(0, 5)))
    with pytest.raises(RankingError):
        build_ranking_candidates(source, 100_000, (BoundaryProposal(0, 100),))


def test_invalid_response_does_not_echo_untrusted_text() -> None:
    source = words()
    candidates = build_ranking_candidates(source, 100_000, (BoundaryProposal(0, 19),))
    secret = "private-transcript-or-fake-key"
    with pytest.raises(RankingError) as exc:
        validate_rankings(source, 100_000, candidates, (assessment(secret, reason=secret),))
    assert secret not in str(exc.value)


def test_forged_candidate_type_is_rejected() -> None:
    with pytest.raises(RankingError):
        validate_rankings(words(), 100_000, (RankingCandidate("bad", None),), ())  # type: ignore[arg-type]
