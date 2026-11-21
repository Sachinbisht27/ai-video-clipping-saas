"""Validate model assessments against the exact source transcript and clips.

This module does not assess content, call a model, or predict performance. A
model's rationale remains untrusted and its ratings require human review.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from clipforge.services.semantic_candidates import (
    BoundaryProposal,
    SemanticCandidate,
    SemanticCandidateError,
    validate_proposals,
)
from clipforge.services.transcription import TranscriptWord


class RankingError(ValueError):
    """Candidates or model assessments cannot be safely matched to source data."""


@dataclass(frozen=True, slots=True)
class RankingCandidate:
    """One validated source clip with a stable ID derived from its word indices."""

    candidate_id: str
    candidate: SemanticCandidate


@dataclass(frozen=True, slots=True)
class RankingAssessment:
    """Untrusted model assessment, with optional citation to source word indices."""

    candidate_id: str
    decision: Literal["suitable", "unsuitable", "uncertain"]
    score: int | None
    reason: str
    evidence_start_word_index: int | None
    evidence_end_word_index: int | None


@dataclass(frozen=True, slots=True)
class AssessedCandidate:
    """An assessment linked to a validated clip and exact source evidence words."""

    candidate: RankingCandidate
    assessment: RankingAssessment
    evidence_words: tuple[TranscriptWord, ...]


@dataclass(frozen=True, slots=True)
class RankingResult:
    """Separate ratings and retain abstentions and candidates missing a response.

    ``responses_complete`` only describes whether every input received an
    assessment; it does not mean assessments are correct or the clips reviewed.
    """

    short: tuple[AssessedCandidate, ...]
    long: tuple[AssessedCandidate, ...]
    unsuitable: tuple[AssessedCandidate, ...]
    uncertain: tuple[AssessedCandidate, ...]
    missing: tuple[RankingCandidate, ...]
    responses_complete: bool


def _candidate_id(start: int, end: int) -> str:
    return f"w{start}_{end}"


def build_ranking_candidates(
    words: Sequence[TranscriptWord], duration_ms: int, proposals: Sequence[BoundaryProposal]
) -> tuple[RankingCandidate, ...]:
    """Reconstruct and validate clips from source word IDs, rejecting duplicates."""
    try:
        clips = validate_proposals(words, duration_ms, proposals, target_max_ms=180_000)
    except SemanticCandidateError:
        raise RankingError("Invalid source transcript or candidate boundaries") from None
    return tuple(
        RankingCandidate(_candidate_id(clip.start_word_index, clip.end_word_index), clip)
        for clip in clips
    )


def validate_rankings(
    words: Sequence[TranscriptWord],
    duration_ms: int,
    candidates: Sequence[RankingCandidate],
    assessments: Sequence[RankingAssessment],
) -> RankingResult:
    """Verify IDs, source data, decisions, and citations before ordering results.

    A missing response or an explicit uncertain assessment is retained and never
    promoted to a score. A malformed, duplicate, or unknown ID rejects the model
    response; error messages do not include transcript text or model output.
    """
    supplied = tuple(candidates)
    if any(
        not isinstance(item, RankingCandidate) or not isinstance(item.candidate, SemanticCandidate)
        for item in supplied
    ):
        raise RankingError("Invalid ranking candidate")
    proposals = tuple(
        BoundaryProposal(item.candidate.start_word_index, item.candidate.end_word_index)
        for item in supplied
    )
    canonical = build_ranking_candidates(words, duration_ms, proposals)
    for supplied_candidate, expected in zip(supplied, canonical, strict=True):
        actual_clip = supplied_candidate.candidate
        source_clip = expected.candidate
        if (
            supplied_candidate.candidate_id != expected.candidate_id
            or actual_clip.start_ms != source_clip.start_ms
            or actual_clip.end_ms != source_clip.end_ms
            or actual_clip.words != source_clip.words
        ):
            raise RankingError("Ranking candidate does not match source transcript")

    by_id = {candidate.candidate_id: candidate for candidate in canonical}
    by_assessment: dict[str, RankingAssessment] = {}
    for assessment in assessments:
        if (
            not isinstance(assessment, RankingAssessment)
            or not isinstance(assessment.candidate_id, str)
            or assessment.candidate_id not in by_id
            or assessment.candidate_id in by_assessment
            or not isinstance(assessment.decision, str)
            or assessment.decision not in {"suitable", "unsuitable", "uncertain"}
            or not isinstance(assessment.reason, str)
            or not assessment.reason.strip()
            or len(assessment.reason) > 500
        ):
            raise RankingError("Invalid, duplicate, or unknown model assessment")
        if assessment.decision == "suitable":
            if type(assessment.score) is not int or not 0 <= assessment.score <= 100:
                raise RankingError("Invalid model score")
        elif assessment.score is not None:
            raise RankingError("Only suitable candidates may have a score")
        evidence_start = assessment.evidence_start_word_index
        evidence_end = assessment.evidence_end_word_index
        no_evidence = evidence_start is None and evidence_end is None
        if no_evidence:
            if assessment.decision != "uncertain":
                raise RankingError("Model evidence is required for this decision")
        else:
            clip = by_id[assessment.candidate_id].candidate
            if (
                type(evidence_start) is not int
                or type(evidence_end) is not int
                or not clip.start_word_index
                <= evidence_start
                <= evidence_end
                <= clip.end_word_index
            ):
                raise RankingError("Model evidence is outside the candidate")
        by_assessment[assessment.candidate_id] = assessment

    short: list[AssessedCandidate] = []
    long: list[AssessedCandidate] = []
    unsuitable: list[AssessedCandidate] = []
    uncertain: list[AssessedCandidate] = []
    missing: list[RankingCandidate] = []
    source_words = tuple(words)
    for candidate in canonical:
        found_assessment = by_assessment.get(candidate.candidate_id)
        if found_assessment is None:
            missing.append(candidate)
            continue
        start = found_assessment.evidence_start_word_index
        end = found_assessment.evidence_end_word_index
        evidence = () if start is None or end is None else source_words[start : end + 1]
        assessed = AssessedCandidate(candidate, found_assessment, evidence)
        if found_assessment.decision == "unsuitable":
            unsuitable.append(assessed)
        elif found_assessment.decision == "uncertain":
            uncertain.append(assessed)
        elif candidate.candidate.end_ms - candidate.candidate.start_ms <= 60_000:
            short.append(assessed)
        else:
            long.append(assessed)

    # Suitable items always have integer scores. Python sorting retains input order on ties.
    short.sort(key=lambda item: -(item.assessment.score or 0))
    long.sort(key=lambda item: -(item.assessment.score or 0))
    return RankingResult(
        short=tuple(short),
        long=tuple(long),
        unsuitable=tuple(unsuitable),
        uncertain=tuple(uncertain),
        missing=tuple(missing),
        responses_complete=not missing,
    )
