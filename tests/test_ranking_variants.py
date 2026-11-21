"""Boundary-variant hints preserve every candidate and require no model call."""

from dataclasses import replace

import pytest

from clipforge.services.ranking import (
    RankingAssessment,
    RankingCandidate,
    RankingError,
    build_ranking_candidates,
    validate_rankings,
)
from clipforge.services.ranking_variants import group_boundary_variants
from clipforge.services.semantic_candidates import BoundaryProposal, SemanticCandidate
from clipforge.services.transcription import TranscriptWord


def clip(first: int, last: int, start_ms: int, end_ms: int) -> RankingCandidate:
    """Build a lightweight interval fixture; production inputs are source-validated."""
    word = TranscriptWord("synthetic", start_ms, end_ms)
    candidate = SemanticCandidate(first, last, start_ms, end_ms, (word,), "within_target")
    return RankingCandidate(f"w{first}_{last}", candidate)


def test_reviewed_seventy_and_three_minute_boundary_variants_are_grouped() -> None:
    # Synthetic text, with the previously reviewed candidate indices and timings.
    short_a = clip(646, 801, 236_135, 306_430)
    short_b = clip(646, 807, 236_135, 307_710)
    long_a = clip(1764, 2142, 701_955, 872_985)
    long_b = clip(1760, 2140, 700_995, 872_425)
    items = (short_a, long_a, short_b, long_b)
    assert group_boundary_variants(items) == (
        ("w646_801", "w646_807"),
        ("w1764_2142", "w1760_2140"),
    )
    assert items == (short_a, long_a, short_b, long_b)


def test_reviewed_short_clip_boundaries_are_grouped_without_losing_either() -> None:
    first = clip(0, 43, 0, 13_555)
    second = clip(0, 52, 0, 16_195)
    candidates = (first, second)
    assert group_boundary_variants(candidates) == (("w0_43", "w0_52"),)
    assert candidates == (first, second)


def test_short_clip_thresholds_include_exact_boundaries() -> None:
    base = clip(0, 1, 0, 12_000)
    exactly_eighty_percent = clip(0, 2, 0, 15_000)
    assert group_boundary_variants((base, exactly_eighty_percent)) == (("w0_1", "w0_2"),)


def test_short_clips_still_require_both_overlap_and_small_endpoint_shifts() -> None:
    assert group_boundary_variants((clip(0, 1, 0, 8000), clip(0, 2, 0, 10_001))) == ()
    # Overlap exceeds 80%, but the ending differs by more than three seconds.
    assert group_boundary_variants((clip(0, 1, 0, 16_000), clip(0, 2, 0, 19_500))) == ()
    # Both endpoint shifts are within three seconds, but overlap is too low.
    assert group_boundary_variants((clip(0, 1, 0, 12_000), clip(2, 3, 2500, 14_500))) == ()


def test_short_clip_rule_does_not_apply_when_one_exceeds_twenty_seconds() -> None:
    assert group_boundary_variants((clip(0, 1, 0, 19_000), clip(0, 2, 0, 21_000))) == ()
    assert group_boundary_variants((clip(0, 1, 0, 20_000), clip(0, 2, 0, 21_000))) == (
        ("w0_1", "w0_2"),
    )


def test_short_clip_grouping_still_uses_the_original_anchor() -> None:
    first = clip(0, 1, 0, 15_000)
    middle = clip(2, 3, 1500, 16_500)
    last = clip(4, 5, 3000, 18_000)
    assert group_boundary_variants((middle, last)) == (("w2_3", "w4_5"),)
    assert group_boundary_variants((first, middle, last)) == (("w0_1", "w2_3"),)


def test_overlap_and_each_endpoint_must_meet_thresholds() -> None:
    base = clip(0, 1, 0, 80_000)
    at_threshold = clip(0, 2, 0, 76_000)
    below_iou = clip(0, 3, 0, 75_999)
    assert group_boundary_variants((base, at_threshold)) == (("w0_1", "w0_2"),)
    assert group_boundary_variants((base, below_iou)) == ()

    long_base = clip(10, 20, 0, 200_000)
    far_end = clip(10, 21, 0, 206_000)  # IoU >95%, but six-second end shift.
    far_start = clip(11, 20, 5_001, 205_001)  # IoU >95%, but start shift >5s.
    assert group_boundary_variants((long_base, far_end)) == ()
    assert group_boundary_variants((long_base, far_start)) == ()


def test_overlapping_but_meaningfully_different_sections_are_not_grouped() -> None:
    assert group_boundary_variants((clip(0, 10, 0, 120_000), clip(5, 15, 30_000, 150_000))) == ()
    assert group_boundary_variants((clip(0, 10, 0, 30_000), clip(11, 20, 30_000, 60_000))) == ()


def test_anchor_comparison_prevents_transitive_chaining() -> None:
    first = clip(0, 100, 0, 200_000)
    middle = clip(1, 101, 4_000, 204_000)
    last = clip(2, 102, 8_000, 208_000)
    assert group_boundary_variants((middle, last)) == (("w1_101", "w2_102"),)
    assert group_boundary_variants((first, middle, last)) == (("w0_100", "w1_101"),)


def test_similar_variants_can_cross_the_sixty_second_category_boundary() -> None:
    words = (
        TranscriptWord("start", 0, 10_000, speaker=0),
        TranscriptWord("thought.", 10_000, 59_000, speaker=1),
        TranscriptWord("continued.", 59_000, 61_000, speaker=1),
    )
    candidates = build_ranking_candidates(
        words, 61_000, (BoundaryProposal(0, 1), BoundaryProposal(0, 2))
    )
    before = validate_rankings(
        words,
        61_000,
        candidates,
        (
            RankingAssessment("w0_1", "suitable", 75, "Clear point", 0, 1),
            RankingAssessment("w0_2", "suitable", 90, "Complete point", 0, 2),
        ),
    )
    assert group_boundary_variants(candidates) == (("w0_1", "w0_2"),)
    after = validate_rankings(
        words,
        61_000,
        candidates,
        (
            RankingAssessment("w0_1", "suitable", 75, "Clear point", 0, 1),
            RankingAssessment("w0_2", "suitable", 90, "Complete point", 0, 2),
        ),
    )
    assert before == after
    assert before.short[0].assessment.score == 75
    assert before.long[0].assessment.score == 90


def test_first_matching_anchor_wins_and_input_order_is_stable() -> None:
    a = clip(0, 10, 0, 100_000)
    b = clip(0, 11, 0, 101_000)
    c = clip(0, 12, 0, 102_000)
    d = clip(20, 30, 150_000, 190_000)
    assert group_boundary_variants((a, b, c, d)) == (("w0_10", "w0_11", "w0_12"),)
    assert group_boundary_variants((c, a, b, d)) == (("w0_12", "w0_10", "w0_11"),)
    assert group_boundary_variants(()) == ()
    assert group_boundary_variants((d,)) == ()


def test_duplicate_ids_and_invalid_candidate_shapes_fail_safely() -> None:
    valid = clip(0, 1, 0, 1000)
    with pytest.raises(RankingError, match="duplicate"):
        group_boundary_variants((valid, valid))
    with pytest.raises(RankingError):
        group_boundary_variants((replace(valid, candidate_id="not-canonical"),))
    with pytest.raises(RankingError):
        group_boundary_variants((replace(valid, candidate=replace(valid.candidate, end_ms=0)),))
    with pytest.raises(RankingError):
        group_boundary_variants(
            (replace(valid, candidate=replace(valid.candidate, start_ms=True)),)
        )
