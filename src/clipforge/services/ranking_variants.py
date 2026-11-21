"""Group similar clip boundaries as review hints without modifying candidates."""

from __future__ import annotations

from collections.abc import Sequence

from clipforge.services.ranking import RankingCandidate, RankingError
from clipforge.services.semantic_candidates import SemanticCandidate


def group_boundary_variants(
    candidates: Sequence[RankingCandidate],
) -> tuple[tuple[str, ...], ...]:
    """Return candidate ID groups that share similar time intervals.

    Candidates must already be validated against the same source transcript.
    Each group is anchored on its first input candidate; subsequent candidates
    must overlap that anchor by at least 95% of their union, and both their start
    and end must differ from the anchor by at most five seconds. When both clips
    are at most 20 seconds long, the thresholds are 80% and three seconds. This
    avoids joining a chain of individually similar but progressively different clips.

    Groups are hints for review, not a quality judgment. All inputs and scores
    remain unchanged; ungrouped candidates are omitted from the returned groups.
    """
    items = tuple(candidates)
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, RankingCandidate) or not isinstance(
            item.candidate, SemanticCandidate
        ):
            raise RankingError("Invalid ranking candidate")
        candidate = item.candidate
        if (
            type(candidate.start_word_index) is not int
            or type(candidate.end_word_index) is not int
            or candidate.start_word_index < 0
            or candidate.end_word_index < candidate.start_word_index
            or item.candidate_id != f"w{candidate.start_word_index}_{candidate.end_word_index}"
            or item.candidate_id in seen
            or type(candidate.start_ms) is not int
            or type(candidate.end_ms) is not int
            or candidate.start_ms < 0
            or candidate.end_ms <= candidate.start_ms
        ):
            raise RankingError("Invalid or duplicate ranking candidate")
        seen.add(item.candidate_id)

    groups: list[tuple[str, ...]] = []
    assigned: set[int] = set()
    for anchor_index, anchor_item in enumerate(items):
        if anchor_index in assigned:
            continue
        anchor = anchor_item.candidate
        members = [anchor_item.candidate_id]
        for other_index in range(anchor_index + 1, len(items)):
            if other_index in assigned:
                continue
            other = items[other_index].candidate
            both_short = (
                anchor.end_ms - anchor.start_ms <= 20_000
                and other.end_ms - other.start_ms <= 20_000
            )
            max_shift_ms = 3_000 if both_short else 5_000
            if (
                abs(anchor.start_ms - other.start_ms) > max_shift_ms
                or abs(anchor.end_ms - other.end_ms) > max_shift_ms
            ):
                continue
            overlap = max(
                0, min(anchor.end_ms, other.end_ms) - max(anchor.start_ms, other.start_ms)
            )
            union = max(anchor.end_ms, other.end_ms) - min(anchor.start_ms, other.start_ms)
            if overlap * 100 >= union * (80 if both_short else 95):
                members.append(items[other_index].candidate_id)
                assigned.add(other_index)
        if len(members) > 1:
            groups.append(tuple(members))
            assigned.add(anchor_index)
    return tuple(groups)
