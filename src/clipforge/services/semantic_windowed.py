"""Process transcript windows without hiding incomplete work or invalid proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from clipforge.services.groq_boundaries import GroqBoundaryError
from clipforge.services.openai_boundaries import BoundaryResult, ModelUsage
from clipforge.services.semantic_candidates import (
    SemanticCandidate,
    SemanticCandidateError,
    validate_proposals,
)
from clipforge.services.transcript_windows import TranscriptWindow, make_windows
from clipforge.services.transcription import TranscriptWord


class WindowedProposer(Protocol):
    def propose(
        self,
        words: Sequence[TranscriptWord],
        *,
        first_word_index: int = 0,
        duration_mode: Literal["both", "short", "long"] = "both",
        input_format: Literal["words", "segments"] = "segments",
    ) -> BoundaryResult: ...


@dataclass(frozen=True, slots=True)
class WindowReport:
    index: int
    window: TranscriptWindow
    status: Literal["processed", "failed"]
    accepted: int
    duplicates: int
    rejected: int
    usage: ModelUsage | None
    error: str | None


@dataclass(frozen=True, slots=True)
class WindowedResult:
    candidates: tuple[SemanticCandidate, ...]
    windows: tuple[WindowReport, ...]
    total_windows: int
    next_window_index: int | None
    processing_complete: bool


def analyze_windows(
    words: Sequence[TranscriptWord],
    duration_ms: int,
    proposer: WindowedProposer,
    *,
    duration_mode: Literal["both", "short", "long"] = "both",
    start_window: int = 0,
    max_windows: int | None = None,
    window_ms: int = 360_000,
    overlap_ms: int = 180_000,
    max_words: int = 900,
) -> WindowedResult:
    """Process selected windows; stop and report provider failures without retry.

    Validate every proposal against the complete source transcript. Exact word-index
    pairs are deduplicated, while other boundary variants remain available. Completion
    refers to request processing, not quality or a guarantee that every possible
    long thought fit in a single window. A retry starts at the earliest rejected or
    unprocessed window; results from separate invocations are not deduplicated.
    """
    if (
        type(start_window) is not int
        or start_window < 0
        or (max_windows is not None and (type(max_windows) is not int or max_windows <= 0))
        or duration_mode not in {"both", "short", "long"}
    ):
        raise ValueError("Invalid window selection settings")
    items = tuple(words)
    validate_proposals(items, duration_ms, ())
    windows = make_windows(items, window_ms=window_ms, overlap_ms=overlap_ms, max_words=max_words)
    if start_window > len(windows):
        raise ValueError("Starting window is beyond the transcript")
    stop = len(windows) if max_windows is None else min(len(windows), start_window + max_windows)
    selected: list[SemanticCandidate] = []
    seen: set[tuple[int, int]] = set()
    reports: list[WindowReport] = []
    next_window = stop
    for index in range(start_window, stop):
        window = windows[index]
        try:
            result = proposer.propose(
                items[window.start_word_index : window.end_word_index],
                first_word_index=window.start_word_index,
                duration_mode=duration_mode,
                input_format="segments",
            )
        except GroqBoundaryError as exc:
            if str(exc) == "Model provider rejected the request (HTTP 429)":
                reason = "provider rate limit reached"
            elif str(exc) == "Model provider rejected the request (HTTP 413)":
                reason = "provider rejected request size"
            elif str(exc) == "Transcript is too large for one model request":
                reason = "window exceeds request size limit"
            elif str(exc) == "The model returned invalid boundary data":
                reason = "provider returned invalid word indices"
            elif str(exc) == "The model response was incomplete or invalid":
                reason = "provider returned an incomplete response"
            elif str(exc) == "The model refused the request":
                reason = "provider refused this section"
            elif str(exc) == "Model could not return structured boundaries":
                reason = "provider could not return structured boundaries"
            else:
                reason = "provider request or response failed"
            reports.append(WindowReport(index, window, "failed", 0, 0, 0, None, reason))
            next_window = min(next_window, index)
            break
        accepted = 0
        duplicates = result.duplicate_proposals
        rejected = result.rejected_proposals
        for proposal in result.proposals:
            if not (
                type(proposal.start_word_index) is int
                and type(proposal.end_word_index) is int
                and window.start_word_index <= proposal.start_word_index
                and proposal.start_word_index <= proposal.end_word_index < window.end_word_index
            ):
                rejected += 1
                continue
            try:
                candidate = validate_proposals(
                    items,
                    duration_ms,
                    (proposal,),
                    target_min_ms=60_000 if duration_mode == "long" else 20_000,
                    target_max_ms=60_000 if duration_mode == "short" else 180_000,
                )[0]
            except SemanticCandidateError:
                rejected += 1
                continue
            identity = (candidate.start_word_index, candidate.end_word_index)
            if identity in seen:
                duplicates += 1
            else:
                selected.append(candidate)
                seen.add(identity)
                accepted += 1
        if rejected:
            next_window = min(next_window, index)
        reports.append(
            WindowReport(
                index, window, "processed", accepted, duplicates, rejected, result.usage, None
            )
        )
    next_index = next_window if next_window < len(windows) else None
    return WindowedResult(
        candidates=tuple(sorted(selected, key=lambda c: (c.start_ms, c.end_ms))),
        windows=tuple(reports),
        total_windows=len(windows),
        next_window_index=next_index,
        processing_complete=(
            start_window == 0
            and next_index is None
            and all(report.rejected == 0 for report in reports)
        ),
    )
