"""Rank saved transcript candidates using an explicitly requested Groq call."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import SettingsError

from clipforge.candidates_cli import _TranscriptData, _WordData
from clipforge.semantic_candidates_cli import Settings
from clipforge.services.groq_ranking import GroqRanker, GroqRankingError
from clipforge.services.ranking import (
    RankingAssessment,
    RankingError,
    build_ranking_candidates,
    validate_rankings,
)
from clipforge.services.semantic_candidates import BoundaryProposal
from clipforge.services.transcription import InvalidAudioError, TranscriptWord, _wav_duration


class _CandidateData(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    start_word_index: int
    end_word_index: int
    start_ms: int
    end_ms: int
    words: list[_WordData]


class _CandidatesData(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    candidates: list[_CandidateData]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Experimentally rank candidate transcripts using Groq. "
            "Transcript text is sent to Groq and may use API credits."
        )
    )
    parser.add_argument("transcript", type=Path, help="Source transcript JSON")
    parser.add_argument("audio", type=Path, help="WAV used to produce the transcript")
    parser.add_argument("candidates", type=Path, help="Candidate JSON for this transcript")
    parser.add_argument("output", type=Path, help="Path for a new private ranking JSON file")
    parser.add_argument("--batch-size", type=int, default=4, help="Candidates per request (1–8)")
    parser.add_argument("--start-batch", type=int, default=0, help="First batch index")
    parser.add_argument("--max-batches", type=int, default=1, help="Maximum requests in this run")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 8 or args.start_batch < 0 or args.max_batches <= 0:
        print("Error: invalid batch settings", file=sys.stderr)
        return 1
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        print("Error: output must be a new file in an existing directory", file=sys.stderr)
        return 1
    try:
        if args.audio.suffix.lower() != ".wav" or not args.audio.is_file():
            raise InvalidAudioError("Choose the WAV used for transcription")
        frames, rate = _wav_duration(args.audio)
        duration_ms = frames * 1000 // rate
        transcript = _TranscriptData.model_validate_json(
            args.transcript.read_text(encoding="utf-8")
        )
        words = tuple(TranscriptWord(**word.model_dump()) for word in transcript.words)
        supplied = _CandidatesData.model_validate_json(args.candidates.read_text(encoding="utf-8"))
        proposals = tuple(
            BoundaryProposal(item.start_word_index, item.end_word_index)
            for item in supplied.candidates
        )
        candidates = build_ranking_candidates(words, duration_ms, proposals)
        for item, canonical in zip(supplied.candidates, candidates, strict=True):
            clip = canonical.candidate
            supplied_words = tuple((word.text, word.start_ms, word.end_ms) for word in item.words)
            expected_words = tuple((word.text, word.start_ms, word.end_ms) for word in clip.words)
            if (item.start_ms, item.end_ms) != (clip.start_ms, clip.end_ms) or (
                supplied_words != expected_words
            ):
                raise RankingError("Candidates do not match the source transcript")
        total_batches = (len(candidates) + args.batch_size - 1) // args.batch_size
        if args.start_batch > total_batches or (candidates and args.start_batch == total_batches):
            raise RankingError("Starting batch is beyond the candidates")
    except InvalidAudioError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RankingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError, UnicodeError, ValidationError, ValueError:
        print("Error: could not read valid transcript and candidate files", file=sys.stderr)
        return 1

    assessments: list[RankingAssessment] = []
    reports: list[dict[str, object]] = []
    if candidates:
        try:
            key = Settings().groq_api_key.get_secret_value()
        except ValidationError, SettingsError, OSError, UnicodeError:
            print("Error: could not load ranking settings", file=sys.stderr)
            return 1
        if not key.strip():
            print("Error: set GROQ_API_KEY before ranking", file=sys.stderr)
            return 1
        ranker = GroqRanker(key)
        stop = min(total_batches, args.start_batch + args.max_batches)
        for index in range(args.start_batch, stop):
            batch = candidates[index * args.batch_size : (index + 1) * args.batch_size]
            candidate_ids = [candidate.candidate_id for candidate in batch]
            try:
                response = ranker.rank(words, batch)
                check = validate_rankings(words, duration_ms, batch, response.assessments)
            except GroqRankingError as exc:
                if str(exc) == "Model provider rejected the request (HTTP 429)":
                    reason = "provider rate limit reached"
                elif str(exc) == "Model provider rejected the request (HTTP 413)":
                    reason = "provider rejected request size"
                elif str(exc) == "Candidates are too large for one ranking request":
                    reason = "candidates exceed the request size limit"
                elif str(exc) == "Model could not return structured rankings":
                    reason = "provider could not return structured rankings"
                else:
                    reason = "provider request or response failed"
                reports.append(
                    {
                        "index": index,
                        "candidate_ids": candidate_ids,
                        "status": "failed",
                        "error": reason,
                    }
                )
                break
            except RankingError:
                reports.append(
                    {
                        "index": index,
                        "candidate_ids": candidate_ids,
                        "status": "failed",
                        "error": "invalid assessment",
                    }
                )
                break
            assessments.extend(response.assessments)
            reports.append(
                {
                    "index": index,
                    "candidate_ids": candidate_ids,
                    "status": "processed" if check.responses_complete else "partial",
                    "short_order": [item.candidate.candidate_id for item in check.short],
                    "long_order": [item.candidate.candidate_id for item in check.long],
                    "missing_ids": [candidate.candidate_id for candidate in check.missing],
                    "usage": None if response.usage is None else asdict(response.usage),
                }
            )
    ranking = validate_rankings(words, duration_ms, candidates, assessments)
    multiple_batches = sum(report["status"] != "failed" for report in reports) > 1
    ranking_data = asdict(ranking)
    if multiple_batches:
        # Independent model calls do not provide calibrated scores. Keep aggregate
        # entries in source input order; each report retains its own ranked order.
        source_order = {candidate.candidate_id: i for i, candidate in enumerate(candidates)}
        for group in ("short", "long"):
            ranking_data[group] = sorted(
                ranking_data[group],
                key=lambda item: source_order[item["candidate"]["candidate_id"]],
            )
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "provider": "groq" if candidates else None,
                        "model": "openai/gpt-oss-120b" if candidates else None,
                        "audio_duration_ms": duration_ms,
                        "total_batches": total_batches,
                        "batch_size": args.batch_size,
                        "score_scope": (
                            None
                            if not assessments
                            else "uncalibrated_batches"
                            if multiple_batches
                            else "single_batch"
                        ),
                        "batches": reports,
                        "ranking": ranking_data,
                    },
                    output,
                    ensure_ascii=False,
                    indent=2,
                )
                output.write("\n")
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
    except OSError:
        print("Error: could not create the ranking file", file=sys.stderr)
        return 1
    print(f"Ranked {len(assessments)} of {len(candidates)} candidates in {args.output}")
    if not ranking.responses_complete:
        print("Some candidates have not been assessed; inspect output", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
