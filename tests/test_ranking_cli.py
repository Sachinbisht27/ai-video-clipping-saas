"""Ranking CLI tests use local files and simulated providers only."""

import json
import sys
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from clipforge import ranking_cli as cli
from clipforge.services.groq_ranking import GroqRankingError, GroqRankingResult
from clipforge.services.openai_boundaries import ModelUsage
from clipforge.services.ranking import RankingAssessment, build_ranking_candidates
from clipforge.services.semantic_candidates import BoundaryProposal
from clipforge.services.transcription import TranscriptWord


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GROQ_API_KEY=fixture-secret\n")
    audio = tmp_path / "source.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(100)
        output.writeframes(b"\x00\x00" * 12000)
    words = (
        TranscriptWord("First", 0, 10_000, speaker=0),
        TranscriptWord("thought.", 10_000, 25_000, speaker=1),
        TranscriptWord("Longer", 30_000, 60_000, speaker=0),
        TranscriptWord("answer.", 60_000, 105_000, speaker=1),
    )
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"words": [asdict(word) for word in words]}))
    candidates = tmp_path / "candidates.json"
    clips = build_ranking_candidates(
        words, 120_000, (BoundaryProposal(0, 1), BoundaryProposal(2, 3))
    )
    candidates.write_text(json.dumps({"candidates": [asdict(clip.candidate) for clip in clips]}))
    result = tmp_path / "rankings.json"
    monkeypatch.setattr(
        sys, "argv", ["rank", str(transcript), str(audio), str(candidates), str(result)]
    )
    return transcript, audio, candidates, result


def score(candidate: Any, value: int = 75) -> RankingAssessment:
    start = candidate.candidate.start_word_index
    return RankingAssessment(
        candidate_id=candidate.candidate_id,
        decision="suitable",
        score=value,
        reason="A complete, clear point",
        evidence_start_word_index=start,
        evidence_end_word_index=start,
    )


def fake_ranker(monkeypatch: pytest.MonkeyPatch, *, assessments: Any = None) -> list[list[str]]:
    calls: list[list[str]] = []

    class FakeRanker:
        def __init__(self, api_key: str) -> None:
            assert api_key == "fixture-secret"

        def rank(self, words: Any, candidates: Any) -> GroqRankingResult:
            calls.append([candidate.candidate_id for candidate in candidates])
            response = tuple(score(candidate) for candidate in candidates)
            if assessments is not None:
                response = assessments(candidates)
            return GroqRankingResult(response, ModelUsage(100, 30, 12))

    monkeypatch.setattr(cli, "GroqRanker", FakeRanker)
    return calls


def test_full_batch_writes_private_short_and_long_ranking(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, output = inputs
    calls = fake_ranker(monkeypatch)
    assert cli.main() == 0
    assert calls == [["w0_1", "w2_3"]]
    result = json.loads(output.read_text())
    assert result["provider"] == "groq"
    assert result["score_scope"] == "single_batch"
    assert result["batches"][0]["candidate_ids"] == ["w0_1", "w2_3"]
    assert result["ranking"]["responses_complete"] is True
    assert result["ranking"]["short"][0]["candidate"]["candidate_id"] == "w0_1"
    assert result["ranking"]["long"][0]["candidate"]["candidate_id"] == "w2_3"
    assert result["batches"][0]["usage"]["input_tokens"] == 100
    assert result["ranking"]["short"][0]["evidence_words"][0]["text"] == "First"
    assert output.stat().st_mode & 0o777 == 0o600
    assert "fixture-secret" not in output.read_text()


def test_corrected_transcript_speakers_override_old_candidate_labels(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, _, _, output = inputs
    data = json.loads(transcript.read_text())
    data["words"][1]["original_speaker"] = data["words"][1]["speaker"]
    data["words"][1]["speaker"] = 0
    transcript.write_text(json.dumps(data))
    calls = fake_ranker(monkeypatch)
    assert cli.main() == 0
    assert calls == [["w0_1", "w2_3"]]
    result = json.loads(output.read_text())
    first = result["ranking"]["short"][0]["candidate"]["candidate"]["words"]
    assert [word["speaker"] for word in first] == [0, 0]


def test_partial_batch_retains_missing_candidates_and_returns_partial_status(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, candidates, output = inputs
    monkeypatch.setattr(
        sys,
        "argv",
        ["rank", str(transcript), str(audio), str(candidates), str(output), "--batch-size", "1"],
    )
    calls = fake_ranker(monkeypatch)
    assert cli.main() == 2
    assert calls == [["w0_1"]]
    result = json.loads(output.read_text())
    assert result["ranking"]["responses_complete"] is False
    assert result["score_scope"] == "single_batch"
    assert result["batches"][0]["candidate_ids"] == ["w0_1"]
    assert [c["candidate_id"] for c in result["ranking"]["missing"]] == ["w2_3"]
    assert len(result["ranking"]["short"]) == 1
    assert not result["ranking"]["long"]


def test_partial_provider_response_is_reported_without_inventing_scores(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, output = inputs
    fake_ranker(monkeypatch, assessments=lambda candidates: (score(candidates[0]),))
    assert cli.main() == 2
    result = json.loads(output.read_text())
    assert result["batches"][0]["status"] == "partial"
    assert result["batches"][0]["candidate_ids"] == ["w0_1", "w2_3"]
    assert result["batches"][0]["missing_ids"] == ["w2_3"]
    assert [c["candidate_id"] for c in result["ranking"]["missing"]] == ["w2_3"]


def test_empty_candidates_do_not_need_credentials_or_provider(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, _, candidates, output = inputs
    (transcript.parent / ".env").unlink()
    candidates.write_text('{"candidates": []}')

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Empty input must not call a provider")

    monkeypatch.setattr(cli, "GroqRanker", no_provider)
    assert cli.main() == 0
    result = json.loads(output.read_text())
    assert result["provider"] is None
    assert result["score_scope"] is None
    assert result["ranking"]["responses_complete"] is True
    assert result["ranking"]["short"] == []


def test_missing_groq_key_fails_without_call_or_output(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, _, _, output = inputs
    (transcript.parent / ".env").write_text("OPENAI_API_KEY=other-service-secret\n")

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("A different provider's key must not trigger a request")

    monkeypatch.setattr(cli, "GroqRanker", no_provider)
    assert cli.main() == 1
    assert not output.exists()


def test_existing_output_and_dangling_symlink_are_never_overwritten(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, output = inputs

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Existing output must fail before a provider request")

    monkeypatch.setattr(cli, "GroqRanker", no_provider)
    output.write_text("keep me")
    assert cli.main() == 1
    assert output.read_text() == "keep me"
    output.unlink()
    output.symlink_to(output.parent / "missing")
    assert cli.main() == 1
    assert output.is_symlink()


@pytest.mark.parametrize(
    "mutation",
    [
        "transcript_text",
        "transcript_time",
        "candidate_text",
        "candidate_time",
        "range",
        "duplicate",
        "malformed",
    ],
)
def test_stale_or_invalid_input_never_reaches_provider(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    transcript, _, candidates, output = inputs
    t = json.loads(transcript.read_text())
    c = json.loads(candidates.read_text())
    if mutation == "transcript_text":
        t["words"][0]["text"] = "different recording"
    elif mutation == "transcript_time":
        t["words"][0]["end_ms"] = 9999
    elif mutation == "candidate_text":
        c["candidates"][0]["words"][0]["text"] = "different recording"
    elif mutation == "candidate_time":
        c["candidates"][0]["end_ms"] = 24999
    elif mutation == "range":
        c["candidates"][0]["end_word_index"] = 999
    elif mutation == "duplicate":
        c["candidates"].append(c["candidates"][0])
    else:
        candidates.write_text("not-json")
    transcript.write_text(json.dumps(t))
    if mutation != "malformed":
        candidates.write_text(json.dumps(c))

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid input must fail before a provider request")

    monkeypatch.setattr(cli, "GroqRanker", no_provider)
    assert cli.main() == 1
    assert not output.exists()


@pytest.mark.parametrize(
    "setting",
    [
        ["--batch-size", "0"],
        ["--batch-size", "9"],
        ["--start-batch", "-1"],
        ["--max-batches", "0"],
        ["--start-batch", "99"],
    ],
)
def test_invalid_batch_settings_fail_before_provider(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch, setting: list[str]
) -> None:
    transcript, audio, candidates, output = inputs
    monkeypatch.setattr(
        sys, "argv", ["rank", str(transcript), str(audio), str(candidates), str(output), *setting]
    )

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid settings must not call provider")

    monkeypatch.setattr(cli, "GroqRanker", no_provider)
    assert cli.main() == 1
    assert not output.exists()


def test_invalid_audio_does_not_call_provider(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, audio, _, output = inputs
    audio.write_bytes(b"not a wav")

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid audio must not call provider")

    monkeypatch.setattr(cli, "GroqRanker", no_provider)
    assert cli.main() == 1
    assert not output.exists()


def test_rate_limit_is_recorded_as_partial_and_never_retried(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, output = inputs
    calls: list[int] = []

    class Limited:
        def __init__(self, api_key: str) -> None:
            pass

        def rank(self, words: Any, candidates: Any) -> GroqRankingResult:
            calls.append(1)
            raise GroqRankingError("Model provider rejected the request (HTTP 429)")

    monkeypatch.setattr(cli, "GroqRanker", Limited)
    assert cli.main() == 2
    result = json.loads(output.read_text())
    assert calls == [1]
    assert result["score_scope"] is None
    assert result["batches"][0]["error"] == "provider rate limit reached"
    assert result["batches"][0]["candidate_ids"] == ["w0_1", "w2_3"]
    assert len(result["ranking"]["missing"]) == 2


@pytest.mark.parametrize("kind", ["unknown", "duplicate", "evidence"])
def test_untrusted_assessments_fail_without_promoting_or_leaking(
    inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    _, _, _, output = inputs
    secret = "fixture-secret private-transcript"

    def invalid(candidates: Any) -> tuple[RankingAssessment, ...]:
        first = score(candidates[0])
        if kind == "unknown":
            return (RankingAssessment(secret, "suitable", 95, secret, 0, 0),)
        if kind == "duplicate":
            return (first, first)
        return (RankingAssessment(candidates[0].candidate_id, "suitable", 95, secret, 0, 3),)

    fake_ranker(monkeypatch, assessments=invalid)
    assert cli.main() == 2
    result = json.loads(output.read_text())
    assert result["batches"][0]["error"] == "invalid assessment"
    assert not result["ranking"]["short"]
    assert not result["ranking"]["long"]
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert secret not in output.read_text()


def test_multiple_provider_batches_are_marked_uncalibrated_and_identified(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, candidates, output = inputs
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rank",
            str(transcript),
            str(audio),
            str(candidates),
            str(output),
            "--batch-size",
            "1",
            "--max-batches",
            "2",
        ],
    )
    calls = fake_ranker(monkeypatch)
    assert cli.main() == 0
    result = json.loads(output.read_text())
    assert calls == [["w0_1"], ["w2_3"]]
    assert result["score_scope"] == "uncalibrated_batches"
    assert [batch["candidate_ids"] for batch in result["batches"]] == [["w0_1"], ["w2_3"]]
    assert result["batches"][0]["short_order"] == ["w0_1"]
    assert result["batches"][1]["long_order"] == ["w2_3"]
    assert result["ranking"]["responses_complete"] is True


def test_batch_local_order_does_not_claim_global_score_order(
    inputs: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, candidates, output = inputs
    data = json.loads(transcript.read_text())
    words = tuple(TranscriptWord(**word) for word in data["words"])
    clips = build_ranking_candidates(
        words,
        120_000,
        (BoundaryProposal(0, 1), BoundaryProposal(0, 0), BoundaryProposal(1, 1)),
    )
    candidates.write_text(json.dumps({"candidates": [asdict(clip.candidate) for clip in clips]}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rank",
            str(transcript),
            str(audio),
            str(candidates),
            str(output),
            "--batch-size",
            "2",
            "--max-batches",
            "2",
        ],
    )

    class FakeRanker:
        def __init__(self, api_key: str) -> None:
            pass

        def rank(self, words: Any, batch: Any) -> GroqRankingResult:
            scores = {"w0_1": 20, "w0_0": 80, "w1_1": 95}
            return GroqRankingResult(tuple(score(c, scores[c.candidate_id]) for c in batch), None)

    monkeypatch.setattr(cli, "GroqRanker", FakeRanker)
    assert cli.main() == 0
    result = json.loads(output.read_text())
    assert result["score_scope"] == "uncalibrated_batches"
    assert result["batches"][0]["short_order"] == ["w0_0", "w0_1"]
    assert result["batches"][1]["short_order"] == ["w1_1"]
    assert [x["candidate"]["candidate_id"] for x in result["ranking"]["short"]] == [
        "w0_1",
        "w0_0",
        "w1_1",
    ]


@pytest.mark.parametrize(
    ("provider_message", "safe_reason"),
    [
        (
            "Model could not return structured rankings",
            "provider could not return structured rankings",
        ),
        (
            "Candidates are too large for one ranking request",
            "candidates exceed the request size limit",
        ),
        ("fixture-secret private-transcript", "provider request or response failed"),
    ],
)
def test_provider_batch_errors_are_specific_static_and_safe(
    inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider_message: str,
    safe_reason: str,
) -> None:
    _, _, _, output = inputs
    calls: list[int] = []

    class FakeRanker:
        def __init__(self, api_key: str) -> None:
            assert api_key == "fixture-secret"

        def rank(self, words: Any, candidates: Any) -> GroqRankingResult:
            calls.append(1)
            raise GroqRankingError(provider_message)

    monkeypatch.setattr(cli, "GroqRanker", FakeRanker)
    assert cli.main() == 2
    assert calls == [1]
    result = json.loads(output.read_text())
    report = result["batches"][0]
    assert report["status"] == "failed"
    assert report["error"] == safe_reason
    assert report["candidate_ids"] == ["w0_1", "w2_3"]
    assert len(result["ranking"]["missing"]) == 2
    assert result["ranking"]["responses_complete"] is False
    assert output.stat().st_mode & 0o777 == 0o600
    captured = capsys.readouterr()
    assert "fixture-secret" not in captured.out + captured.err
    assert "private-transcript" not in captured.out + captured.err
    assert "fixture-secret" not in json.dumps(report)
    assert "private-transcript" not in json.dumps(report)
    assert "fixture-secret" not in output.read_text()
