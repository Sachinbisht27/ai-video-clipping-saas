"""The experimental CLI keeps all tests local and uses a fake proposer."""

import json
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

from clipforge import semantic_candidates_cli as cli
from clipforge.services.openai_boundaries import BoundaryResult, ModelUsage
from clipforge.services.semantic_candidates import BoundaryProposal


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=fixture-secret\n")
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(100)
        wav.writeframes(b"\x00\x00" * 6500)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "Hello.", "start_ms": 0, "end_ms": 1000, "speaker": 0},
                    {"text": "Answer.", "start_ms": 1000, "end_ms": 25000, "speaker": 1},
                ]
            }
        )
    )
    output = tmp_path / "candidates.json"
    monkeypatch.setattr(sys, "argv", ["semantic", str(transcript), str(audio), str(output)])
    return transcript, audio, output


def test_cli_sends_only_after_local_validation_and_writes_private_file(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output = inputs

    class FakeProposer:
        def __init__(self, api_key: str, *, model: str) -> None:
            assert api_key == "fixture-secret"
            assert model == cli._MODELS["openai"]

        def propose(self, words: Any) -> BoundaryResult:
            assert [w.text for w in words] == ["Hello.", "Answer."]
            return BoundaryResult((BoundaryProposal(0, 1),), ModelUsage(100, 25, 5))

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", FakeProposer)
    assert cli.main() == 0
    result = json.loads(output.read_text())
    assert result["model"] == cli._MODELS["openai"]
    assert result["audio_duration_ms"] == 65000
    assert result["usage"] == {"input_tokens": 100, "output_tokens": 25, "reasoning_tokens": 5}
    assert result["candidates"][0]["end_ms"] == 25000
    assert [w["speaker"] for w in result["candidates"][0]["words"]] == [0, 1]
    assert output.stat().st_mode & 0o777 == 0o600
    assert "fixture-secret" not in output.read_text()


def test_cli_refuses_existing_output_and_symlink(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output = inputs

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Existing output must be rejected before provider call")

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", no_provider)
    output.write_text("keep me")
    assert cli.main() == 1
    assert output.read_text() == "keep me"
    output.unlink()
    output.symlink_to(output.parent / "missing")
    assert cli.main() == 1
    assert output.is_symlink()


@pytest.mark.parametrize(
    "invalid",
    [
        "not json",
        '{"words":[{"text":"x","start_ms":true,"end_ms":1}]}',
        '{"words":[{"text":"x","start_ms":0,"end_ms":65001}]}',
    ],
)
def test_bad_transcript_never_reaches_provider(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    transcript, _, output = inputs
    transcript.write_text(invalid)

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid input must never reach provider")

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", no_provider)
    assert cli.main() == 1
    assert not output.exists()


def test_bad_audio_never_reaches_provider(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, audio, output = inputs
    audio.write_bytes(b"bad wav")

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid audio must never reach provider")

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", no_provider)
    assert cli.main() == 1
    assert not output.exists()


def test_empty_transcript_needs_no_provider_or_key(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, _, output = inputs
    transcript.write_text('{"words": []}')
    (transcript.parent / ".env").unlink()

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("An empty transcript must not reach provider")

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", no_provider)
    assert cli.main() == 0
    result = json.loads(output.read_text())
    assert result["candidates"] == []
    assert result["model"] is None


def test_missing_key_fails_without_request_or_output(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, _, output = inputs
    (transcript.parent / ".env").unlink()

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Missing credentials must fail before provider call")

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", no_provider)
    assert cli.main() == 1
    assert not output.exists()


def test_invalid_provider_boundaries_fail_without_output(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, output = inputs

    class FakeProposer:
        def __init__(self, api_key: str, *, model: str) -> None:
            pass

        def propose(self, words: Any) -> BoundaryResult:
            return BoundaryResult((BoundaryProposal(0, 999),), None)

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", FakeProposer)
    assert cli.main() == 1
    assert not output.exists()


def test_provider_failure_does_not_expose_secret_or_private_text(
    inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, output = inputs

    class FakeProposer:
        def __init__(self, api_key: str, *, model: str) -> None:
            pass

        def propose(self, words: Any) -> BoundaryResult:
            raise cli.OpenAIBoundaryError("fixture-secret private text")

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", FakeProposer)
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert "fixture-secret" not in captured.out + captured.err
    assert "private text" not in captured.out + captured.err
    assert not output.exists()


def test_groq_requires_explicit_selection_and_uses_its_own_key(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, output = inputs
    (transcript.parent / ".env").write_text(
        "OPENAI_API_KEY=fixture-openai\nGROQ_API_KEY=fixture-groq\n"
    )
    monkeypatch.setattr(
        sys, "argv", ["semantic", str(transcript), str(audio), str(output), "--provider", "groq"]
    )

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            assert api_key == "fixture-groq"

        def propose(self, words: Any) -> BoundaryResult:
            return BoundaryResult((BoundaryProposal(0, 1),), ModelUsage(50, 10))

    def no_openai(*args: object, **kwargs: object) -> None:
        pytest.fail("Groq selection must not send data to OpenAI")

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", no_openai)
    monkeypatch.setattr(cli, "GroqBoundaryProposer", FakeGroq)
    assert cli.main() == 0
    result = json.loads(output.read_text())
    assert result["provider"] == "groq"
    assert result["model"] == "openai/gpt-oss-120b"
    assert result["candidates"][0]["end_ms"] == 25000
    assert "fixture-groq" not in output.read_text()


def test_groq_missing_key_does_not_fall_back_to_openai(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, output = inputs
    monkeypatch.setattr(
        sys, "argv", ["semantic", str(transcript), str(audio), str(output), "--provider", "groq"]
    )

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Missing Groq credentials must not result in a provider request")

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", no_provider)
    monkeypatch.setattr(cli, "GroqBoundaryProposer", no_provider)
    assert cli.main() == 1
    assert not output.exists()


@pytest.mark.parametrize(
    ("provider_error", "expected"),
    [
        ("Transcript is too large for one model request", "shorter excerpt"),
        ("Model provider rejected the request (HTTP 429)", "account quota"),
    ],
)
def test_groq_explains_local_size_and_rate_limits(
    inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider_error: str,
    expected: str,
) -> None:
    transcript, audio, output = inputs
    (transcript.parent / ".env").write_text("GROQ_API_KEY=fixture-groq\n")
    monkeypatch.setattr(
        sys, "argv", ["semantic", str(transcript), str(audio), str(output), "--provider", "groq"]
    )

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            pass

        def propose(self, words: Any) -> BoundaryResult:
            raise cli.GroqBoundaryError(provider_error)

    monkeypatch.setattr(cli, "GroqBoundaryProposer", FakeGroq)
    assert cli.main() == 1
    assert expected in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("length", "target_min", "target_max", "status"),
    [
        ("short", 20_000, 60_000, "within_target"),
        ("long", 60_000, 180_000, "shorter_than_target"),
        ("both", 20_000, 180_000, "within_target"),
    ],
)
def test_windowed_cli_forwards_length_and_returns_validated_result(
    inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    length: str,
    target_min: int,
    target_max: int,
    status: str,
) -> None:
    transcript, audio, output = inputs
    (transcript.parent / ".env").write_text("GROQ_API_KEY=fixture-groq\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semantic",
            str(transcript),
            str(audio),
            str(output),
            "--provider",
            "groq",
            "--windowed",
            "--length",
            length,
        ],
    )
    seen: list[tuple[int, str, str]] = []

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            assert api_key == "fixture-groq"

        def propose(
            self,
            words: Any,
            *,
            first_word_index: int = 0,
            duration_mode: str = "both",
            input_format: str = "segments",
        ) -> BoundaryResult:
            seen.append((first_word_index, duration_mode, input_format))
            return BoundaryResult(
                (BoundaryProposal(first_word_index, first_word_index + len(words) - 1),),
                ModelUsage(10, 20),
            )

    monkeypatch.setattr(cli, "GroqBoundaryProposer", FakeGroq)
    assert cli.main() == 0
    result = json.loads(output.read_text())
    assert seen == [(0, length, "segments")]
    assert result["duration_mode"] == length
    assert (result["target_min_ms"], result["target_max_ms"]) == (target_min, target_max)
    assert result["candidates"][0]["duration_status"] == status
    assert result["windowing"]["processing_complete"] is True
    assert output.stat().st_mode & 0o777 == 0o600


def test_windowed_cli_reports_partial_and_preserves_output(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, output = inputs
    (transcript.parent / ".env").write_text("GROQ_API_KEY=fixture-groq\n")
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(100)
        wav.writeframes(b"\x00\x00" * 50_000)
    transcript.write_text(
        json.dumps(
            {
                "words": [
                    {"text": f"word-{i}", "start_ms": t, "end_ms": t + 1000}
                    for i, t in enumerate((0, 100_000, 200_000, 300_000, 400_000, 490_000))
                ]
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semantic",
            str(transcript),
            str(audio),
            str(output),
            "--provider",
            "groq",
            "--windowed",
            "--max-windows",
            "1",
        ],
    )
    seen: list[int] = []

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            pass

        def propose(
            self,
            words: Any,
            *,
            first_word_index: int = 0,
            duration_mode: str = "both",
            input_format: str = "segments",
        ) -> BoundaryResult:
            seen.append(first_word_index)
            return BoundaryResult((BoundaryProposal(first_word_index, first_word_index),), None)

    monkeypatch.setattr(cli, "GroqBoundaryProposer", FakeGroq)
    assert cli.main() == 2
    result = json.loads(output.read_text())
    assert seen == [0]
    assert result["windowing"]["total_windows"] > 1
    assert result["windowing"]["next_window_index"] == 1
    assert not result["windowing"]["processing_complete"]
    assert result["candidates"]


def test_windowed_groq_failure_stops_without_openai_fallback(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, output = inputs
    (transcript.parent / ".env").write_text(
        "GROQ_API_KEY=fixture-groq\nOPENAI_API_KEY=fixture-openai\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["semantic", str(transcript), str(audio), str(output), "--provider", "groq", "--windowed"],
    )
    seen: list[int] = []

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            pass

        def propose(
            self,
            words: Any,
            *,
            first_word_index: int = 0,
            duration_mode: str = "both",
            input_format: str = "segments",
        ) -> BoundaryResult:
            seen.append(first_word_index)
            raise cli.GroqBoundaryError("Model provider rejected the request (HTTP 429)")

    def no_openai(*args: object, **kwargs: object) -> None:
        pytest.fail("Groq failure must not trigger OpenAI fallback")

    monkeypatch.setattr(cli, "GroqBoundaryProposer", FakeGroq)
    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", no_openai)
    assert cli.main() == 2
    result = json.loads(output.read_text())
    assert seen == [0]
    assert result["windowing"]["windows"][0]["error"] == "provider rate limit reached"
    assert result["windowing"]["next_window_index"] == 0
    assert not result["windowing"]["processing_complete"]
    assert result["candidates"] == []


@pytest.mark.parametrize(
    "args",
    [
        ["--provider", "openai", "--windowed"],
        ["--provider", "openai", "--length", "long"],
        ["--provider", "groq", "--start-window", "1"],
        ["--provider", "groq", "--windowed", "--start-window", "-1"],
        ["--provider", "groq", "--windowed", "--max-windows", "0"],
    ],
)
def test_invalid_window_config_does_not_call_provider(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    transcript, audio, output = inputs
    monkeypatch.setattr(sys, "argv", ["semantic", str(transcript), str(audio), str(output), *args])

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid options must not contact any provider")

    monkeypatch.setattr(cli, "GroqBoundaryProposer", no_provider)
    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", no_provider)
    assert cli.main() == 1
    assert not output.exists()


def test_windowed_empty_transcript_requires_no_key_or_request(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, output = inputs
    transcript.write_text('{"words": []}')
    (transcript.parent / ".env").unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        ["semantic", str(transcript), str(audio), str(output), "--provider", "groq", "--windowed"],
    )

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("An empty transcript must not contact any provider")

    monkeypatch.setattr(cli, "GroqBoundaryProposer", no_provider)
    assert cli.main() == 0
    assert json.loads(output.read_text())["candidates"] == []


def test_combined_mode_marks_70_second_suggestion_within_preferred_range(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, output = inputs
    (transcript.parent / ".env").write_text("GROQ_API_KEY=fixture-groq\n")
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(100)
        wav.writeframes(b"\x00\x00" * 10_000)
    transcript.write_text(
        json.dumps({"words": [{"text": "Complete?", "start_ms": 0, "end_ms": 70_000}]})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semantic",
            str(transcript),
            str(audio),
            str(output),
            "--provider",
            "groq",
            "--windowed",
            "--length",
            "both",
        ],
    )

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            pass

        def propose(
            self,
            words: Any,
            *,
            first_word_index: int = 0,
            duration_mode: str = "both",
            input_format: str = "segments",
        ) -> BoundaryResult:
            return BoundaryResult((BoundaryProposal(first_word_index, first_word_index),), None)

    monkeypatch.setattr(cli, "GroqBoundaryProposer", FakeGroq)
    assert cli.main() == 0
    result = json.loads(output.read_text())
    assert (result["target_min_ms"], result["target_max_ms"]) == (20_000, 180_000)
    assert result["candidates"][0]["duration_status"] == "within_target"


def test_nonwindowed_groq_preserves_valid_suggestions_and_reports_rejections(
    inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript, audio, output = inputs
    (transcript.parent / ".env").write_text("GROQ_API_KEY=fixture-groq-secret\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semantic",
            str(transcript),
            str(audio),
            str(output),
            "--provider",
            "groq",
            "--length",
            "long",
        ],
    )

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            assert api_key == "fixture-groq-secret"

        def propose(
            self, words: Any, *, duration_mode: str = "both", input_format: str = "segments"
        ) -> BoundaryResult:
            assert duration_mode == "long"
            assert input_format == "segments"
            return BoundaryResult((BoundaryProposal(0, 1),), ModelUsage(10, 10), 2)

    monkeypatch.setattr(cli, "GroqBoundaryProposer", FakeGroq)
    assert cli.main() == 2
    data = json.loads(output.read_text())
    assert data["rejected_proposals"] == 2
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["end_ms"] == 25_000
    assert data["windowing"] is None
    assert output.stat().st_mode & 0o777 == 0o600
    captured = capsys.readouterr()
    assert "fixture-groq-secret" not in captured.out + captured.err
    assert "Hello." not in captured.out + captured.err
    assert "proposals were rejected" in captured.err
