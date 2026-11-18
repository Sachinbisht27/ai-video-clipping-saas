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
            assert model == cli._MODEL

        def propose(self, words: Any) -> BoundaryResult:
            assert [w.text for w in words] == ["Hello.", "Answer."]
            return BoundaryResult((BoundaryProposal(0, 1),), ModelUsage(100, 25, 5))

    monkeypatch.setattr(cli, "OpenAIBoundaryProposer", FakeProposer)
    assert cli.main() == 0
    result = json.loads(output.read_text())
    assert result["model"] == cli._MODEL
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
