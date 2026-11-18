"""Local candidate CLI behavior; no provider requests are made."""

import json
import sys
import wave
from pathlib import Path

import pytest

from clipforge import candidates_cli


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(100)
        wav.writeframes(b"\x00\x00" * 6100)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "Hello", "start_ms": 0, "end_ms": 10000, "speaker": 0},
                    {"text": "there.", "start_ms": 10000, "end_ms": 21000, "speaker": 1},
                ]
            }
        )
    )
    return transcript, audio, tmp_path / "candidates.json"


def test_cli_writes_private_candidate_with_speaker_words(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, output = inputs
    monkeypatch.setattr(sys, "argv", ["candidates", str(transcript), str(audio), str(output)])
    assert candidates_cli.main() == 0
    result = json.loads(output.read_text())
    assert result["audio_duration_ms"] == 61000
    assert len(result["candidates"]) == 1
    assert [word["speaker"] for word in result["candidates"][0]["words"]] == [0, 1]
    assert output.stat().st_mode & 0o777 == 0o600


def test_cli_refuses_existing_file_and_symlink(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, output = inputs
    output.write_text("keep me")
    monkeypatch.setattr(sys, "argv", ["candidates", str(transcript), str(audio), str(output)])
    assert candidates_cli.main() == 1
    assert output.read_text() == "keep me"
    output.unlink()
    output.symlink_to(output.parent / "missing")
    assert candidates_cli.main() == 1
    assert output.is_symlink()


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"words": [{"text": "x", "start_ms": true, "end_ms": 1}]}',
        '{"words": [{"text": "x", "start_ms": 0, "end_ms": 61001}]}',
    ],
)
def test_cli_rejects_invalid_transcripts_without_creating_output(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    transcript, audio, output = inputs
    transcript.write_text(content)
    monkeypatch.setattr(sys, "argv", ["candidates", str(transcript), str(audio), str(output)])
    assert candidates_cli.main() == 1
    assert not output.exists()


def test_cli_rejects_invalid_audio_without_creating_output(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, audio, output = inputs
    audio.write_bytes(b"not a wav")
    monkeypatch.setattr(sys, "argv", ["candidates", str(transcript), str(audio), str(output)])
    assert candidates_cli.main() == 1
    assert not output.exists()
