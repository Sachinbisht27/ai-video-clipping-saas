from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

import pytest

from clipforge import audio_endings_cli


def make_silent_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 48_000)


def test_cli_keeps_source_and_writes_private_unverified_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "source.wav"
    result = tmp_path / "review.json"
    make_silent_wav(audio)
    original = audio.read_bytes()
    monkeypatch.setattr(sys, "argv", ["audio_endings", str(audio), "0", "2500", str(result)])

    assert audio_endings_cli.main() == 0
    data = json.loads(result.read_text())
    assert data["start_ms"] == 0
    assert data["end_ms"] == 2500
    assert data["status"] == "uncertain"
    assert data["suggestions"] == []
    assert audio.read_bytes() == original
    assert os.stat(result).st_mode & 0o777 == 0o600


def test_cli_refuses_to_overwrite_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "source.wav"
    output = tmp_path / "review.json"
    make_silent_wav(audio)
    output.write_text("keep me")
    monkeypatch.setattr(sys, "argv", ["audio_endings", str(audio), "0", "2500", str(output)])

    assert audio_endings_cli.main() == 1
    assert output.read_text() == "keep me"


def test_cli_rejects_bad_transcript_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "source.wav"
    transcript = tmp_path / "transcript.json"
    output = tmp_path / "review.json"
    make_silent_wav(audio)
    transcript.write_text('{"words": [{"start_ms": "wrong"}]}')
    monkeypatch.setattr(
        sys,
        "argv",
        ["audio_endings", str(audio), "0", "2500", str(output), "--transcript", str(transcript)],
    )

    assert audio_endings_cli.main() == 1
    assert not output.exists()
