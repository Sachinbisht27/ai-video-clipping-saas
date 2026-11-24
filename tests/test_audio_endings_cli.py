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


@pytest.mark.parametrize(
    "transcript_range, expected", [(None, None), ((2000, 2300), True), ((0, 100), False)]
)
def test_cli_reports_transcript_overlap_without_guessing_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transcript_range: tuple[int, int] | None,
    expected: bool | None,
) -> None:
    audio = tmp_path / "source.wav"
    output = tmp_path / "review.json"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x70\x17" * 33_600 + b"\0\0" * 1_600 + b"\x70\x17" * 12_800)
    argv = ["audio_endings", str(audio), "0", "2500", str(output)]
    if transcript_range is not None:
        transcript = tmp_path / "transcript.json"
        transcript.write_text(
            json.dumps(
                {
                    "words": [
                        {
                            "text": "word",
                            "start_ms": transcript_range[0],
                            "end_ms": transcript_range[1],
                        }
                    ]
                }
            )
        )
        argv.extend(["--transcript", str(transcript)])
    monkeypatch.setattr(sys, "argv", argv)

    assert audio_endings_cli.main() == 0
    suggestions = json.loads(output.read_text())["suggestions"]
    assert suggestions
    assert all(item["overlaps_transcript_word"] is expected for item in suggestions)
    assert all(
        item["removed_transcript_words"] == (None if transcript_range is None else 0)
        for item in suggestions
    )


def test_cli_warns_when_an_earlier_ending_removes_a_whole_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "source.wav"
    transcript = tmp_path / "transcript.json"
    output = tmp_path / "review.json"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x70\x17" * 33_600 + b"\0\0" * 1_600 + b"\x70\x17" * 12_800)
    transcript.write_text(
        json.dumps(
            {
                "words": [
                    {"text": "first", "start_ms": 2000, "end_ms": 2200},
                    {"text": "last", "start_ms": 2300, "end_ms": 2400},
                ]
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["audio_endings", str(audio), "0", "2500", str(output), "--transcript", str(transcript)],
    )

    assert audio_endings_cli.main() == 0
    suggestions = json.loads(output.read_text())["suggestions"]
    assert len(suggestions) == 1
    assert suggestions[0]["overlaps_transcript_word"] is True
    assert suggestions[0]["removed_transcript_words"] == 1


def test_cli_rejects_transcript_timings_beyond_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio, transcript, output = (
        tmp_path / name for name in ("audio.wav", "words.json", "out.json")
    )
    make_silent_wav(audio)
    transcript.write_text(json.dumps({"words": [{"text": "word", "start_ms": 0, "end_ms": 4000}]}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["audio_endings", str(audio), "0", "2500", str(output), "--transcript", str(transcript)],
    )
    assert audio_endings_cli.main() == 1
    assert not output.exists()
