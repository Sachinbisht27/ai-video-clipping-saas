from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import pytest

from clipforge import audio_context_cli


def test_cli_creates_preview_and_reports_source_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "preview.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x34\x12" * 16_000)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audio_context", str(source), "100", "800", str(output), "--context-ms", "500"],
    )

    assert audio_context_cli.main() == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["original_end_ms"] == 800
    assert metadata["preview_end_ms"] == 1000
    assert metadata["included_context_ms"] == 200
    assert metadata["truncated_at_eof"] is True
    with wave.open(str(output), "rb") as wav:
        assert wav.getnframes() == 14_400


def test_cli_keeps_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "preview.wav"
    output.write_bytes(b"do not replace")
    monkeypatch.setattr(
        sys, "argv", ["audio_context", str(tmp_path / "missing.wav"), "0", "500", str(output)]
    )
    assert audio_context_cli.main() == 1
    assert output.read_bytes() == b"do not replace"
