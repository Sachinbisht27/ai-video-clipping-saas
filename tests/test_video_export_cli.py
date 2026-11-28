from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipforge import video_export_cli
from clipforge.services.video_export import VideoExportError


def test_cli_passes_selected_range_and_reports_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    video, audio, output = (tmp_path / name for name in ("source.mp4", "audio.wav", "clip.mp4"))
    received: list[object] = []

    def export(*args: object) -> SimpleNamespace:
        received.extend(args)
        return SimpleNamespace(duration_ms=3250)

    monkeypatch.setattr(video_export_cli, "export_video_clip", export)
    monkeypatch.setattr(
        sys, "argv", ["video_export", str(video), str(audio), str(output), "100", "3350"]
    )
    assert video_export_cli.main() == 0
    assert received == [video, audio, output, 100, 3350]
    assert "3.250s" in capsys.readouterr().out


def test_cli_reports_export_errors_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: object) -> None:
        raise VideoExportError("Unsupported video timing")

    monkeypatch.setattr(video_export_cli, "export_video_clip", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "video_export",
            str(tmp_path / "v.mp4"),
            str(tmp_path / "a.wav"),
            str(tmp_path / "out.mp4"),
            "0",
            "100",
        ],
    )
    assert video_export_cli.main() == 1
    result = capsys.readouterr()
    assert "Unsupported video timing" in result.err
    assert "Traceback" not in result.err
