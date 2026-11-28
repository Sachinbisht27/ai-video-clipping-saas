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
    options: list[object] = []

    def export(*args: object, **kwargs: object) -> SimpleNamespace:
        received.extend(args)
        options.append(kwargs["allow_video_tail_hold"])
        return SimpleNamespace(duration_ms=3250, held_video_frames=0)

    monkeypatch.setattr(video_export_cli, "export_video_clip", export)
    monkeypatch.setattr(
        sys, "argv", ["video_export", str(video), str(audio), str(output), "100", "3350"]
    )
    assert video_export_cli.main() == 0
    assert received == [video, audio, output, 100, 3350]
    assert options == [False]
    assert "3.250s" in capsys.readouterr().out


def test_cli_reports_export_errors_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
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


def test_cli_tail_hold_is_explicit_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    options: list[object] = []

    def export(*args: object, **kwargs: object) -> SimpleNamespace:
        options.append(kwargs["allow_video_tail_hold"])
        return SimpleNamespace(
            duration_ms=10495, held_video_frames=1, video_tail_ms=15, video_frame_duration_ms=40
        )

    monkeypatch.setattr(video_export_cli, "export_video_clip", export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "video_export",
            str(tmp_path / "v.mp4"),
            str(tmp_path / "a.wav"),
            str(tmp_path / "out.mp4"),
            "0",
            "10495",
            "--allow-video-tail-hold",
        ],
    )
    assert video_export_cli.main() == 0
    assert options == [True]
    assert "Held the final video frame" in capsys.readouterr().out
