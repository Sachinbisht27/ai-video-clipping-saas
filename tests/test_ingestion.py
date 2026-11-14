"""Tests for local media ingestion, using short generated videos."""

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from clipforge.services.ingestion import (
    InvalidVideoError,
    MediaToolError,
    OutputError,
    extract_audio,
    probe_video,
)


@pytest.fixture(scope="module")
def media_tools() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required for media integration tests")


@pytest.fixture
def video(tmp_path: Path, media_tools: None) -> Path:
    # The punctuation exercises paths that must be passed as arguments, not shell code.
    path = tmp_path / "sample; video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_probe_and_extract_audio(video: Path, tmp_path: Path) -> None:
    metadata = probe_video(video)
    assert 0.9 <= metadata.duration_seconds <= 1.2
    assert metadata.size_bytes == video.stat().st_size

    output = tmp_path / "extracted; audio.wav"
    result = extract_audio(video, output)
    assert result == metadata
    with wave.open(str(output), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 16000
        assert audio.getsampwidth() == 2
        assert 14000 <= audio.getnframes() <= 18000


def test_invalid_and_missing_video_are_rejected(tmp_path: Path, media_tools: None) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    for source in (corrupt, tmp_path / "missing.mp4"):
        with pytest.raises(InvalidVideoError):
            extract_audio(source, tmp_path / "audio.wav")
        assert not (tmp_path / "audio.wav").exists()


def test_video_without_audio_is_rejected(tmp_path: Path, media_tools: None) -> None:
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=10:d=1",
            "-c:v",
            "mpeg4",
            str(silent),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(InvalidVideoError):
        probe_video(silent)


def test_existing_output_is_never_overwritten(video: Path, tmp_path: Path) -> None:
    output = tmp_path / "existing.wav"
    output.write_bytes(b"keep this file")
    with pytest.raises(OutputError):
        extract_audio(video, output)
    assert output.read_bytes() == b"keep this file"


def test_missing_ffprobe_reports_a_tool_error(
    video: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(MediaToolError):
        probe_video(video)


def test_missing_ffmpeg_leaves_no_partial_output(
    video: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None
    (bin_dir / "ffprobe").symlink_to(ffprobe)
    monkeypatch.setenv("PATH", str(bin_dir))
    output_dir = tmp_path / "audio"
    output_dir.mkdir()
    with pytest.raises(MediaToolError):
        extract_audio(video, output_dir / "audio.wav")
    assert list(output_dir.iterdir()) == []


def test_cli_produces_audio(video: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli.wav"
    result = subprocess.run(
        [sys.executable, "-m", "clipforge.cli", str(video), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()


def test_cli_reports_failure_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "missing.wav"
    result = subprocess.run(
        [sys.executable, "-m", "clipforge.cli", str(tmp_path / "missing.mp4"), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
