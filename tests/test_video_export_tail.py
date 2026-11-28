"""Exercise an explicit, bounded video-tail hold on real generated media."""

from __future__ import annotations

import json
import shutil
import subprocess
from array import array
from pathlib import Path

import pytest

from clipforge.services import video_export as video_export_service
from clipforge.services.ingestion import extract_audio
from clipforge.services.video_export import VideoExportError, export_video_clip


def run_tool(args: list[str]) -> bytes:
    return subprocess.run(args, check=True, capture_output=True, timeout=60).stdout


@pytest.fixture
def tail_media(tmp_path: Path) -> tuple[Path, Path]:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required for media integration tests")
    video = tmp_path / "source.mp4"
    run_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:r=25:d=2,"
            "drawbox=x=0:y=0:w=iw:h=ih:color=blue:t=fill:enable='gte(t,1)',"
            "drawbox=x=0:y=0:w=iw:h=ih:color=green:t=fill:enable='gte(t,1.96)'",
            "-f",
            "lavfi",
            "-i",
            r"aevalsrc=0.5*sin(2*PI*if(lt(t\,2)\,440\,880)*t):s=16000:d=2.3",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(video),
        ]
    )
    audio = tmp_path / "reference.wav"
    extract_audio(video, audio)
    return video, audio


def pixel(path: Path, seconds: float) -> tuple[int, int, int]:
    raw = run_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(seconds),
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            "scale=1:1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    assert len(raw) == 3
    return raw[0], raw[1], raw[2]


def audio_frequency(path: Path, seconds: float, duration: float = 0.08) -> float:
    raw = run_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(seconds),
            "-i",
            str(path),
            "-t",
            str(duration),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ]
    )
    samples = array("h")
    samples.frombytes(raw)
    assert len(samples) >= 1000
    crossings = sum(a <= 0 < b for a, b in zip(samples, samples[1:], strict=False))
    return crossings * 16000 / len(samples)


def test_default_still_rejects_audio_after_video_end(
    tail_media: tuple[Path, Path], tmp_path: Path
) -> None:
    source, audio = tail_media
    output = tmp_path / "rejected.mp4"
    with pytest.raises(VideoExportError):
        export_video_clip(source, audio, output, 0, 2200)
    assert not output.exists()


def test_opted_in_hold_keeps_last_original_frame_and_full_audio_tail(
    tail_media: tuple[Path, Path], tmp_path: Path
) -> None:
    source, audio = tail_media
    source_bytes, audio_bytes = source.read_bytes(), audio.read_bytes()
    output = tmp_path / "held.mp4"
    result = export_video_clip(source, audio, output, 0, 2200, allow_video_tail_hold=True)

    assert result.video_tail_ms == pytest.approx(200, abs=1)
    assert result.held_video_frames >= 5
    assert result.video_frame_duration_ms == pytest.approx(40, abs=0.01)
    assert source.read_bytes() == source_bytes
    assert audio.read_bytes() == audio_bytes

    # Blue before the source's distinct last green frame ensures the exporter
    # doesn't freeze an earlier scene; the held frames must match the source end.
    before = pixel(output, 1.88)
    original_final = pixel(source, 1.96)
    held_final = pixel(output, 2.12)
    assert before[2] > before[1] * 2
    assert original_final[1] > original_final[2] * 2
    assert held_final[1] > held_final[2] * 2
    assert all(abs(a - b) <= 12 for a, b in zip(original_final, held_final, strict=True))

    # The audible tone changes after the picture ends. This checks that the
    # requested audio tail survives instead of being silently shortened.
    assert 400 <= audio_frequency(output, 1.7) <= 480
    assert 820 <= audio_frequency(output, 2.05) <= 940
    probe = json.loads(
        run_tool(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,start_time,duration",
                "-of",
                "json",
                str(output),
            ]
        )
    )
    tracks = {stream["codec_type"]: stream for stream in probe["streams"]}
    assert float(tracks["audio"]["duration"]) >= 2.19
    assert float(tracks["video"]["duration"]) >= 2.19


def test_hold_cap_is_inclusive_and_exceeding_it_is_rejected(
    tail_media: tuple[Path, Path], tmp_path: Path
) -> None:
    source, audio = tail_media
    output = tmp_path / "cap.mp4"
    result = export_video_clip(source, audio, output, 0, 2250, allow_video_tail_hold=True)
    assert result.video_tail_ms == pytest.approx(250, abs=1)
    assert output.is_file()

    rejected = tmp_path / "over-cap.mp4"
    with pytest.raises(VideoExportError):
        export_video_clip(source, audio, rejected, 0, 2251, allow_video_tail_hold=True)
    assert not rejected.exists()


def test_opt_in_does_not_hold_frames_for_a_range_inside_video(
    tail_media: tuple[Path, Path], tmp_path: Path
) -> None:
    source, audio = tail_media
    output = tmp_path / "inside.mp4"
    result = export_video_clip(source, audio, output, 250, 1500, allow_video_tail_hold=True)
    assert result.video_tail_ms == 0
    assert result.held_video_frames == 0
    assert output.is_file()


def test_hold_may_start_at_the_last_real_frame(
    tail_media: tuple[Path, Path], tmp_path: Path
) -> None:
    source, audio = tail_media
    output = tmp_path / "last-frame.mp4"
    result = export_video_clip(source, audio, output, 1960, 2100, allow_video_tail_hold=True)
    assert result.video_tail_ms == pytest.approx(100, abs=1)
    assert result.held_video_frames >= 2
    held = pixel(output, 0.08)
    assert held[1] > held[2] * 2


@pytest.mark.parametrize("start,end", [(1961, 2100), (2000, 2100), (0, 3000)])
def test_hold_rejects_missing_video_frame_or_out_of_audio(
    tail_media: tuple[Path, Path], tmp_path: Path, start: int, end: int
) -> None:
    source, audio = tail_media
    output = tmp_path / "rejected.mp4"
    with pytest.raises(VideoExportError):
        export_video_clip(source, audio, output, start, end, allow_video_tail_hold=True)
    assert not output.exists()


@pytest.mark.parametrize("dangling", [False, True])
def test_hold_never_overwrites_existing_output_or_symlink(
    tail_media: tuple[Path, Path], tmp_path: Path, dangling: bool
) -> None:
    source, audio = tail_media
    output = tmp_path / "existing.mp4"
    if dangling:
        output.symlink_to(tmp_path / "not-created.mp4")
    else:
        output.write_bytes(b"keep this file")
    with pytest.raises(VideoExportError):
        export_video_clip(source, audio, output, 0, 2200, allow_video_tail_hold=True)
    if dangling:
        assert output.is_symlink()
        assert not (tmp_path / "not-created.mp4").exists()
    else:
        assert output.read_bytes() == b"keep this file"


def test_hold_rejects_non_boolean_option(tail_media: tuple[Path, Path], tmp_path: Path) -> None:
    source, audio = tail_media
    output = tmp_path / "rejected.mp4"
    with pytest.raises(VideoExportError):
        export_video_clip(source, audio, output, 0, 2200, allow_video_tail_hold=1)  # type: ignore[arg-type]
    assert not output.exists()


def test_hold_rejects_nonzero_audio_offset(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required for media integration tests")
    source = tmp_path / "offset.mp4"
    run_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=25:d=2",
            "-itsoffset",
            "0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1.8",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    audio, output = tmp_path / "offset.wav", tmp_path / "rejected.mp4"
    extract_audio(source, audio)
    with pytest.raises(VideoExportError, match="zero-aligned"):
        export_video_clip(source, audio, output, 0, 2100, allow_video_tail_hold=True)
    assert not output.exists()


def test_failed_hold_encoder_cleans_up_staged_output(
    tail_media: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, audio = tail_media
    original_run = video_export_service._run_tool

    def fail_encoder(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        if args[0] == "ffmpeg" and "-c:v" in args:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="encoder failed")
        return original_run(args, timeout=timeout)

    monkeypatch.setattr(video_export_service, "_run_tool", fail_encoder)
    folder = tmp_path / "output"
    folder.mkdir()
    with pytest.raises(VideoExportError):
        export_video_clip(source, audio, folder / "result.mp4", 0, 2200, allow_video_tail_hold=True)
    assert not list(folder.iterdir())
