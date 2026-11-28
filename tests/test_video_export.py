"""Real-media tests for the local video export experiment."""

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


@pytest.fixture(scope="module")
def media_tools() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required for media integration tests")


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, check=True, capture_output=True, timeout=60)


@pytest.fixture
def marker_video(tmp_path: Path, media_tools: None) -> Path:
    """Red and 440 Hz change to blue and 880 Hz at exactly two seconds."""
    path = tmp_path / "source; marker.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:r=25:d=4,"
            "drawbox=x=0:y=0:w=iw:h=ih:color=blue:t=fill:enable='gte(t,2)'",
            "-f",
            "lavfi",
            "-i",
            r"aevalsrc=0.5*sin(2*PI*if(lt(t\,2)\,440\,880)*t):s=16000:d=4",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-shortest",
            str(path),
        ]
    )
    return path


@pytest.fixture
def marker_audio(marker_video: Path, tmp_path: Path) -> Path:
    path = tmp_path / "source; reference.wav"
    extract_audio(marker_video, path)
    return path


def _pixel(path: Path, seconds: float) -> tuple[int, int, int]:
    output = _run(
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
    ).stdout
    assert len(output) == 3
    return output[0], output[1], output[2]


def _tone(path: Path, seconds: float) -> float:
    data = _run(
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
            "0.25",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ]
    ).stdout
    samples = array("h")
    samples.frombytes(data)
    assert len(samples) > 3000
    crossings = sum(a <= 0 < b for a, b in zip(samples, samples[1:], strict=False))
    return crossings * 16000 / len(samples)


def _probe(path: Path) -> dict[str, object]:
    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,start_time,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(output.stdout)


def test_export_keeps_audio_and_video_markers_in_sync_across_non_keyframe_cut(
    marker_video: Path, marker_audio: Path, tmp_path: Path
) -> None:
    output = tmp_path / "selected; clip.mp4"
    source_bytes, audio_bytes = marker_video.read_bytes(), marker_audio.read_bytes()
    export_video_clip(marker_video, marker_audio, output, 1250, 3250)
    assert output.is_file()
    assert marker_video.read_bytes() == source_bytes
    assert marker_audio.read_bytes() == audio_bytes

    data = _probe(output)
    assert {stream["codec_type"] for stream in data["streams"]} >= {"video", "audio"}
    assert 1.90 <= float(data["format"]["duration"]) <= 2.12
    for stream in data["streams"]:
        if stream["codec_type"] in {"audio", "video"}:
            assert abs(float(stream["start_time"])) <= 0.08
            assert 1.90 <= float(stream["duration"]) <= 2.12
    # Both modalities change near 0.75 s in the output; positions on either
    # side independently catch A/V drift and keyframe-only seeking.
    before, after = _pixel(output, 0.64), _pixel(output, 0.86)
    assert before[0] > before[2] * 2
    assert after[2] > after[0] * 2
    assert 400 <= _tone(output, 0.25) <= 480
    assert 820 <= _tone(output, 1.20) <= 940
    assert 400 <= _tone(output, 0.40) <= 480
    assert 820 <= _tone(output, 0.90) <= 940


@pytest.mark.parametrize("start,end", [(-1, 1000), (1000, 1000), (2000, 1000), (0, 4001)])
def test_rejects_invalid_ranges_without_leaving_partial_output(
    marker_video: Path, marker_audio: Path, tmp_path: Path, start: int, end: int
) -> None:
    output = tmp_path / "rejected.mp4"
    with pytest.raises(VideoExportError):
        export_video_clip(marker_video, marker_audio, output, start, end)
    assert not output.exists()


def test_rejects_modified_reference_wav(
    marker_video: Path, marker_audio: Path, tmp_path: Path
) -> None:
    wrong = tmp_path / "wrong.wav"
    data = bytearray(marker_audio.read_bytes())
    data[-10] ^= 0x7F
    wrong.write_bytes(data)
    output = tmp_path / "rejected.mp4"
    with pytest.raises(VideoExportError):
        export_video_clip(marker_video, wrong, output, 500, 2500)
    assert not output.exists()


@pytest.mark.parametrize("dangling", [False, True])
def test_refuses_existing_output_or_symlink(
    marker_video: Path, marker_audio: Path, tmp_path: Path, dangling: bool
) -> None:
    output = tmp_path / "existing.mp4"
    if dangling:
        output.symlink_to(tmp_path / "nonexistent.mp4")
    else:
        output.write_bytes(b"keep existing")
    with pytest.raises(VideoExportError):
        export_video_clip(marker_video, marker_audio, output, 500, 2500)
    if dangling:
        assert output.is_symlink()
        assert not (tmp_path / "nonexistent.mp4").exists()
    else:
        assert output.read_bytes() == b"keep existing"


def test_rejects_nonzero_audio_stream_offset(tmp_path: Path, media_tools: None) -> None:
    video = tmp_path / "offset.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=25:d=4",
            "-itsoffset",
            "0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=3",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    reference = tmp_path / "offset.wav"
    extract_audio(video, reference)
    output = tmp_path / "rejected.mp4"
    with pytest.raises(VideoExportError):
        export_video_clip(video, reference, output, 1000, 2000)
    assert not output.exists()


def test_rejects_discontinuous_audio_timestamps(tmp_path: Path, media_tools: None) -> None:
    video = tmp_path / "discontinuous.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=25:d=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=3.5",
            "-af",
            "asetpts='PTS+if(gte(T,1.5),0.5/TB,0)'",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    # Confirm this FFmpeg build preserved the intended mid-stream timestamp gap.
    raw = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(video),
        ]
    ).stdout.decode()
    timestamps = [float(line.strip().split(",")[0]) for line in raw.splitlines() if line.strip()]
    assert any(b - a > 0.3 for a, b in zip(timestamps, timestamps[1:], strict=False))
    reference, output = tmp_path / "discontinuous.wav", tmp_path / "rejected.mp4"
    extract_audio(video, reference)
    with pytest.raises(VideoExportError):
        export_video_clip(video, reference, output, 1000, 3000)
    assert not output.exists()


def test_rejects_variable_or_discontinuous_video_timestamps(
    tmp_path: Path, media_tools: None
) -> None:
    video = tmp_path / "variable-frames.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=25:d=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=4",
            "-vf",
            "setpts='if(gte(N,50),PTS+0.12/TB,PTS)'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    reference, output = tmp_path / "variable-frames.wav", tmp_path / "rejected.mp4"
    extract_audio(video, reference)
    with pytest.raises(VideoExportError):
        export_video_clip(video, reference, output, 1000, 3000)
    assert not output.exists()


def test_rejects_video_with_multiple_audio_streams(tmp_path: Path, media_tools: None) -> None:
    video = tmp_path / "two-audio-streams.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=25:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=16000:duration=2",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:a:0",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    reference, output = tmp_path / "two-audio.wav", tmp_path / "rejected.mp4"
    extract_audio(video, reference)
    with pytest.raises(VideoExportError):
        export_video_clip(video, reference, output, 250, 1750)
    assert not output.exists()


def test_invalid_video_and_audio_are_rejected(tmp_path: Path, media_tools: None) -> None:
    video, audio, output = tmp_path / "bad.mp4", tmp_path / "bad.wav", tmp_path / "out.mp4"
    video.write_bytes(b"not video")
    audio.write_bytes(b"not audio")
    with pytest.raises(VideoExportError):
        export_video_clip(video, audio, output, 0, 1000)
    assert not output.exists()


def test_rejects_cut_past_video_even_when_audio_continues(
    tmp_path: Path, media_tools: None
) -> None:
    video = tmp_path / "shorter-video.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=25:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=4",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    reference, output = tmp_path / "source.wav", tmp_path / "rejected.mp4"
    extract_audio(video, reference)
    with pytest.raises(VideoExportError):
        export_video_clip(video, reference, output, 1800, 2200)
    assert not output.exists()


def test_missing_encoder_leaves_no_partial_output(
    marker_video: Path, marker_audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None
    (binary_dir / "ffprobe").symlink_to(ffprobe)
    monkeypatch.setenv("PATH", str(binary_dir))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(VideoExportError):
        export_video_clip(marker_video, marker_audio, output_dir / "result.mp4", 500, 2500)
    assert not list(output_dir.iterdir())


def test_source_cannot_be_used_as_output(marker_video: Path, marker_audio: Path) -> None:
    before = marker_video.read_bytes()
    with pytest.raises(VideoExportError):
        export_video_clip(marker_video, marker_audio, marker_video, 500, 2500)
    assert marker_video.read_bytes() == before


@pytest.mark.parametrize("video_duration,audio_duration", [(0.4, 0.4), (0.4, 2.0), (2.0, 0.4)])
def test_rejects_successfully_encoded_but_incomplete_output(
    marker_video: Path,
    marker_audio: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_duration: float,
    audio_duration: float,
) -> None:
    # This MP4 is readable and has both streams, but at least one is too short.
    incomplete = tmp_path / "incomplete.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=64x64:r=25:d={video_duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=16000:duration={audio_duration}",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(incomplete),
        ]
    )
    original_run = video_export_service._run_tool

    def fake_encoder(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        if args[0] == "ffmpeg":
            shutil.copyfile(incomplete, args[-1])
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return original_run(args, timeout=timeout)

    monkeypatch.setattr(video_export_service, "_run_tool", fake_encoder)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(VideoExportError):
        export_video_clip(marker_video, marker_audio, output_dir / "result.mp4", 1250, 3250)
    assert not list(output_dir.iterdir())


def test_rejects_output_that_fails_decode_before_publication(
    marker_video: Path, marker_audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_run = video_export_service._run_tool
    decode_checked = False

    def fail_decode(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        nonlocal decode_checked
        if args[0] == "ffmpeg" and "-xerror" in args:
            decode_checked = True
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="decode failed")
        return original_run(args, timeout=timeout)

    monkeypatch.setattr(video_export_service, "_run_tool", fail_decode)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(VideoExportError, match="cannot be decoded"):
        export_video_clip(marker_video, marker_audio, output_dir / "result.mp4", 1250, 3250)
    assert decode_checked
    assert not list(output_dir.iterdir())
