"""Safety and exact-copy tests for local context previews.

Synthetic PCM checks audio boundaries, not recognition of speech or laughter.
"""

from __future__ import annotations

import os
import struct
import wave
from pathlib import Path

import pytest

from clipforge.services.audio_context import AudioContextError, create_context_preview

RATE = 16_000


def make_wav(path: Path, milliseconds: int) -> None:
    """Write a repeating sequence so copying the wrong interval is observable."""
    pattern = b"".join(struct.pack("<h", value) for value in range(-50, 50))
    frames = milliseconds * RATE // 1000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes(pattern * (frames // 100) + pattern[: (frames % 100) * 2])


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "source.wav"
    make_wav(path, 8000)
    return path


def read_interval(path: Path, start_ms: int, end_ms: int) -> bytes:
    with wave.open(str(path), "rb") as source:
        source.setpos(start_ms * RATE // 1000)
        return source.readframes((end_ms - start_ms) * RATE // 1000)


def test_preview_preserves_exact_pcm_and_source(audio: Path, tmp_path: Path) -> None:
    original = audio.read_bytes()
    dest = tmp_path / "preview.wav"
    result = create_context_preview(audio, 1000, 3000, 1250, dest)
    assert result.start_ms == 1000
    assert result.original_end_ms == 3000
    assert result.preview_end_ms == 4250
    assert result.requested_context_ms == 1250
    assert result.included_context_ms == 1250
    assert result.truncated_at_eof is False
    with wave.open(str(dest), "rb") as preview:
        assert (preview.getnchannels(), preview.getsampwidth(), preview.getframerate()) == (
            1,
            2,
            RATE,
        )
        assert preview.getnframes() == 3250 * RATE // 1000
        assert preview.readframes(preview.getnframes()) == read_interval(audio, 1000, 4250)
    assert audio.read_bytes() == original
    assert os.stat(dest).st_mode & 0o777 == 0o600


def test_end_of_file_clamps_context_and_reports_exact_amount(audio: Path, tmp_path: Path) -> None:
    dest = tmp_path / "clamped.wav"
    result = create_context_preview(audio, 6000, 7500, 1000, dest)
    assert result.preview_end_ms == 8000
    assert result.requested_context_ms == 1000
    assert result.included_context_ms == 500
    assert result.truncated_at_eof is True
    with wave.open(str(dest), "rb") as preview:
        assert preview.readframes(preview.getnframes()) == read_interval(audio, 6000, 8000)


def test_zero_context_reproduces_exact_original(audio: Path, tmp_path: Path) -> None:
    dest = tmp_path / "unchanged.wav"
    result = create_context_preview(audio, 200, 4200, 0, dest)
    assert result.preview_end_ms == 4200
    assert result.included_context_ms == 0
    assert result.truncated_at_eof is False
    with wave.open(str(dest), "rb") as preview:
        assert preview.readframes(preview.getnframes()) == read_interval(audio, 200, 4200)


def test_maximum_context_ending_exactly_at_eof_is_not_truncated(
    audio: Path, tmp_path: Path
) -> None:
    result = create_context_preview(audio, 1000, 3000, 5000, tmp_path / "max.wav")
    assert result.preview_end_ms == 8000
    assert result.included_context_ms == 5000
    assert result.truncated_at_eof is False


def test_partial_millisecond_at_source_end_is_not_extended_or_claimed(tmp_path: Path) -> None:
    audio = tmp_path / "fraction.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes(b"\0\0" * (RATE + 8))
    dest = tmp_path / "preview.wav"
    result = create_context_preview(audio, 0, 1000, 1, dest)
    assert result.preview_end_ms == 1000
    assert result.included_context_ms == 0
    assert result.truncated_at_eof is True
    with wave.open(str(dest), "rb") as preview:
        assert preview.getnframes() == RATE


@pytest.mark.parametrize(
    "start,end,context",
    [
        (-1, 1000, 0),
        (100, 100, 0),
        (200, 100, 0),
        (0, 8001, 0),
        (0, 1000, -1),
        (0, 1000, 5001),
        (True, 1000, 0),
        (0, 1000, False),
    ],
)
def test_rejects_bad_boundaries_without_output(
    audio: Path, tmp_path: Path, start: int, end: int, context: int
) -> None:
    dest = tmp_path / "preview.wav"
    with pytest.raises(AudioContextError):
        create_context_preview(audio, start, end, context, dest)
    assert not dest.exists()


def test_existing_file_and_dangling_symlink_are_not_overwritten(
    audio: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "existing.wav"
    dest.write_bytes(b"keep this")
    with pytest.raises(AudioContextError):
        create_context_preview(audio, 0, 1000, 100, dest)
    assert dest.read_bytes() == b"keep this"
    link = tmp_path / "link.wav"
    link.symlink_to(tmp_path / "missing.wav")
    with pytest.raises(AudioContextError):
        create_context_preview(audio, 0, 1000, 100, link)
    assert link.is_symlink()
    assert not (tmp_path / "missing.wav").exists()


def test_source_cannot_be_overwritten(audio: Path) -> None:
    original = audio.read_bytes()
    with pytest.raises(AudioContextError):
        create_context_preview(audio, 0, 1000, 100, audio)
    assert audio.read_bytes() == original


def test_malformed_missing_and_truncated_audio_leave_no_output(audio: Path, tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.wav"
    invalid.write_bytes(b"not a wav")
    truncated = tmp_path / "truncated.wav"
    truncated.write_bytes(audio.read_bytes()[:-10])
    for source in (invalid, truncated, tmp_path / "missing.wav"):
        dest = tmp_path / "preview.wav"
        with pytest.raises(AudioContextError):
            create_context_preview(source, 0, 1000, 200, dest)
        assert not dest.exists()


@pytest.mark.parametrize("channels,width,rate", [(2, 2, RATE), (1, 1, RATE), (1, 2, 8000)])
def test_unsupported_audio_format_leaves_no_output(
    tmp_path: Path, channels: int, width: int, rate: int
) -> None:
    audio = tmp_path / "unsupported.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(bytes(rate * channels * width))
    dest = tmp_path / "preview.wav"
    with pytest.raises(AudioContextError):
        create_context_preview(audio, 0, 500, 200, dest)
    assert not dest.exists()


def test_long_preview_reads_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "long.wav"
    make_wav(audio, 120_000)
    requests: list[int] = []
    original = wave.Wave_read.readframes

    def tracked_read(source: wave.Wave_read, frames: int) -> bytes:
        requests.append(frames)
        return original(source, frames)

    monkeypatch.setattr(wave.Wave_read, "readframes", tracked_read)
    dest = tmp_path / "preview.wav"
    create_context_preview(audio, 1000, 101_000, 1000, dest)
    assert requests
    assert max(requests) <= RATE
    assert len(requests) > 2


def test_failed_write_removes_partial_output(
    audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = audio.read_bytes()
    dest = tmp_path / "failed.wav"
    write = wave.Wave_write.writeframesraw
    calls = 0

    def fail_second_write(output: wave.Wave_write, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated write failure")
        write(output, data)

    monkeypatch.setattr(wave.Wave_write, "writeframesraw", fail_second_write)
    with pytest.raises(AudioContextError):
        create_context_preview(audio, 0, 5000, 1000, dest)
    assert calls >= 2
    assert not dest.exists()
    assert audio.read_bytes() == original
