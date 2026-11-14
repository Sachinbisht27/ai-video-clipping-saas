"""Deepgram transcription contract tests; no external requests or credentials."""

import json
import sys
import traceback
import wave
from pathlib import Path

import httpx
import pytest

from clipforge import transcribe_cli
from clipforge.services.transcription import (
    DeepgramTranscriber,
    Transcript,
    TranscriptionError,
    TranscriptWord,
)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 16000)
    return path


def deepgram_result(words: list[dict[str, object]]) -> dict[str, object]:
    return {
        "results": {
            "channels": [
                {
                    "detected_language": "en",
                    "alternatives": [
                        {"transcript": "Hello world." if words else "", "words": words}
                    ],
                }
            ]
        }
    }


def test_transcription_sends_audio_and_normalizes_word_timestamps(audio: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.host == "api.deepgram.com"
        assert request.url.path == "/v1/listen"
        assert request.url.params["model"] == "nova-3"
        assert request.headers["authorization"] == "Token fake-test-key"
        assert request.headers["content-type"] == "audio/wav"
        assert request.read() == audio.read_bytes()
        return httpx.Response(
            200,
            json=deepgram_result(
                [
                    {
                        "word": "hello",
                        "punctuated_word": "Hello,",
                        "start": 0.125,
                        "end": 0.480,
                        "confidence": 0.98,
                        "speaker": 0,
                    },
                    {"word": "world", "start": 0.5, "end": 0.9, "confidence": 0.91},
                ]
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        transcript = DeepgramTranscriber("fake-test-key", client=client).transcribe(audio)

    assert transcript.provider == "deepgram"
    assert transcript.language == "en"
    assert len(transcript.words) == 2
    assert transcript.words[0].text == "Hello,"
    assert (transcript.words[0].start_ms, transcript.words[0].end_ms) == (125, 480)
    assert transcript.words[0].speaker == 0
    assert transcript.words[0].confidence == pytest.approx(0.98)
    assert (transcript.words[1].start_ms, transcript.words[1].end_ms) == (500, 900)
    assert not transcript.words[0].timing_adjusted
    assert transcript.words[0].original_end_ms is None


@pytest.mark.parametrize("end", [1.475, 1.795, 2.105])
def test_word_end_overrun_is_clamped_and_recorded(audio: Path, end: float) -> None:
    response = deepgram_result([{"word": "last", "start": 0.9, "end": end}])
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        transcript = DeepgramTranscriber("fake-test-key", client=client).transcribe(audio)
    word = transcript.words[0]
    assert (word.start_ms, word.end_ms) == (900, 1000)
    assert word.timing_adjusted
    assert word.original_end_ms == round(end * 1000)
    assert word.original_start_ms is None


@pytest.mark.parametrize("start", [1.0, 1.1])
def test_word_start_at_or_after_audio_end_is_rejected(audio: Path, start: float) -> None:
    response = deepgram_result([{"word": "outside", "start": start, "end": start + 0.5}])
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        with pytest.raises(TranscriptionError, match="beyond the audio"):
            DeepgramTranscriber("fake-test-key", client=client).transcribe(audio)


def test_zero_duration_in_range_is_preserved(audio: Path) -> None:
    response = deepgram_result([{"word": "short", "start": 0.5, "end": 0.5}])
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        transcript = DeepgramTranscriber("fake-test-key", client=client).transcribe(audio)
    assert transcript.words[0].start_ms == transcript.words[0].end_ms == 500
    assert not transcript.words[0].timing_adjusted


def test_submillisecond_rounding_does_not_cross_the_wav_end(tmp_path: Path) -> None:
    audio = tmp_path / "fractional.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 16014)
    # Duration is 1.000875 seconds; rounded timestamps would be 1001 ms,
    # but the last full millisecond inside the audio is 1000 ms.
    response = deepgram_result([{"word": "last", "start": 1.0008, "end": 1.00085}])
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        transcript = DeepgramTranscriber("fake-test-key", client=client).transcribe(audio)
    word = transcript.words[0]
    assert (word.start_ms, word.end_ms) == (1000, 1000)
    assert word.timing_adjusted
    assert word.original_start_ms == word.original_end_ms == 1001


def test_invalid_wav_is_rejected_before_network(tmp_path: Path) -> None:
    path = tmp_path / "invalid.wav"
    path.write_bytes(b"not a wav")

    def unexpected(_: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid WAV must not be sent to Deepgram")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(TranscriptionError, match="valid.*WAV"):
            DeepgramTranscriber("fake-test-key", client=client).transcribe(path)


def test_truncated_wav_is_rejected_before_network(audio: Path) -> None:
    audio.write_bytes(audio.read_bytes()[:-20])

    def unexpected(_: httpx.Request) -> httpx.Response:
        pytest.fail("Truncated WAV must not be sent to Deepgram")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(TranscriptionError, match="truncated"):
            DeepgramTranscriber("fake-test-key", client=client).transcribe(audio)


def test_silent_audio_produces_empty_transcript(audio: Path) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=deepgram_result([])))
    ) as client:
        transcript = DeepgramTranscriber("fake-test-key", client=client).transcribe(audio)
    assert len(transcript.words) == 0


@pytest.mark.parametrize("status", [401, 429, 500])
def test_api_errors_do_not_disclose_key_or_response(audio: Path, status: int) -> None:
    secret = "fake-test-key"

    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="fake-test-key private transcript")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(TranscriptionError) as exc:
            DeepgramTranscriber(secret, client=client).transcribe(audio)
    assert secret not in str(exc.value)
    assert secret not in "".join(traceback.format_exception(exc.value))
    assert "private transcript" not in str(exc.value)


def test_network_error_does_not_disclose_key(audio: Path) -> None:
    secret = "fake-test-key"

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(secret, request=request)

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(TranscriptionError) as exc:
            DeepgramTranscriber(secret, client=client).transcribe(audio)
    assert secret not in str(exc.value)
    assert secret not in "".join(traceback.format_exception(exc.value))


@pytest.mark.parametrize(
    "response",
    [
        {},
        deepgram_result([{"word": "bad", "start": -0.1, "end": 0.3}]),
        deepgram_result([{"word": "bad", "start": 1.0, "end": 0.1}]),
        deepgram_result([{"word": "bad", "start": "not-a-number", "end": 1.0}]),
    ],
)
def test_malformed_provider_response_is_rejected(audio: Path, response: dict[str, object]) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        with pytest.raises(TranscriptionError):
            DeepgramTranscriber("fake-test-key", client=client).transcribe(audio)


def test_missing_file_fails_before_network(tmp_path: Path) -> None:
    def unexpected(_: httpx.Request) -> httpx.Response:
        pytest.fail("Missing input must not be sent to Deepgram")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(TranscriptionError):
            DeepgramTranscriber("fake-test-key", client=client).transcribe(tmp_path / "missing.wav")


def test_cli_loads_test_dotenv_and_writes_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_REGION", raising=False)
    (tmp_path / ".env").write_text("DEEPGRAM_API_KEY=fixture-key\n")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"synthetic-audio")
    output = tmp_path / "transcript.json"
    monkeypatch.setattr(sys, "argv", ["transcribe", str(audio), str(output)])

    class FakeTranscriber:
        def __init__(self, api_key: str, *, region: str) -> None:
            assert api_key == "fixture-key"
            assert region == "global"

        def transcribe(self, path: Path) -> Transcript:
            assert path == audio
            return Transcript(
                provider="fixture",
                language="en",
                words=(TranscriptWord(text="Hello", start_ms=125, end_ms=480),),
            )

    monkeypatch.setattr(transcribe_cli, "DeepgramTranscriber", FakeTranscriber)
    assert transcribe_cli.main() == 0
    saved = json.loads(output.read_text())
    assert saved["full_text"] == "Hello"
    assert saved["words"][0]["start_ms"] == 125
    assert output.stat().st_mode & 0o777 == 0o600


def test_cli_does_not_overwrite_or_call_provider_for_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "fixture-key")
    output = tmp_path / "transcript.json"
    output.write_text("keep this")
    monkeypatch.setattr(sys, "argv", ["transcribe", str(tmp_path / "sample.wav"), str(output)])

    def unexpected(_: str) -> object:
        pytest.fail("Existing output must be rejected before any provider call")

    monkeypatch.setattr(transcribe_cli, "DeepgramTranscriber", unexpected)
    assert transcribe_cli.main() == 1
    assert output.read_text() == "keep this"
