"""Language and diarizer options use fake audio, responses, and credentials."""

import json
import sys
import wave
from pathlib import Path

import httpx
import pytest

from clipforge import transcribe_cli
from clipforge.services.transcription import (
    DeepgramTranscriber,
    Transcript,
    TranscriptionConfigurationError,
    TranscriptWord,
)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "test.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 16000)
    return path


def response_data() -> dict[str, object]:
    return {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {
                                    "word": "नमस्ते",
                                    "start": 0.2,
                                    "end": 0.4,
                                    "speaker": 1,
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }


@pytest.mark.parametrize("region", ["global", "india"])
@pytest.mark.parametrize("language", ["en", "multi"])
@pytest.mark.parametrize("diarizer", ["v1", "v2"])
def test_options_use_expected_params_and_keep_privacy(
    audio: Path, region: str, language: str, diarizer: str
) -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert (
            request.url.host
            == {
                "global": "api.deepgram.com",
                "india": "api.in.deepgram.com",
            }[region]
        )
        assert request.url.params["model"] == "nova-3"
        assert request.url.params["language"] == language
        assert request.url.params["diarize_model"] == diarizer
        assert "diarize" not in request.url.params
        assert request.url.params["mip_opt_out"] == "true"
        assert request.url.params["smart_format"] == "true"
        assert request.read() == audio.read_bytes()
        return httpx.Response(200, json=response_data())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = DeepgramTranscriber(
            "fixture-key", client=client, region=region, language=language, diarizer=diarizer
        ).transcribe(audio)

    assert len(requests) == 1
    assert result.language == language
    assert result.words[0].text == "नमस्ते"
    assert result.words[0].speaker == 1
    assert (result.words[0].start_ms, result.words[0].end_ms) == (200, 400)


def test_defaults_use_english_and_v1(audio: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.params["language"] == "en"
        assert request.url.params["diarize_model"] == "v1"
        assert "diarize" not in request.url.params
        assert request.url.params["mip_opt_out"] == "true"
        return httpx.Response(200, json=response_data())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        DeepgramTranscriber("fixture-key", client=client).transcribe(audio)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("language", ""),
        ("language", "hi"),
        ("language", "https://untrusted.invalid"),
        ("diarizer", ""),
        ("diarizer", "latest"),
        ("diarizer", "https://untrusted.invalid"),
    ],
)
def test_unsupported_options_fail_before_network(audio: Path, option: str, value: str) -> None:
    def unexpected(_: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid options must not trigger a network request")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(TranscriptionConfigurationError) as exc:
            DeepgramTranscriber("fixture-key", client=client, **{option: value}).transcribe(audio)
    assert "fixture-key" not in str(exc.value)


@pytest.mark.parametrize(
    ("flags", "language", "diarizer"),
    [([], "en", "v1"), (["--language", "multi", "--diarizer", "v2"], "multi", "v2")],
)
def test_cli_passes_requested_options_and_preserves_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    language: str,
    diarizer: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "fixture-key")
    monkeypatch.setenv("DEEPGRAM_REGION", "india")
    source, destination = tmp_path / "audio.wav", tmp_path / "transcript.json"
    monkeypatch.setattr(sys, "argv", ["transcribe", str(source), str(destination), *flags])

    class FakeTranscriber:
        def __init__(self, api_key: str, *, region: str, language: str, diarizer: str) -> None:
            assert (api_key, region, language, diarizer) == (
                "fixture-key",
                "india",
                expected_language,
                expected_diarizer,
            )

        def transcribe(self, path: Path) -> Transcript:
            assert path == source
            return Transcript(
                provider="fake",
                language=expected_language,
                words=(TranscriptWord(text="hello", start_ms=0, end_ms=100),),
            )

    expected_language, expected_diarizer = language, diarizer
    monkeypatch.setattr(transcribe_cli, "DeepgramTranscriber", FakeTranscriber)
    assert transcribe_cli.main() == 0
    saved = json.loads(destination.read_text())
    assert saved["language"] == language
    assert saved["diarization_model_requested"] == diarizer


@pytest.mark.parametrize("flags", [["--language", "unsupported"], ["--diarizer", "latest"]])
def test_cli_invalid_option_does_not_call_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "fixture-key")
    monkeypatch.setattr(
        sys, "argv", ["transcribe", str(tmp_path / "in.wav"), str(tmp_path / "out.json"), *flags]
    )

    def unexpected(*_: object, **__: object) -> object:
        pytest.fail("Invalid CLI options must not instantiate a provider")

    monkeypatch.setattr(transcribe_cli, "DeepgramTranscriber", unexpected)
    with pytest.raises(SystemExit) as exc:
        transcribe_cli.main()
    assert exc.value.code == 2
    assert not (tmp_path / "out.json").exists()
