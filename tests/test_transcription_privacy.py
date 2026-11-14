"""Regional routing and privacy checks using simulated Deepgram responses."""

import json
import sys
import wave
from pathlib import Path

import httpx
import pytest
from pydantic_settings import SettingsError

from clipforge import transcribe_cli
from clipforge.services.transcription import (
    DeepgramTranscriber,
    Transcript,
    TranscriptionConfigurationError,
    TranscriptionProviderError,
)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 16000)
    return path


def empty_response() -> dict[str, object]:
    return {"results": {"channels": [{"alternatives": [{"words": []}]}]}}


@pytest.mark.parametrize(
    ("region", "expected_host"),
    [(None, "api.deepgram.com"), ("global", "api.deepgram.com"), ("india", "api.in.deepgram.com")],
)
def test_approved_region_sends_opted_out_request(
    audio: Path, region: str | None, expected_host: str
) -> None:
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.scheme == "https"
        assert request.url.host == expected_host
        assert request.url.path == "/v1/listen"
        assert request.url.params["mip_opt_out"] == "true"
        assert request.headers["authorization"] == "Token fixture-key"
        assert request.read() == audio.read_bytes()
        return httpx.Response(200, json=empty_response())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        if region is None:
            transcriber = DeepgramTranscriber("fixture-key", client=client)
        else:
            transcriber = DeepgramTranscriber("fixture-key", region=region, client=client)
        assert transcriber.transcribe(audio).words == ()
    assert len(seen) == 1


@pytest.mark.parametrize(
    "region", ["", "eu", "global ", "https://attacker.example", "api.in.deepgram.com"]
)
def test_unapproved_region_is_rejected_before_any_request(audio: Path, region: str) -> None:
    def unexpected(_: httpx.Request) -> httpx.Response:
        pytest.fail("An invalid region must never make a network request")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(TranscriptionConfigurationError):
            DeepgramTranscriber("fixture-key", region=region, client=client).transcribe(audio)


@pytest.mark.parametrize("status", [302, 307, 308, 503])
def test_india_failure_or_redirect_never_falls_back(audio: Path, status: int) -> None:
    hosts: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        assert request.url.params["mip_opt_out"] == "true"
        return httpx.Response(status, headers={"Location": "https://untrusted.invalid/v1/listen"})

    # Even a client configured to follow redirects must not transmit audio or
    # credentials to a redirected destination.
    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        with pytest.raises(TranscriptionProviderError):
            DeepgramTranscriber("fixture-key", region="india", client=client).transcribe(audio)
    assert hosts == ["api.in.deepgram.com"]


@pytest.mark.parametrize(
    ("dotenv_region", "environment_region", "expected"),
    [(None, None, "global"), ("india", None, "india"), ("india", "global", "global")],
)
def test_cli_loads_region_and_environment_overrides_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dotenv_region: str | None,
    environment_region: str | None,
    expected: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPGRAM_REGION", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "fixture-key")
    if dotenv_region is not None:
        (tmp_path / ".env").write_text(f"DEEPGRAM_REGION={dotenv_region}\n")
    if environment_region is not None:
        monkeypatch.setenv("DEEPGRAM_REGION", environment_region)
    source = tmp_path / "audio.wav"
    source.write_bytes(b"unused by simulated provider")
    destination = tmp_path / "transcript.json"
    monkeypatch.setattr(sys, "argv", ["transcribe", str(source), str(destination)])
    observed: list[str] = []

    class FakeTranscriber:
        def __init__(self, api_key: str, *, region: str) -> None:
            assert api_key == "fixture-key"
            observed.append(region)

        def transcribe(self, audio: Path) -> Transcript:
            assert audio == source
            return Transcript(provider="fake", language="en", words=())

    monkeypatch.setattr(transcribe_cli, "DeepgramTranscriber", FakeTranscriber)
    assert transcribe_cli.main() == 0
    assert observed == [expected]
    assert json.loads(destination.read_text())["provider"] == "fake"


def test_cli_reports_invalid_region_before_network_without_exposing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "fixture-secret")
    monkeypatch.setenv("DEEPGRAM_REGION", "https://untrusted.invalid")
    monkeypatch.setattr(
        sys, "argv", ["transcribe", str(tmp_path / "audio.wav"), str(tmp_path / "out.json")]
    )
    assert transcribe_cli.main() == 1
    captured = capsys.readouterr()
    assert "fixture-secret" not in captured.out + captured.err
    assert not (tmp_path / "out.json").exists()


def test_cli_hides_secrets_from_configuration_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["transcribe", str(tmp_path / "audio.wav"), str(tmp_path / "out.json")]
    )

    def bad_settings() -> None:
        raise SettingsError("fixture-secret")

    monkeypatch.setattr(transcribe_cli, "Settings", bad_settings)
    assert transcribe_cli.main() == 1
    captured = capsys.readouterr()
    assert "fixture-secret" not in captured.out + captured.err
