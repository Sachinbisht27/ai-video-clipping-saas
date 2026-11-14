"""Provider-independent transcript types and prerecorded Deepgram transcription."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

import httpx


class TranscriptionError(Exception):
    """An expected, safe-to-display transcription error."""


class TranscriptionConfigurationError(TranscriptionError):
    """Transcription is not configured."""


class InvalidAudioError(TranscriptionError):
    """The selected local audio is not usable."""


class TranscriptionProviderError(TranscriptionError):
    """The transcription provider could not process the request."""


class InvalidTranscriptError(TranscriptionError):
    """The provider returned an invalid transcript."""


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    """One spoken word, timed in milliseconds from the start of the audio."""

    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None
    speaker: int | None = None


@dataclass(frozen=True, slots=True)
class Transcript:
    """Normalized transcript that is independent of a particular provider."""

    provider: str
    language: str
    words: tuple[TranscriptWord, ...]

    @property
    def full_text(self) -> str:
        return " ".join(word.text for word in self.words)


class Transcriber(Protocol):
    """Interface for prerecorded audio transcription providers."""

    def transcribe(self, audio: Path) -> Transcript: ...


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTranscriptError("Transcription provider returned invalid word data")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidTranscriptError("Transcription provider returned invalid word data")
    return result


def _parse_response(data: object, language: str) -> Transcript:
    try:
        if not isinstance(data, dict):
            raise TypeError
        words_data = data["results"]["channels"][0]["alternatives"][0]["words"]
        if not isinstance(words_data, list):
            raise TypeError
        words: list[TranscriptWord] = []
        for item in words_data:
            if not isinstance(item, dict):
                raise TypeError
            text = item.get("punctuated_word") or item["word"]
            if not isinstance(text, str) or not text.strip():
                raise TypeError
            start = _number(item["start"])
            end = _number(item["end"])
            if start < 0 or end < start:
                raise InvalidTranscriptError("Transcription provider returned invalid word timings")
            confidence = item.get("confidence")
            confidence = None if confidence is None else _number(confidence)
            if confidence is not None and not 0 <= confidence <= 1:
                raise InvalidTranscriptError("Transcription provider returned invalid confidence")
            speaker = item.get("speaker")
            if speaker is not None and (
                isinstance(speaker, bool) or not isinstance(speaker, int) or speaker < 0
            ):
                raise InvalidTranscriptError("Transcription provider returned invalid speaker data")
            words.append(
                TranscriptWord(
                    text=text,
                    start_ms=round(start * 1000),
                    end_ms=round(end * 1000),
                    confidence=confidence,
                    speaker=speaker,
                )
            )
    except (KeyError, IndexError, TypeError, OverflowError):
        message = "Transcription provider returned an invalid response"
        raise InvalidTranscriptError(message) from None
    return Transcript(provider="deepgram", language=language, words=tuple(words))


def _chunks(handle: BinaryIO) -> Iterator[bytes]:
    # The handle is kept open by the caller until the request completes.
    while chunk := handle.read(64 * 1024):
        yield chunk


class DeepgramTranscriber:
    """Transcribe an explicitly selected WAV file using Deepgram's REST API."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        model: str = "nova-3",
        language: str = "en",
    ) -> None:
        if not api_key.strip():
            raise TranscriptionConfigurationError("Set DEEPGRAM_API_KEY before transcribing")
        self._api_key = api_key
        self._client = client
        self._model = model
        self._language = language

    def transcribe(self, audio: Path) -> Transcript:
        path = Path(audio)
        if path.suffix.lower() != ".wav" or not path.is_file():
            raise InvalidAudioError("Choose an existing WAV audio file")
        try:
            size = path.stat().st_size
        except OSError:
            raise InvalidAudioError("Could not access the audio file") from None
        if size <= 0 or size > 2_000_000_000:
            raise InvalidAudioError("Audio must be non-empty and no larger than 2 GB")

        def post(client: httpx.Client) -> httpx.Response:
            with path.open("rb") as audio_file:
                return client.post(
                    "https://api.deepgram.com/v1/listen",
                    params={
                        "model": self._model,
                        "language": self._language,
                        "smart_format": "true",
                        "diarize": "true",
                    },
                    headers={
                        "Authorization": f"Token {self._api_key}",
                        "Content-Type": "audio/wav",
                        "Content-Length": str(size),
                    },
                    content=_chunks(audio_file),
                    timeout=httpx.Timeout(660, connect=10),
                )

        try:
            if self._client is None:
                with httpx.Client() as client:
                    response = post(client)
            else:
                response = post(self._client)
        except OSError:
            raise InvalidAudioError("Could not read the audio file") from None
        except httpx.HTTPError:
            raise TranscriptionProviderError("Could not reach the transcription provider") from None
        if not response.is_success:
            raise TranscriptionProviderError(
                f"Transcription provider rejected the request (HTTP {response.status_code})"
            )
        try:
            data: object = response.json()
        except ValueError:
            message = "Transcription provider returned an invalid response"
            raise InvalidTranscriptError(message) from None
        return _parse_response(data, self._language)
