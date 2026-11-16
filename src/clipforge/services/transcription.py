"""Provider-independent transcript types and prerecorded Deepgram transcription."""

from __future__ import annotations

import math
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

import httpx

_DEEPGRAM_ENDPOINTS = {
    "global": "https://api.deepgram.com/v1/listen",
    "india": "https://api.in.deepgram.com/v1/listen",
}


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
    timing_adjusted: bool = False
    original_start_ms: int | None = None
    original_end_ms: int | None = None


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


def _parse_response(data: object, language: str, frames: int, sample_rate: int) -> Transcript:
    # Integer timestamps must not round beyond the actual final audio sample.
    duration_seconds = frames / sample_rate
    duration_ms = frames * 1000 // sample_rate
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
            if start >= duration_seconds:
                raise InvalidTranscriptError(
                    "Transcription provider returned a word beyond the audio"
                )
            original_start_ms = round(start * 1000)
            original_end_ms = round(end * 1000)
            start_adjusted = original_start_ms > duration_ms
            end_adjusted = end > duration_seconds or original_end_ms > duration_ms
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
                    start_ms=min(original_start_ms, duration_ms),
                    end_ms=min(original_end_ms, duration_ms),
                    confidence=confidence,
                    speaker=speaker,
                    timing_adjusted=start_adjusted or end_adjusted,
                    original_start_ms=original_start_ms if start_adjusted else None,
                    original_end_ms=original_end_ms if end_adjusted else None,
                )
            )
    except KeyError, IndexError, TypeError, OverflowError:
        message = "Transcription provider returned an invalid response"
        raise InvalidTranscriptError(message) from None
    return Transcript(provider="deepgram", language=language, words=tuple(words))


def _chunks(handle: BinaryIO) -> Iterator[bytes]:
    # The handle is kept open by the caller until the request completes.
    while chunk := handle.read(64 * 1024):
        yield chunk


def _wav_duration(path: Path) -> tuple[int, int]:
    """Return PCM sample frames and rate; reject unreadable or truncated audio."""
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            sample_rate = audio.getframerate()
            if frames <= 0 or sample_rate <= 0:
                raise InvalidAudioError("Audio must contain at least one sample")
            # Check the last sample frame exists without loading the whole file.
            audio.setpos(frames - 1)
            expected_bytes = audio.getnchannels() * audio.getsampwidth()
            if len(audio.readframes(1)) != expected_bytes:
                raise InvalidAudioError("WAV audio is truncated")
            return frames, sample_rate
    except OSError, EOFError, wave.Error, ValueError:
        raise InvalidAudioError("Choose a valid, uncompressed PCM WAV audio file") from None


class DeepgramTranscriber:
    """Transcribe PCM WAV audio, clamping and flagging out-of-range word ends."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        model: str = "nova-3",
        language: str = "en",
        diarizer: str = "v1",
        region: str = "global",
    ) -> None:
        if not api_key.strip():
            raise TranscriptionConfigurationError("Set DEEPGRAM_API_KEY before transcribing")
        if region not in _DEEPGRAM_ENDPOINTS:
            raise TranscriptionConfigurationError("DEEPGRAM_REGION must be 'global' or 'india'")
        if language not in {"en", "multi"}:
            raise TranscriptionConfigurationError("Language must be 'en' or 'multi'")
        if diarizer not in {"v1", "v2"}:
            raise TranscriptionConfigurationError("Diarizer must be 'v1' or 'v2'")
        self._api_key = api_key
        self._client = client
        self._model = model
        self._language = language
        self._diarizer = diarizer
        self._endpoint = _DEEPGRAM_ENDPOINTS[region]

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
        frames, sample_rate = _wav_duration(path)

        def post(client: httpx.Client) -> httpx.Response:
            with path.open("rb") as audio_file:
                return client.post(
                    self._endpoint,
                    params={
                        "model": self._model,
                        "language": self._language,
                        "smart_format": "true",
                        "diarize_model": self._diarizer,
                        "mip_opt_out": "true",
                    },
                    headers={
                        "Authorization": f"Token {self._api_key}",
                        "Content-Type": "audio/wav",
                        "Content-Length": str(size),
                    },
                    content=_chunks(audio_file),
                    timeout=httpx.Timeout(660, connect=10),
                    follow_redirects=False,
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
        return _parse_response(data, self._language, frames, sample_rate)
