"""Experimental Groq suggestions; calling this sends transcript text externally.

The returned word-index proposals are untrusted and must be validated against
all source words and the WAV duration before use.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx

from clipforge.services.openai_boundaries import (
    _INSTRUCTIONS,
    _SCHEMA,
    BoundaryResult,
    ModelUsage,
    OpenAIBoundaryError,
    _parse_proposals,
)
from clipforge.services.transcription import TranscriptWord

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = "openai/gpt-oss-120b"
# A local guardrail, not a promise that a request fits any account's quota.
_MAX_INPUT_BYTES = 32_000
_MAX_OUTPUT_TOKENS = 2_048


class GroqBoundaryError(ValueError):
    """A safe-to-display configuration, request, or provider-response error."""


def _usage(data: object) -> ModelUsage | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise GroqBoundaryError("The model returned invalid usage data")
    prompt_tokens = data.get("prompt_tokens")
    completion_tokens = data.get("completion_tokens")
    if (
        type(prompt_tokens) is not int
        or type(completion_tokens) is not int
        or prompt_tokens < 0
        or completion_tokens < 0
    ):
        raise GroqBoundaryError("The model returned invalid usage data")
    details = data.get("completion_tokens_details")
    reasoning_tokens = None
    if details is not None:
        if not isinstance(details, dict):
            raise GroqBoundaryError("The model returned invalid usage data")
        reasoning_tokens = details.get("reasoning_tokens")
        if reasoning_tokens is not None and (
            type(reasoning_tokens) is not int
            or reasoning_tokens < 0
            or reasoning_tokens > completion_tokens
        ):
            raise GroqBoundaryError("The model returned invalid usage data")
    return ModelUsage(prompt_tokens, completion_tokens, reasoning_tokens)


def _text(data: object) -> str:
    if not isinstance(data, dict):
        raise GroqBoundaryError("The model returned an invalid response")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise GroqBoundaryError("The model returned an invalid response")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise GroqBoundaryError("The model response was incomplete or invalid")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise GroqBoundaryError("The model returned an invalid response")
    if message.get("refusal"):
        raise GroqBoundaryError("The model refused the request")
    content = message.get("content")
    if not isinstance(content, str):
        raise GroqBoundaryError("The model returned an invalid response")
    return content


class GroqBoundaryProposer:
    """Request boundary suggestions from Groq after explicit external-data consent."""

    def __init__(self, api_key: str, *, client: httpx.Client | None = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise GroqBoundaryError("A Groq API key is required")
        self._api_key = api_key
        self._client = client

    def propose(
        self, words: Sequence[TranscriptWord], *, first_word_index: int = 0
    ) -> BoundaryResult:
        """Send the provided numbered words; output remains unverified.

        Boundary safety flags describe only supplied words. The caller must
        validate the full transcript and WAV both before and after this call.
        """
        if type(first_word_index) is not int or first_word_index < 0:
            raise GroqBoundaryError("The source word offset is invalid")
        if any(
            not isinstance(word, TranscriptWord)
            or not isinstance(word.text, str)
            or not word.text.strip()
            or type(word.start_ms) is not int
            or type(word.end_ms) is not int
            or word.start_ms < 0
            or word.end_ms < word.start_ms
            or (word.speaker is not None and (type(word.speaker) is not int or word.speaker < 0))
            for word in words
        ):
            raise GroqBoundaryError("The transcript contains an invalid word")
        if not words:
            return BoundaryResult((), None)
        if any(words[i].start_ms < words[i - 1].start_ms for i in range(1, len(words))):
            raise GroqBoundaryError("The transcript contains invalid word timings")
        prefix_ends: list[int] = []
        furthest_end = 0
        for word in words:
            furthest_end = max(furthest_end, word.end_ms)
            prefix_ends.append(furthest_end)
        numbered = [
            {
                "index": first_word_index + index,
                "text": word.text,
                "start_ms": word.start_ms,
                "end_ms": word.end_ms,
                "speaker": word.speaker,
                "safe_start": index == 0 or prefix_ends[index - 1] <= word.start_ms,
                "safe_end": index == len(words) - 1
                or prefix_ends[index] <= words[index + 1].start_ms,
            }
            for index, word in enumerate(words)
        ]
        input_text = json.dumps(numbered, ensure_ascii=False)
        try:
            input_size = len(input_text.encode("utf-8"))
        except UnicodeError:
            raise GroqBoundaryError("The transcript contains an invalid word") from None
        if input_size > _MAX_INPUT_BYTES:
            raise GroqBoundaryError("Transcript is too large for one model request")
        payload = {
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": _INSTRUCTIONS},
                {"role": "user", "content": input_text},
            ],
            "max_completion_tokens": _MAX_OUTPUT_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "clip_boundaries", "strict": True, "schema": _SCHEMA},
            },
        }

        def post(client: httpx.Client) -> httpx.Response:
            return client.post(
                _ENDPOINT,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=httpx.Timeout(120, connect=10),
                follow_redirects=False,
            )

        try:
            if self._client is None:
                with httpx.Client() as client:
                    response = post(client)
            else:
                response = post(self._client)
        except httpx.HTTPError:
            raise GroqBoundaryError("Could not reach the model provider") from None
        if not response.is_success:
            raise GroqBoundaryError(
                f"Model provider rejected the request (HTTP {response.status_code})"
            )
        try:
            data: object = response.json()
        except ValueError:
            raise GroqBoundaryError("The model returned an invalid response") from None
        try:
            proposals = _parse_proposals(_text(data), first_word_index, len(words))
        except OpenAIBoundaryError:
            raise GroqBoundaryError("The model returned invalid boundary data") from None
        if not isinstance(data, dict):
            raise GroqBoundaryError("The model returned an invalid response")
        return BoundaryResult(proposals, _usage(data.get("usage")))
