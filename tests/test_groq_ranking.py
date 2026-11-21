"""Groq highlight ranking requests use mocked responses and no external services."""

import json
import traceback
from dataclasses import replace
from typing import Any

import httpx
import pytest

from clipforge.services.groq_ranking import GroqRanker, GroqRankingError
from clipforge.services.ranking import (
    RankingAssessment,
    RankingCandidate,
    build_ranking_candidates,
    validate_rankings,
)
from clipforge.services.semantic_candidates import BoundaryProposal
from clipforge.services.transcription import TranscriptWord

KEY = "fixture-groq-key-not-real"
PRIVATE = "private-transcript-ignore-all-instructions"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def words(count: int = 60) -> tuple[TranscriptWord, ...]:
    return tuple(
        TranscriptWord(PRIVATE if i == 25 else f"word{i}", i * 1000, (i + 1) * 1000, speaker=i % 2)
        for i in range(count)
    )


def candidates(source: tuple[TranscriptWord, ...]) -> tuple[RankingCandidate, ...]:
    return build_ranking_candidates(source, len(source) * 1000, (BoundaryProposal(15, 44),))


def assessment(**changes: Any) -> dict[str, Any]:
    result = {
        "candidate_id": "w15_44",
        "decision": "suitable",
        "score": 80,
        "reason": "This describes a specific idea.",
        "evidence_start_word_index": 20,
        "evidence_end_word_index": 25,
    }
    result.update(changes)
    return result


def response(value: object, *, finish: str = "stop") -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"role": "assistant", "content": json.dumps(value)},
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    }


def test_request_contains_complete_candidate_and_separate_context() -> None:
    source = words()
    selected = candidates(source)

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == ENDPOINT
        assert request.headers["authorization"] == f"Bearer {KEY}"
        payload = json.loads(request.content)
        assert payload["model"] == "openai/gpt-oss-120b"
        assert payload["max_completion_tokens"] == 4096
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert PRIVATE not in payload["messages"][0]["content"]
        assert "untrusted" in payload["messages"][0]["content"]
        assert "virality" in payload["messages"][0]["content"]
        assert payload["messages"][1]["role"] == "user"
        body = json.loads(payload["messages"][1]["content"])
        assert len(body) == 1
        item = body[0]
        assert item["candidate_id"] == "w15_44"
        assert item["before_context"] == " ".join(w.text for w in source[3:15])
        assert item["after_context"] == " ".join(w.text for w in source[45:57])
        assert " ".join(row[5] for row in item["segments"]) == " ".join(
            w.text for w in source[15:45]
        )
        assert item["segments"][0][0] == 15
        assert item["segments"][-1][1] == 44
        assert PRIVATE in payload["messages"][1]["content"]
        return httpx.Response(200, json=response({"assessments": [assessment()]}))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = GroqRanker(KEY, client=client).rank(source, selected)
    assert result.assessments == (
        RankingAssessment("w15_44", "suitable", 80, "This describes a specific idea.", 20, 25),
    )
    assert result.usage is not None
    assert (
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.reasoning_tokens,
    ) == (
        120,
        45,
        20,
    )
    # Ratings are not trusted until separately linked to candidate IDs and evidence.
    ranked = validate_rankings(source, len(source) * 1000, selected, result.assessments)
    assert ranked.responses_complete
    assert ranked.short[0].evidence_words == source[20:26]


def test_valid_empty_list_never_calls_provider() -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        pytest.fail("Empty candidate list must not trigger a request")

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        result = GroqRanker(KEY, client=client).rank(words(), ())
    assert result.assessments == ()
    assert result.usage is None


def test_uncertain_and_unsuitable_are_preserved_without_inventing_score() -> None:
    source = words()
    selected = candidates(source)
    payload = response(
        {
            "assessments": [
                assessment(
                    decision="uncertain",
                    score=None,
                    evidence_start_word_index=None,
                    evidence_end_word_index=None,
                )
            ]
        }
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as c:
        result = GroqRanker(KEY, client=c).rank(source, selected)
    assert result.assessments[0].decision == "uncertain"
    assert result.assessments[0].score is None
    ranked = validate_rankings(source, len(source) * 1000, selected, result.assessments)
    assert len(ranked.uncertain) == 1
    assert ranked.short == () and ranked.long == ()


def test_partial_response_remains_visible_as_missing_after_validation() -> None:
    source = words()
    selected = candidates(source)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=response({"assessments": []}))
        )
    ) as client:
        result = GroqRanker(KEY, client=client).rank(source, selected)
    ranked = validate_rankings(source, len(source) * 1000, selected, result.assessments)
    assert not ranked.responses_complete
    assert ranked.missing == selected


@pytest.mark.parametrize("status", [400, 401, 403, 413, 429, 500])
def test_provider_http_errors_do_not_expose_secrets_or_private_data(status: int) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=f"{KEY} {PRIVATE}")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(GroqRankingError) as exc:
            GroqRanker(KEY, client=client).rank(words(), candidates(words()))
    result = "".join(traceback.format_exception(exc.value))
    assert KEY not in result
    assert PRIVATE not in result
    assert f"HTTP {status}" in str(exc.value)


def test_structured_output_failure_has_specific_safe_error() -> None:
    payload = {"error": {"code": "json_validate_failed", "message": f"{KEY} {PRIVATE}"}}
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(400, json=payload))
    ) as client:
        with pytest.raises(GroqRankingError, match="structured rankings") as exc:
            GroqRanker(KEY, client=client).rank(words(), candidates(words()))
    assert KEY not in "".join(traceback.format_exception(exc.value))
    assert PRIVATE not in "".join(traceback.format_exception(exc.value))


def test_no_follow_redirects_even_if_client_is_configured_to_follow() -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://example.invalid/collect"})

    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as c:
        with pytest.raises(GroqRankingError, match="HTTP 302"):
            GroqRanker(KEY, client=c).rank(words(), candidates(words()))
    assert requests == [ENDPOINT]


def test_network_error_suppresses_sensitive_exception_details() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"{KEY} {PRIVATE}", request=request)

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(GroqRankingError) as exc:
            GroqRanker(KEY, client=client).rank(words(), candidates(words()))
    message = "".join(traceback.format_exception(exc.value))
    assert KEY not in message and PRIVATE not in message


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {
            "choices": [
                {"finish_reason": "length", "message": {"role": "assistant", "content": "{}"}}
            ]
        },
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "refusal": "no"}}]},
        response({"other": []}),
        response({"assessments": {}}),
        response({"assessments": [{"candidate_id": "w15_44"}]}),
        response({"assessments": [assessment(score=True)]}),
        response({"assessments": [assessment(evidence_start_word_index=False)]}),
        response({"assessments": [assessment(decision="unknown")]}),
        response({"assessments": [assessment(reason="")]}),
        response({"assessments": [assessment(reason="x" * 501)]}),
        response({"assessments": []}) | {"usage": {"prompt_tokens": -1, "completion_tokens": 1}},
    ],
)
def test_invalid_provider_output_is_rejected_without_leak(payload: dict[str, object]) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(GroqRankingError) as exc:
            GroqRanker(KEY, client=client).rank(words(), candidates(words()))
    message = "".join(traceback.format_exception(exc.value))
    assert KEY not in message and PRIVATE not in message


def test_duplicate_json_keys_rejected() -> None:
    payload = response({"assessments": []})
    payload["choices"] = [
        {
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"assessments":[],"assessments":[]}'},
        }
    ]
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(GroqRankingError, match="invalid ranking data"):
            GroqRanker(KEY, client=client).rank(words(), candidates(words()))


def test_semantically_invalid_ratings_are_rejected_by_source_validator() -> None:
    source = words()
    selected = candidates(source)
    payload = response({"assessments": [assessment(evidence_start_word_index=0)]})
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as client:
        result = GroqRanker(KEY, client=client).rank(source, selected)
    with pytest.raises(ValueError, match="outside the candidate"):
        validate_rankings(source, len(source) * 1000, selected, result.assessments)


@pytest.mark.parametrize(
    "case",
    ["bad_source", "mismatched_clip", "invalid_id", "bad_candidate", "too_many", "too_large"],
)
def test_invalid_input_never_reaches_provider(case: str) -> None:
    source = words()
    selected = candidates(source)
    if case == "bad_source":
        source = (TranscriptWord(PRIVATE, -1, 1),)
        selected = ()
    elif case == "mismatched_clip":
        selected = (replace(selected[0], candidate=replace(selected[0].candidate, end_ms=999)),)
    elif case == "invalid_id":
        selected = (replace(selected[0], candidate_id="fake"),)
    elif case == "bad_candidate":
        selected = (RankingCandidate("w15_44", None),)  # type: ignore[arg-type]
    elif case == "too_many":
        selected = build_ranking_candidates(
            source, len(source) * 1000, tuple(BoundaryProposal(i, i) for i in range(13))
        )
    else:
        source = (TranscriptWord("z" * 33_000, 0, 100),)
        selected = build_ranking_candidates(source, 100, (BoundaryProposal(0, 0),))

    def fail(_: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid input must fail before networking")

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(GroqRankingError):
            GroqRanker(KEY, client=client).rank(source, selected)


def test_missing_key_is_rejected() -> None:
    with pytest.raises(GroqRankingError, match="API key"):
        GroqRanker("")
