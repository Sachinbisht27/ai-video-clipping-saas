"""Human-supplied speaker mappings; no inference or external services."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from clipforge import speaker_cli
from clipforge.candidates_cli import _TranscriptData
from clipforge.services.speaker_corrections import SpeakerCorrectionError, correct_speakers
from clipforge.services.transcription import TranscriptWord


def test_correction_preserves_provider_label_timings_and_other_data() -> None:
    words = (
        TranscriptWord("First", 0, 100, speaker=0),
        TranscriptWord("Second", 100, 250, speaker=1),
        TranscriptWord(
            "Again.",
            250,
            500,
            confidence=0.8,
            speaker=2,
            timing_adjusted=True,
            original_end_ms=600,
        ),
        TranscriptWord("Unknown", 500, 550, speaker=None),
    )
    corrected = correct_speakers(words, {2: 0})
    assert [word.speaker for word in corrected] == [0, 1, 0, None]
    assert corrected[2].original_speaker == 2
    assert (corrected[2].start_ms, corrected[2].end_ms) == (250, 500)
    assert corrected[2].original_end_ms == 600
    assert corrected[2].confidence == 0.8
    assert words[2].speaker == 2
    assert words[2].original_speaker is None
    assert corrected[0] is words[0]


def test_subsequent_correction_preserves_first_provider_label() -> None:
    words = (TranscriptWord("first", 0, 100, speaker=2),)
    first = correct_speakers(words, {2: 0})
    second = correct_speakers(first, {0: 1})
    assert (second[0].speaker, second[0].original_speaker) == (1, 2)


def test_mappings_apply_simultaneously_without_cascading() -> None:
    words = (TranscriptWord("a", 0, 50, speaker=0), TranscriptWord("b", 50, 100, speaker=1))
    corrected = correct_speakers(words, {0: 1, 1: 0})
    assert [(w.speaker, w.original_speaker) for w in corrected] == [(1, 0), (0, 1)]


@pytest.mark.parametrize("mapping", [{}, {True: 0}, {0: True}, {-1: 0}, {0: -1}, {3: 0}])
def test_rejects_invalid_or_missing_source_labels(mapping: dict[int, int]) -> None:
    with pytest.raises(SpeakerCorrectionError):
        correct_speakers((TranscriptWord("a", 0, 10, speaker=0),), mapping)


def test_invalid_existing_original_speaker_is_rejected() -> None:
    with pytest.raises(SpeakerCorrectionError):
        correct_speakers((TranscriptWord("a", 0, 10, speaker=0, original_speaker=-1),), {0: 1})


def _input(path: Path) -> str:
    text = json.dumps(
        {
            "provider": "example",
            "language": "multi",
            "diarization_model_requested": "v2",
            "full_text": "नमस्ते hello again",
            "custom_metadata": {"keep": "yes"},
            "words": [
                {"text": "नमस्ते", "start_ms": 0, "end_ms": 200, "speaker": 0},
                {"text": "hello", "start_ms": 200, "end_ms": 400, "speaker": 1},
                {
                    "text": "again",
                    "start_ms": 400,
                    "end_ms": 600,
                    "speaker": 2,
                    "custom_word_metadata": "keep",
                },
            ],
        },
        ensure_ascii=False,
    )
    path.write_text(text, encoding="utf-8")
    return text


def test_cli_creates_private_new_transcript_and_candidate_parser_retains_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "transcript.json", tmp_path / "corrected.json"
    original = _input(source)
    monkeypatch.setattr(sys, "argv", ["speaker", str(source), str(output), "--map", "2=0"])
    assert speaker_cli.main() == 0
    assert source.read_text() == original
    saved = json.loads(output.read_text())
    assert saved["provider"] == "example"
    assert saved["diarization_model_requested"] == "v2"
    assert saved["custom_metadata"] == {"keep": "yes"}
    assert saved["words"][2]["custom_word_metadata"] == "keep"
    assert saved["words"][2]["speaker"] == 0
    assert saved["words"][2]["original_speaker"] == 2
    assert (saved["words"][2]["start_ms"], saved["words"][2]["end_ms"]) == (400, 600)
    assert output.stat().st_mode & 0o777 == 0o600
    parsed = _TranscriptData.model_validate_json(output.read_text())
    words = [TranscriptWord(**word.model_dump()) for word in parsed.words]
    assert asdict(words[2])["original_speaker"] == 2


def test_cli_refuses_existing_output_and_dangling_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "transcript.json", tmp_path / "corrected.json"
    _input(source)
    output.write_text("keep")
    monkeypatch.setattr(sys, "argv", ["speaker", str(source), str(output), "--map", "2=0"])
    assert speaker_cli.main() == 1
    assert output.read_text() == "keep"
    output.unlink()
    output.symlink_to(tmp_path / "missing")
    assert speaker_cli.main() == 1
    assert output.is_symlink()


def test_cli_rejects_unknown_and_duplicate_mappings_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "transcript.json", tmp_path / "corrected.json"
    _input(source)
    for flags in (["--map", "9=0"], ["--map", "2=0", "--map", "2=1"]):
        monkeypatch.setattr(sys, "argv", ["speaker", str(source), str(output), *flags])
        assert speaker_cli.main() == 1
        assert not output.exists()


@pytest.mark.parametrize("value", ["-1=0", "1=-1", "1:0", "foo=1", "1=2=3"])
def test_cli_rejects_invalid_mapping_before_creating_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    source, output = tmp_path / "transcript.json", tmp_path / "corrected.json"
    _input(source)
    monkeypatch.setattr(sys, "argv", ["speaker", str(source), str(output), "--map", value])
    with pytest.raises(SystemExit) as exc:
        speaker_cli.main()
    assert exc.value.code == 2
    assert not output.exists()
