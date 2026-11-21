"""Apply explicit speaker-label corrections without modifying source words."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from clipforge.services.transcription import TranscriptWord


class SpeakerCorrectionError(ValueError):
    """A speaker mapping or transcript contains invalid labels."""


def _label(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def correct_speakers(
    words: Sequence[TranscriptWord], mapping: Mapping[int, int]
) -> tuple[TranscriptWord, ...]:
    """Apply human-supplied current-label mappings simultaneously.

    The first provider label is kept on each corrected word as ``original_speaker``.
    A mapping applies to every occurrence of the current label in this transcript;
    it is not a speaker-identity inference and cannot split a label by turn. Unknown
    labels stay unknown. All other word data, including timings, remains unchanged.
    """
    if not isinstance(mapping, Mapping) or not mapping:
        raise SpeakerCorrectionError("Provide at least one speaker mapping")
    if any(not _label(source) or not _label(target) for source, target in mapping.items()):
        raise SpeakerCorrectionError("Speaker labels must be non-negative integers")
    items = tuple(words)
    for word in items:
        if (
            not isinstance(word, TranscriptWord)
            or (word.speaker is not None and not _label(word.speaker))
            or (word.original_speaker is not None and not _label(word.original_speaker))
        ):
            raise SpeakerCorrectionError("Transcript contains invalid speaker labels")
    available = {word.speaker for word in items if word.speaker is not None}
    if not set(mapping).issubset(available):
        raise SpeakerCorrectionError("A source speaker label is not present in the transcript")
    corrected: list[TranscriptWord] = []
    for word in items:
        if (
            word.speaker is None
            or word.speaker not in mapping
            or mapping[word.speaker] == word.speaker
        ):
            corrected.append(word)
            continue
        corrected.append(
            replace(
                word,
                speaker=mapping[word.speaker],
                original_speaker=(
                    word.original_speaker if word.original_speaker is not None else word.speaker
                ),
            )
        )
    return tuple(corrected)
