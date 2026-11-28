# Changelog

## Unreleased

This project is experimental and is not yet a complete video-clipping service.

### Added

- An interactive public homepage and example clip review experience.
- An experimental local workflow to extract audio, transcribe with Deepgram, request clip suggestions from Groq, resume windowed analysis and export a manually selected range.
- Local tools for transcript-based candidates, speaker-label corrections, ranking experiments and reviewing clip endings.
- Playwright browser tests, a Bruno health request and development guides.

### Known limitations

- The homepage example uses sample content. Sign-in and a connected creator dashboard are not implemented.
- Suggestions are unranked and may be empty or incomplete. Creators must review context, speaker labels and boundaries before exporting.
- Long recordings may need multiple runs. Provider rate limits can interrupt processing; separate result files are not automatically combined or deduplicated.
- One long video was processed across multiple runs; its suggestions have not all been reviewed. Laughter, overlapping speech and natural endings need further listening validation.
- Video export supports only certain audio and video timelines; check the exported clip and sound.
