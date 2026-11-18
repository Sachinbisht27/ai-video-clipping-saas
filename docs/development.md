# Local development

These steps use Python 3.15, [uv](https://docs.astral.sh/uv/getting-started/installation/), [FFmpeg and FFprobe](https://ffmpeg.org/download.html) installed on your machine. See the [README](../README.md#first-time-setup) to clone the repository and create `.env`.

If you don't have Python 3.15, [install it with uv](https://docs.astral.sh/uv/guides/install-python/#installing-a-specific-version):

```sh
uv python install 3.15
```

## Install dependencies

From the project folder, run:

```sh
uv sync --locked
```

## Run the API locally

```sh
uv run uvicorn clipforge.main:app --reload
```

Open [API documentation](http://127.0.0.1:8000/docs). Stop the server with `Ctrl+C`. The health endpoint reports whether the API is responding; it does not check the database.

## Transcribe a video

Add your Deepgram key to `.env` as `DEEPGRAM_API_KEY`. For a local MP4 containing video and audio, run:

```sh
mkdir -p outputs
uv run python -m clipforge.cli /path/to/video.mp4 outputs/audio.wav
uv run python -m clipforge.transcribe_cli outputs/audio.wav outputs/transcript.json
```

The first command extracts mono, 16 kHz PCM WAV audio. The second sends that audio to Deepgram and saves words with timestamps in milliseconds. Both commands require new output filenames; they will not overwrite existing files.

Transcription defaults to English and speaker model `v1`. For mixed-language audio, try multilingual mode and speaker model `v2`:

```sh
uv run python -m clipforge.transcribe_cli outputs/audio.wav outputs/multilingual.json --language multi --diarizer v2
```

Speaker labels are anonymous and may need correction. The output records which model was requested.

Use content you have permission to process. Transcription may use Deepgram credits. Keep the output private; `outputs/` is ignored by Git.

## Find candidate sections

After transcription, use the same WAV and the transcript it produced:

```sh
uv run python -m clipforge.candidates_cli outputs/transcript.json outputs/audio.wav outputs/candidates.json
```

This runs locally and creates a private, new JSON file with chronological, unranked candidates, their word timings, and speaker labels. It uses punctuation and pauses as hints, with word boundaries as a fallback; Deepgram utterance boundaries are not yet used. Candidates span 20–60 seconds and do not cross pauses of five seconds or more. Short sections, including a short remainder, may be omitted. Empty input can produce no candidates.

Check the results by listening: boundaries do not guarantee a complete thought, and speaker labels can be wrong. Times are relative to the WAV, not necessarily the source video. Use the exact WAV used for transcription: the tool checks timing bounds but cannot verify that the files match or that audio and video timelines align.

## Configuration

| Setting | Purpose |
| --- | --- |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Local PostgreSQL credentials used by Docker Compose. |
| `DEEPGRAM_API_KEY` | Required for transcription, but not for starting the API. |
| `DEEPGRAM_REGION` | `global` (default) or `india`. The client does not silently fall back to another region. |

PostgreSQL uses the configured credentials when it first initializes its volume. Changing `.env` afterward does not update an existing database.

Every transcription request sets `mip_opt_out=true` to opt out of Deepgram's model-improvement program. This applies to new requests, not earlier submissions. Deepgram may retain usage metadata; its settings do not automatically govern this application's storage or other providers.

## Checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

GitHub Actions runs these checks for pushes and pull requests. FFmpeg and FFprobe must be available for the media integration tests.
