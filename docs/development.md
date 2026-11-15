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

Use content you have permission to process. Transcription may use Deepgram credits. Keep the output private; `outputs/` is ignored by Git.

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
