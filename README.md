# AI Video Clipping SaaS

A starting point for an AI video clipping service, built with FastAPI and Python 3.12. The project provides an API health endpoint, a local video-to-audio ingestion step, and a transcription provider. Highlight selection and clipping are not implemented yet.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/) for local development
- FFmpeg and FFprobe on `PATH` for video ingestion and its integration tests
- Docker and Docker Compose for the containerized setup

## Local development

Install dependencies and start the API:

```sh
uv sync --locked --group dev
uv run uvicorn clipforge.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. The interactive API documentation is at `http://127.0.0.1:8000/docs`.

Check that the API process is running:

```sh
curl http://127.0.0.1:8000/api/v1/health
```

This endpoint reports process liveness; it does not check database connectivity.

## Local video ingestion

For a video you are authorized to process, validate the MP4 and extract a mono, 16 kHz WAV file for transcription:

```sh
uv run python -m clipforge.cli /path/to/source.mp4 /path/to/audio.wav
```

The input must have video and audio streams. The output must not already exist. Keep source videos and generated audio out of version control; the `.gitignore` covers common media formats and local `data/`, `uploads/`, and `outputs/` directories. This command runs locally and does not upload or transcribe the video.

## Local transcription

Add your development key as `DEEPGRAM_API_KEY` in `.env`, then transcribe an extracted uncompressed PCM WAV file:

```sh
uv run python -m clipforge.transcribe_cli /path/to/audio.wav outputs/transcript.json
```

Create the output directory beforehand. The command sends the selected audio to Deepgram and may consume API credits. Use audio you have permission to process. The output contains the spoken words and their start and end times in milliseconds; store transcripts in the ignored `outputs/` directory because they may contain private content. The output file must not already exist. Word endings beyond the actual WAV duration are capped and marked `timing_adjusted`, with the provider's original time retained in `original_end_ms`. A word starting outside the audio is rejected. Bounds alone do not guarantee that a timestamp is accurately aligned with speech.

The key is loaded from `.env` for local use; an environment variable takes precedence. Set `DEEPGRAM_REGION=global` (the default) or `DEEPGRAM_REGION=india` to select the Deepgram endpoint. The client rejects unknown regions and does not silently fall back to a different region. Docker Compose passes both settings to the API container when it is created; recreate the container after changing them.

Each transcription request sets `mip_opt_out=true` to opt the audio and transcript out of Deepgram's model-improvement program. Deepgram's stated retention and regional-processing guarantees apply only to Deepgram's processing, not automatically to this application's storage or other providers. Deepgram may retain usage metadata; consult its current data policy and account terms. The setting does not alter earlier requests. The API does not yet expose a transcription endpoint. Automated tests simulate Deepgram responses and do not call the service.

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

GitHub Actions runs these checks on pushes and pull requests.

## Run with Docker Compose

Create a local environment file, then build and start the API and PostgreSQL:

```sh
cp .env.example .env
docker compose up --build
```

The API is available at `http://127.0.0.1:8000`. PostgreSQL is available to the API within the Compose network and its data is stored in the `postgres_data` volume. Stop the services with `docker compose down`; the database volume is retained.

The example credentials are for local development. Change them for any other environment and keep `.env` private. PostgreSQL uses the configured database name and credentials when initializing a new volume; changing them later does not update an existing database.
