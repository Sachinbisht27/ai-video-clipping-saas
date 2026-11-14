# AI Video Clipping SaaS

A starting point for an AI video clipping service, built with FastAPI and Python 3.12. The project provides an API health endpoint and a local video-to-audio ingestion step. Transcription, highlight selection, and clipping are not implemented yet.

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
