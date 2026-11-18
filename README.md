# AI Video Clipping SaaS

[![CI](https://github.com/Sachinbisht27/ai-video-clipping-saas/actions/workflows/ci.yml/badge.svg)](https://github.com/Sachinbisht27/ai-video-clipping-saas/actions/workflows/ci.yml)
[![Python 3.15](https://img.shields.io/badge/Python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

A work-in-progress tool for finding short clips in longer videos.

The backend can extract audio from an MP4, transcribe it with Deepgram, and find unranked candidate sections. Highlight ranking and video clipping are still in development.

## Requirements

- [Git](https://git-scm.com/downloads)
- [Docker and Docker Compose](https://docs.docker.com/get-started/get-docker/) to run the API and database
- A Deepgram API key only if you want to transcribe audio

For local development and transcription without Docker, you'll also need [uv](https://docs.astral.sh/uv/getting-started/installation/), [Python 3.15](https://docs.astral.sh/uv/guides/install-python/#installing-a-specific-version), and [FFmpeg and FFprobe](https://ffmpeg.org/download.html).

## First-time setup

1. Clone the repository and enter the project folder:

   ```sh
   git clone https://github.com/Sachinbisht27/ai-video-clipping-saas.git
   cd ai-video-clipping-saas
   ```

2. Create a local configuration file. This command preserves an existing `.env`:

   ```sh
   if [ ! -e .env ] && [ ! -L .env ]; then
     cp .env.example .env
   fi
   ```

   The example database password is for local development. Change it in `.env` before starting a new database if needed. Add your Deepgram API key as `DEEPGRAM_API_KEY` when you are ready to transcribe.

3. Start your Docker engine, then build and start the API and PostgreSQL:

   ```sh
   docker compose up --build -d
   ```

Open the [API documentation](http://127.0.0.1:8000/docs) or check [API health](http://127.0.0.1:8000/api/v1/health). The health endpoint checks the API process, not the database. Docker runs the backend and database; there is no video-clipping interface yet.

To stop the containers while keeping database data:

```sh
docker compose down
```

## Try transcription

To extract audio and transcribe a local video, follow the [local development guide](docs/development.md#transcribe-a-video). The command sends audio to Deepgram and may use API credits. Use a video you have permission to process.

## Configuration and privacy

Keep `.env`, videos, and transcripts private. The `.env` file and the `data/`, `uploads/`, and `outputs/` folders are excluded from Git.

`DEEPGRAM_REGION` selects `global` (default) or `india`. Transcription requests ask Deepgram to opt out of its model-improvement program. See [configuration](docs/development.md#configuration) for details.

## Development and checks

The [local development guide](docs/development.md) covers running the API without Docker, transcription, candidate selection, configuration, and checks.

## Contributing

If you have repository access, make changes on a branch and open a pull request against `main`. The required GitHub checks must pass before merging.

## License

No license has been added yet.
