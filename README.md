# AI Video Clipping SaaS

A work-in-progress tool for finding short clips in longer videos.

The backend can extract audio and transcribe it with Deepgram. Highlight selection and video clipping are still in development.

## Run the API

You need Docker and Docker Compose.

```sh
test -e .env || cp -n .env.example .env
docker compose up --build -d
```

The first command keeps an existing `.env` file. The example password is for local development. Open the [API docs](http://127.0.0.1:8000/docs) once the containers are running.

To stop the API and database without deleting database data:

```sh
docker compose down
```

## Transcribe a video locally

You need Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), FFmpeg and FFprobe. Add your Deepgram key to `.env` as `DEEPGRAM_API_KEY`, then run:

```sh
uv sync --locked
mkdir -p outputs
uv run python -m clipforge.cli /path/to/video.mp4 outputs/audio.wav
uv run python -m clipforge.transcribe_cli outputs/audio.wav outputs/transcript.json
```

Use a video you have permission to process and new output filenames. Transcription sends audio to Deepgram and may use API credits. Requests opt out of Deepgram's model-improvement program. Set `DEEPGRAM_REGION` in `.env` to `global` (default) or `india`.

Keep `.env`, recordings, and transcripts private. The `outputs/` directory is excluded from Git.

## Checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```
