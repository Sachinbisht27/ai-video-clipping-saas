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

## Try AI-assisted boundaries

This experiment sends transcript text, word timings, and anonymous speaker labels to the selected provider. Set the matching key in your local `.env`, then run one of:

```sh
# OpenAI; requires OPENAI_API_KEY and may use API credits
uv run python -m clipforge.semantic_candidates_cli outputs/transcript.json outputs/audio.wav outputs/ai-candidates.json --provider openai

# Groq; requires GROQ_API_KEY; check your account's free-tier limits
uv run python -m clipforge.semantic_candidates_cli outputs/transcript.json outputs/audio.wav outputs/ai-candidates.json --provider groq
```

Both commands require a new output file and the WAV used for transcription. They check returned word indices against the audio and write a private JSON file. Groq asks for short (20–60 seconds) and long (60–180 seconds) clips by default; `--length short` or `--length long` requests one type. These lengths are preferences, so results can be shorter or longer. OpenAI currently targets 20–60 seconds.

For longer transcripts, Groq can process overlapping sections. Limit the first run to one section to check your account's quota:

```sh
uv run python -m clipforge.semantic_candidates_cli outputs/transcript.json outputs/audio.wav outputs/first-section.json --provider groq --windowed --max-windows 1
```

The output reports each attempted section, rejected proposals, token usage where available, and the next section to process. A partial run exits with status 2. Resume with `--start-window` and a new output filename. Exact duplicate word ranges are removed within each run; alternate boundaries and candidates from separate runs are preserved. Word limits or provider quotas can shorten sections or leave them unprocessed; overlap does not guarantee that every long thought fits in a section. Groq's windowed mode proposes boundaries at transcript segment edges, which may need refinement.

The per-request transcript data limit is 128,000 bytes for OpenAI and 32,000 bytes for Groq; a request below these limits may still exceed provider quotas.

Requests may incur charges, including on failure, depending on the selected provider and account. OpenAI requests set `store=false`; see [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data). Groq provides separate [data controls](https://console.groq.com/docs/your-data); zero data retention must be configured there, not in this application.

This is an experiment, not a finished clipping feature. Listen to results: valid timestamps do not prove a thought is complete, and transcript or speaker errors can affect suggestions. The application cannot verify that the WAV and transcript match or that the video and WAV timelines align.

## Correct speaker labels

When you know two labels refer to the same person, make a corrected copy:

```sh
uv run python -m clipforge.speaker_cli outputs/transcript.json outputs/corrected.json --map 2=0
```

This changes every occurrence of label 2 in that transcript to label 0 and keeps the original label on changed words. It does not identify speakers or split a label used for multiple people. Review the mapping first; the original file stays unchanged.

## Rank candidates

The experimental ranking command sends candidate transcript text and nearby context to Groq. It uses `GROQ_API_KEY`, and requests may use account quota or incur charges. Use the same source transcript, WAV and candidate JSON that belong together:

```sh
uv run python -m clipforge.ranking_cli outputs/transcript.json outputs/audio.wav outputs/ai-candidates.json outputs/ranking.json
```

The default is one batch of up to four candidates. Use `--start-batch` and a new output file to continue, or `--max-batches` to request more. A partial or failed run saves available results and exits with status 2. Assessments may be suitable, unsuitable or uncertain; missing assessments are reported. The score and explanation are model opinions based on transcript text, not predictions of virality or checks of video or audio quality.

Scores from separate batches are not calibrated against each other. Each batch has its own short and long order; when multiple batches are processed, the combined results stay in input order and must not be treated as a global leaderboard. Review the proposed clips and their cited source words before use. The `boundary_variants` groups highlight similar time ranges for review; they do not select a better version, change scores, or remove candidates. When both clips are at most 20 seconds, each must overlap the group’s first clip by at least 80% of their combined span, with starts and ends within three seconds. Other pairs require 95% overlap and a five-second limit.

## Configuration

| Setting | Purpose |
| --- | --- |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Local PostgreSQL credentials used by Docker Compose. |
| `DEEPGRAM_API_KEY` | Required for transcription, but not for starting the API. |
| `DEEPGRAM_REGION` | `global` (default) or `india`. The client does not silently fall back to another region. |
| `OPENAI_API_KEY`, `GROQ_API_KEY` | Use the key for your selected experimental boundary provider. |

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
