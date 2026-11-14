FROM ghcr.io/astral-sh/uv:0.13.8 AS uv
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev && \
    useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
CMD ["uvicorn", "clipforge.main:app", "--host", "0.0.0.0", "--port", "8000"]
