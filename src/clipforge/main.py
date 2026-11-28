"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="AI Video Clipping SaaS API",
    description="AI-powered video highlight generation platform",
    version="0.1.0",
)


static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def homepage() -> FileResponse:
    """Serve the public homepage."""
    return FileResponse(static_dir / "index.html")


@app.get("/api/v1/health")
async def health_check() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {
        "status": "healthy",
        "service": "clipforge-api",
        "version": "0.1.0",
    }
