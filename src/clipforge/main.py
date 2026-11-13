"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="AI Video Clipping SaaS API",
    description="AI-powered video highlight generation platform",
    version="0.1.0",
)


@app.get("/api/v1/health")
async def health_check() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {
        "status": "healthy",
        "service": "clipforge-api",
        "version": "0.1.0",
    }
