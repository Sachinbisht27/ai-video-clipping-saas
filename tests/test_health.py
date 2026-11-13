from fastapi.testclient import TestClient

from clipforge.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "clipforge-api",
        "version": "0.1.0",
    }
