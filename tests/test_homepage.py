from fastapi.testclient import TestClient

from clipforge.main import app


def test_homepage_and_assets_are_served() -> None:
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Interactive concept" in page.text
        for path in ("/static/home.css", "/static/home.js"):
            assert client.get(path).status_code == 200
