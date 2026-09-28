from pathlib import Path

from fastapi.testclient import TestClient

from main import app


def test_root_serves_index_html(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("API_TOKEN", "dev-token")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("SANDBOX_ROOT", str(tmp_path / "sandbox"))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "storage-agent" in response.text.lower()
