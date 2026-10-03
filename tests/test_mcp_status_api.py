from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.mcp_status import router


def make_app(mcp_status):
    app = FastAPI()
    app.include_router(router)
    app.state.mcp_status = mcp_status
    return app


def test_mcp_status_returns_configured_servers():
    status = [{"name": "dummy", "transport": "stdio", "connected": True, "tools": ["ping"]}]
    app = make_app(status)
    client = TestClient(app)

    response = client.get("/mcp/status")

    assert response.status_code == 200
    assert response.json() == status
