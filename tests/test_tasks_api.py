import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver

from agent.core import build_agent
from api.tasks import router
from storage.db import get_task, init_db


def make_app(model, db):
    app = FastAPI()
    app.include_router(router)
    app.state.settings = type("S", (), {"API_TOKEN": "dev-token"})()
    app.state.db = db
    app.state.agent = build_agent(
        model=model,
        tools=[],
        system_prompt="test",
        max_tool_turns=20,
        checkpointer=InMemorySaver(),
        summarize_trigger_tokens=4000,
        summarize_keep_messages=20,
    )
    return app


@pytest.fixture
async def db(tmp_path):
    database = await init_db(tmp_path / "test.db")
    yield database
    await database.close()


def test_post_tasks_without_token_returns_401(db):
    app = make_app(GenericFakeChatModel(messages=iter([])), db)
    client = TestClient(app)

    response = client.post("/tasks", json={"input": "do something"})

    assert response.status_code == 401


def test_post_tasks_with_token_returns_202_and_task_id(db):
    app = make_app(GenericFakeChatModel(messages=iter([AIMessage(content="done")])), db)
    client = TestClient(app)

    response = client.post(
        "/tasks",
        json={"input": "do something"},
        headers={"Authorization": "Bearer dev-token"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert "task_id" in response.json()


def test_get_unknown_task_returns_404(db):
    app = make_app(GenericFakeChatModel(messages=iter([])), db)
    client = TestClient(app)

    response = client.get("/tasks/no-such-id", headers={"Authorization": "Bearer dev-token"})

    assert response.status_code == 404


async def test_task_completes_and_result_is_pollable(db):
    app = make_app(GenericFakeChatModel(messages=iter([AIMessage(content="the answer")])), db)

    # TestClient must be used as a context manager: it runs each request on
    # its own portal event loop, and background asyncio.create_task() work
    # started during a request gets orphaned once that portal loop tears
    # down between bare (non-context-manager) client.post() calls.
    with TestClient(app) as client:
        response = client.post(
            "/tasks",
            json={"input": "what is the answer"},
            headers={"Authorization": "Bearer dev-token"},
        )
        task_id = response.json()["task_id"]

        for _ in range(50):
            task = await get_task(db, task_id)
            if task["status"] != "pending" and task["status"] != "running":
                break
            await asyncio.sleep(0.05)

    assert task["status"] == "completed"
    assert task["result"] == "the answer"
    assert task["started_at"] is not None
    assert task["finished_at"] is not None


async def test_task_records_failure_without_crashing(db):
    app = make_app(GenericFakeChatModel(messages=iter([])), db)

    # A bare hand-rolled model object can't be passed to build_agent(): its
    # SummarizationMiddleware unconditionally calls model.with_retry(), which
    # only a real LangChain Runnable provides. Monkeypatch the already-built
    # agent's ainvoke directly instead, which is what forces a failure at the
    # actual ainvoke boundary this test wants to exercise.
    async def failing_ainvoke(*args, **kwargs):
        raise RuntimeError("LLM unreachable")

    app.state.agent.ainvoke = failing_ainvoke

    with TestClient(app) as client:
        response = client.post(
            "/tasks",
            json={"input": "do something"},
            headers={"Authorization": "Bearer dev-token"},
        )
        task_id = response.json()["task_id"]

        for _ in range(50):
            task = await get_task(db, task_id)
            if task["status"] != "pending" and task["status"] != "running":
                break
            await asyncio.sleep(0.05)

    assert task["status"] == "failed"
    assert "LLM unreachable" in task["error"]
