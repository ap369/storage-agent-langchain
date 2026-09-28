import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.messages import AIMessage, AIMessageChunk, ToolCall
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

from agent.core import build_agent
from api.chat import router
from storage.db import init_db


class FakeToolCallingModel(GenericFakeChatModel):
    """GenericFakeChatModel's _stream() only chunks `content` and legacy
    additional_kwargs["function_call"] — it never emits anything for the
    modern `tool_calls` field, so a scripted message carrying tool_calls
    (with empty content) streams zero chunks and the v3 streaming path
    fails with "stream finished without producing a message". This
    override always yields exactly one chunk per call, carrying
    tool_call_chunks when present, using a single next() call (delegating
    to super()._stream() for the no-tool-calls case would call next()
    again internally and double-consume the scripted message iterator)."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = next(self.messages)
        message_ = AIMessage(content=message) if isinstance(message, str) else message
        tool_call_chunks = [
            {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
            for i, tc in enumerate(message_.tool_calls)
        ] if message_.tool_calls else []
        yield ChatGenerationChunk(message=AIMessageChunk(
            content=message_.content,
            tool_call_chunks=tool_call_chunks,
        ))


@tool
def get_naming_convention() -> str:
    """Return the PureStorage volume naming convention."""
    return "prod-erp-data-500g"


def make_app(model, tools):
    app = FastAPI()
    app.include_router(router)
    app.state.settings = type("S", (), {"API_TOKEN": "dev-token"})()
    app.state.agent = build_agent(
        model=model,
        tools=tools,
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


def test_websocket_rejects_wrong_token(db):
    app = make_app(GenericFakeChatModel(messages=iter([])), [])
    app.state.db = db
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"token": "wrong-token"})
        reply = ws.receive_json()
        assert reply == {"type": "error", "message": "unauthorized"}


def test_websocket_rejects_malformed_first_frame(db):
    app = make_app(GenericFakeChatModel(messages=iter([])), [])
    app.state.db = db
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text("not json at all")
        reply = ws.receive_json()
        assert reply == {"type": "error", "message": "unauthorized"}


def test_websocket_rejects_empty_token_even_if_configured_token_is_empty(db):
    # A misconfigured empty API_TOKEN must never authenticate a client that
    # sends an empty token to match it (same fail-open class as auth.py).
    app = make_app(GenericFakeChatModel(messages=iter([])), [])
    app.state.settings = type("S", (), {"API_TOKEN": ""})()
    app.state.db = db
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"token": ""})
        reply = ws.receive_json()
        assert reply == {"type": "error", "message": "unauthorized"}


def test_websocket_streams_tool_call_and_final_answer(db):
    model = FakeToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[
            ToolCall(name="get_naming_convention", args={}, id="call_1")
        ]),
        AIMessage(content="It's prod-erp-data-500g."),
    ]))
    app = make_app(model, [get_naming_convention])
    app.state.db = db
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"token": "dev-token"})
        ws.send_json({"type": "message", "conversation_id": None, "content": "What's the naming convention?"})

        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("final", "error"):
                break

    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert types[-1] == "final"
    assert events[-1]["content"] == "It's prod-erp-data-500g."
    assert events[-1]["conversation_id"]


def test_websocket_reports_agent_failure_as_error_without_crashing(db):
    class FailingModel:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("LLM unreachable")

        async def astream_events(self, *args, **kwargs):
            raise RuntimeError("LLM unreachable")

    app = FastAPI()
    app.include_router(router)
    app.state.settings = type("S", (), {"API_TOKEN": "dev-token"})()
    app.state.agent = FailingModel()
    app.state.db = db
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"token": "dev-token"})
        ws.send_json({"type": "message", "conversation_id": None, "content": "hi"})
        reply = ws.receive_json()

        assert reply["type"] == "error"
        # the raw exception text ("LLM unreachable") must not reach the client
        assert reply["message"] == "internal error"
        # connection must still be open for a second message
        ws.send_json({"type": "message", "conversation_id": None, "content": "hi again"})
        reply2 = ws.receive_json()
        assert reply2["type"] == "error"
