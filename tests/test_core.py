import pytest
from langchain.messages import AIMessage, ToolCall
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver

from agent.core import DuplicateToolName, build_agent


class FakeToolCallingModel(GenericFakeChatModel):
    """GenericFakeChatModel doesn't implement bind_tools() (raises
    NotImplementedError), but create_agent() always calls it. Since this
    fake model already decides its output from a scripted list regardless
    of which tools are bound, binding is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


@tool
def echo(text: str) -> str:
    """Echo the given text back."""
    return text


@tool
def boom() -> str:
    """Always raises."""
    raise ValueError("boom")


def test_build_agent_raises_on_duplicate_tool_name():
    @tool
    def duplicate_name() -> str:
        """A tool."""
        return "a"

    duplicate_name.name = "echo"

    with pytest.raises(DuplicateToolName):
        build_agent(
            model=GenericFakeChatModel(messages=iter([])),
            tools=[echo, duplicate_name],
            system_prompt="test",
            max_tool_turns=20,
            checkpointer=InMemorySaver(),
            summarize_trigger_tokens=4000,
            summarize_keep_messages=20,
        )


async def test_build_agent_runs_a_tool_and_returns_final_answer():
    model = FakeToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[ToolCall(name="echo", args={"text": "hi"}, id="call_1")]),
        AIMessage(content="done"),
    ]))
    agent = build_agent(
        model=model,
        tools=[echo],
        system_prompt="test",
        max_tool_turns=20,
        checkpointer=InMemorySaver(),
        summarize_trigger_tokens=4000,
        summarize_keep_messages=20,
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "say hi"}]},
        config={"configurable": {"thread_id": "t1"}},
    )

    assert result["messages"][-1].content == "done"
    tool_messages = [m for m in result["messages"] if getattr(m, "name", None) == "echo"]
    assert tool_messages[0].content == "hi"


async def test_build_agent_converts_tool_exception_to_error_message_and_continues():
    model = FakeToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[ToolCall(name="boom", args={}, id="call_1")]),
        AIMessage(content="acknowledged"),
    ]))
    agent = build_agent(
        model=model,
        tools=[boom],
        system_prompt="test",
        max_tool_turns=20,
        checkpointer=InMemorySaver(),
        summarize_trigger_tokens=4000,
        summarize_keep_messages=20,
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "do it"}]},
        config={"configurable": {"thread_id": "t2"}},
    )

    tool_messages = [m for m in result["messages"] if getattr(m, "name", None) == "boom"]
    assert "Error: boom" in tool_messages[0].content
    assert result["messages"][-1].content == "acknowledged"


async def test_build_agent_enforces_tool_call_limit():
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError

    def always_call_echo():
        while True:
            yield AIMessage(content="", tool_calls=[ToolCall(name="echo", args={"text": "x"}, id="call")])

    model = FakeToolCallingModel(messages=always_call_echo())
    agent = build_agent(
        model=model,
        tools=[echo],
        system_prompt="test",
        max_tool_turns=1,
        checkpointer=InMemorySaver(),
        summarize_trigger_tokens=4000,
        summarize_keep_messages=20,
    )

    with pytest.raises(ToolCallLimitExceededError):
        await agent.ainvoke(
            {"messages": [{"role": "user", "content": "loop forever"}]},
            config={"configurable": {"thread_id": "t3"}},
        )


async def test_sqlite_checkpointer_persists_conversation_across_separate_invocations(tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = tmp_path / "checkpoints.db"
    model = FakeToolCallingModel(messages=iter([
        AIMessage(content="Hi Bob!"),
        AIMessage(content="Your name is Bob."),
    ]))

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        await checkpointer.setup()
        agent = build_agent(
            model=model,
            tools=[],
            system_prompt="test",
            max_tool_turns=20,
            checkpointer=checkpointer,
            summarize_trigger_tokens=4000,
            summarize_keep_messages=20,
        )
        config = {"configurable": {"thread_id": "persist-test"}}

        await agent.ainvoke({"messages": [{"role": "user", "content": "My name is Bob."}]}, config=config)
        await agent.ainvoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config=config)

        state = await agent.aget_state(config)
        assert len(state.values["messages"]) == 4
