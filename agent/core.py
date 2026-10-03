from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)


class DuplicateToolName(Exception):
    pass


def build_agent(
    model,
    tools,
    system_prompt,
    max_tool_turns,
    checkpointer,
    summarize_trigger_tokens,
    summarize_keep_messages,
):
    seen = set()
    for t in tools:
        if t.name in seen:
            raise DuplicateToolName(f"duplicate tool name: {t.name!r}")
        seen.add(t.name)

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=[
            # on_failure="error" so an exhausted retry fails the run (a /tasks
            # task is recorded as failed) instead of returning the error as an
            # ordinary assistant reply.
            ModelRetryMiddleware(on_failure="error"),
            ToolErrorMiddleware(on_error=lambda exc, request: f"Error: {exc}"),
            ToolCallLimitMiddleware(run_limit=max_tool_turns, exit_behavior="error"),
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", summarize_trigger_tokens),
                keep=("messages", summarize_keep_messages),
            ),
        ],
    )
