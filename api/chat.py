import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket

from storage.db import create_conversation

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        first_frame = json.loads(await websocket.receive_text())
        token = first_frame.get("token")
    except json.JSONDecodeError:
        token = None

    if token != websocket.app.state.settings.API_TOKEN:
        await websocket.send_json({"type": "error", "message": "unauthorized"})
        await websocket.close(code=4401)
        return

    agent = websocket.app.state.agent
    db = websocket.app.state.db

    while True:
        try:
            raw = await websocket.receive_text()
        except Exception:
            return

        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "message": "invalid message"})
            continue

        conversation_id = message.get("conversation_id")
        if conversation_id is None:
            conversation_id = await create_conversation(db, source="webview")

        try:
            await _run_turn(agent, message["content"], conversation_id, websocket)
        except Exception as exc:
            logger.warning("agent run failed", exc_info=True)
            await websocket.send_json({"type": "error", "message": str(exc)})


async def _run_turn(agent, content: str, conversation_id: str, websocket: WebSocket) -> None:
    config = {"configurable": {"thread_id": conversation_id}}
    stream = await agent.astream_events(
        {"messages": [{"role": "user", "content": content}]},
        config=config,
        version="v3",
    )

    final_content = ""

    async def consume_tool_calls() -> None:
        async for call in stream.tool_calls:
            await websocket.send_json({
                "type": "tool_call",
                "name": call.tool_name,
                "input": call.input,
            })
            # .output is only populated once output_deltas has been drained,
            # even when there are no deltas to yield.
            async for _ in call.output_deltas:
                pass
            output = call.output
            await websocket.send_json({
                "type": "tool_result",
                "name": call.tool_name,
                "output": str(output.content) if output is not None else None,
            })

    async def consume_messages() -> None:
        nonlocal final_content
        async for message in stream.messages:
            final_content = await message.text

    await asyncio.gather(consume_tool_calls(), consume_messages())

    await websocket.send_json({
        "type": "final",
        "content": final_content,
        "conversation_id": conversation_id,
    })
