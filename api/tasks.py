import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request

from api.schemas import CreateTaskRequest, TaskResponse
from auth import parse_bearer_token, verify_token
from storage.db import create_conversation, create_task, get_task, update_task

router = APIRouter()

# asyncio.create_task()'s docs warn the event loop only holds a weak
# reference to a task: without a strong reference kept somewhere, a task can
# be garbage-collected mid-run. This set is that strong reference.
_background_tasks: set[asyncio.Task] = set()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_token(request: Request, authorization: str | None) -> None:
    token = parse_bearer_token(authorization)
    if not verify_token(token, request.app.state.settings.API_TOKEN):
        raise HTTPException(status_code=401, detail="unauthorized")


@router.post("/tasks", status_code=202, response_model=TaskResponse)
async def post_task(
    body: CreateTaskRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> TaskResponse:
    _require_token(request, authorization)

    db = request.app.state.db
    conversation_id = await create_conversation(db, source="task")
    task_id = await create_task(db, conversation_id, body.input)

    task = asyncio.create_task(_run_task(request.app, task_id, conversation_id, body.input))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return TaskResponse(task_id=task_id, status="pending")


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> TaskResponse:
    _require_token(request, authorization)

    db = request.app.state.db
    task = await get_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="not found")

    return TaskResponse(
        task_id=task["id"],
        **{k: v for k, v in task.items() if k not in ("id", "conversation_id")},
    )


async def _run_task(app, task_id: str, conversation_id: str, input_text: str) -> None:
    db = app.state.db
    await update_task(db, task_id, status="running", started_at=_now())

    try:
        result = await app.state.agent.ainvoke(
            {"messages": [{"role": "user", "content": input_text}]},
            config={"configurable": {"thread_id": conversation_id}},
        )
        final_content = result["messages"][-1].content
        await update_task(db, task_id, status="completed", result=final_content, finished_at=_now())
    except Exception as exc:
        await update_task(db, task_id, status="failed", error=str(exc), finished_at=_now())
