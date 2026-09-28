from pydantic import BaseModel


class CreateTaskRequest(BaseModel):
    input: str


class TaskResponse(BaseModel):
    task_id: str
    status: str
    input: str | None = None
    result: str | None = None
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
