import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    input TEXT NOT NULL,
    result TEXT,
    error TEXT,
    conversation_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
"""

_ALLOWED_TASK_FIELDS = {"status", "result", "error", "started_at", "finished_at"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def init_db(db_path: Path) -> aiosqlite.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await db.executescript(SCHEMA)
    await db.commit()
    return db


async def create_conversation(db: aiosqlite.Connection, source: str) -> str:
    conversation_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO conversations (id, source, created_at) VALUES (?, ?, ?)",
        (conversation_id, source, _now()),
    )
    await db.commit()
    return conversation_id


async def create_task(db: aiosqlite.Connection, conversation_id: str, input_text: str) -> str:
    task_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO tasks (id, status, input, conversation_id, created_at) VALUES (?, 'pending', ?, ?, ?)",
        (task_id, input_text, conversation_id, _now()),
    )
    await db.commit()
    return task_id


async def update_task(db: aiosqlite.Connection, task_id: str, **fields: Any) -> None:
    if not fields:
        return
    unknown = set(fields) - _ALLOWED_TASK_FIELDS
    if unknown:
        raise ValueError(f"unknown task field(s): {unknown}")

    columns = ", ".join(f"{key} = ?" for key in fields)
    await db.execute(f"UPDATE tasks SET {columns} WHERE id = ?", (*fields.values(), task_id))
    await db.commit()


async def get_task(db: aiosqlite.Connection, task_id: str) -> dict[str, Any] | None:
    cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None
