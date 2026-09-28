import pytest

from storage.db import create_conversation, create_task, get_task, init_db, update_task


async def test_init_db_creates_tables(tmp_path):
    db = await init_db(tmp_path / "test.db")
    try:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert {"conversations", "tasks"} <= tables
    finally:
        await db.close()


async def test_create_conversation_inserts_row(tmp_path):
    db = await init_db(tmp_path / "test.db")
    try:
        conversation_id = await create_conversation(db, source="webview")
        cursor = await db.execute("SELECT source FROM conversations WHERE id = ?", (conversation_id,))
        row = await cursor.fetchone()
        assert row[0] == "webview"
    finally:
        await db.close()


async def test_create_task_defaults_to_pending(tmp_path):
    db = await init_db(tmp_path / "test.db")
    try:
        conversation_id = await create_conversation(db, source="task")
        task_id = await create_task(db, conversation_id, "do something")

        task = await get_task(db, task_id)

        assert task["status"] == "pending"
        assert task["input"] == "do something"
        assert task["conversation_id"] == conversation_id
        assert task["result"] is None
        assert task["error"] is None
    finally:
        await db.close()


async def test_update_task_sets_fields(tmp_path):
    db = await init_db(tmp_path / "test.db")
    try:
        conversation_id = await create_conversation(db, source="task")
        task_id = await create_task(db, conversation_id, "do something")

        await update_task(db, task_id, status="completed", result="done", finished_at="2026-01-01T00:00:00")

        task = await get_task(db, task_id)
        assert task["status"] == "completed"
        assert task["result"] == "done"
        assert task["finished_at"] == "2026-01-01T00:00:00"
    finally:
        await db.close()


async def test_update_task_rejects_unknown_field(tmp_path):
    db = await init_db(tmp_path / "test.db")
    try:
        conversation_id = await create_conversation(db, source="task")
        task_id = await create_task(db, conversation_id, "do something")

        with pytest.raises(ValueError):
            await update_task(db, task_id, not_a_real_column="x")
    finally:
        await db.close()


async def test_get_task_returns_none_for_unknown_id(tmp_path):
    db = await init_db(tmp_path / "test.db")
    try:
        assert await get_task(db, "no-such-id") is None
    finally:
        await db.close()
