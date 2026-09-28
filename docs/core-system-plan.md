# Core System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build storage-agent's core system from scratch — sandboxed file tools, an allowlisted REST tool, an MCP client, a LangChain agent core with persistence, a chat WebSocket, and a trigger REST API — as a single runnable, tested FastAPI app.

**Architecture:** One FastAPI process. Every capability (file/REST/MCP tools) is a LangChain `BaseTool`; `agent/core.py::build_agent()` is the one place that assembles them and hands them to `langchain.agents.create_agent()`. SQLite holds `conversations`/`tasks` bookkeeping; conversation message history is a LangGraph checkpointer, not a hand-rolled table. Both the WebSocket and the trigger API drive the same built-once agent.

**Tech Stack:** Python 3.12, `uv`, FastAPI, `langchain` + `langchain-openai` + `langchain[mcp]` (beta) + `langgraph-checkpoint-sqlite`, `aiosqlite`, `httpx`, `pytest`/`pytest-asyncio`/`respx`.

**Spec:** `docs/storage-agent-spec.md`, Part 1 ("Core System Design"). This plan implements that part; Part 3 (skills) is a separate, already-written plan to execute afterward on top of this one.

## Global Constraints

- Python 3.12 via `uv`; every dependency added with `uv add` (or `uv add --dev`), never hand-edited into `pyproject.toml`.
- Sandbox path safety (`resolve_in_sandbox`): reject if the incoming path is absolute or contains `".."`; resolve with `(sandbox_root / path).resolve(strict=False)`; reject unless the result `is_relative_to(sandbox_root)` — this is what defeats a symlink inside the sandbox pointing outside it.
- REST allowlist: only an API config's declared `operations` become callable tools; `auth_type` is one of `none`/`bearer`/`api_key_header`/`basic`; `auth_value` supports `${ENV_VAR}` interpolation.
- MCP: one `langchain.mcp.MCPAdapter` per configured server (`langchain[mcp]>=1.4.0`, **beta** — pin the version); each server connection is attempted independently, a failing one is logged and skipped, never blocking the others or the app.
- MCP TLS verification: the spec flags a prior deployment's need to bypass TLS verification for `streamable_http`/`sse` servers with a broken OS trust store. **Deliberately not implemented here (YAGNI)** — this is a fresh build with no known broken deployment machine and no remote MCP server configured yet; default `verify=True` behavior is kept. If a real `CERTIFICATE_VERIFY_FAILED` case shows up against an actual remote server, that's the point to revisit `MCPAdapter`'s transport-level verification hook, not before.
- Agent (`agent/core.py::build_agent`): `ToolErrorMiddleware(on_error=lambda exc, request: f"Error: {exc}")` converts every tool exception to a string; `ToolCallLimitMiddleware(run_limit=settings.MAX_TOOL_TURNS, exit_behavior="error")` (default 20) caps tool-calling turns per message; `SummarizationMiddleware(trigger=("tokens", settings.SUMMARIZE_TRIGGER_TOKENS), keep=("messages", settings.SUMMARIZE_KEEP_MESSAGES))` (defaults 4000/20) bounds prompt growth; a `DuplicateToolName` check runs before `create_agent()` is called.
- Conversation history is a LangGraph checkpointer keyed by `thread_id=conversation_id` — no hand-rolled `messages` table.
- Auth: one shared bearer token (`settings.API_TOKEN`) on both the WebSocket's first frame (`{"token": "..."}`) and the trigger API's `Authorization: Bearer <token>` header.
- WebSocket frames are exactly `{"type": "tool_call"|"tool_result"|"final"|"error", ...}`.
- SQLite schema is only `conversations(id, source, created_at)` and `tasks(id, status, input, result, error, conversation_id, created_at, started_at, finished_at)`.
- `astream_events(..., version="v3")` is itself a **beta** LangGraph API (per its 1.2.0 changelog) — the exact attribute names on its projections are verified empirically in Task 9, not assumed.
- Env vars (`settings.py`): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `SANDBOX_ROOT`, `API_TOKEN`, `DB_PATH`, `SYSTEM_PROMPT_PATH`, `API_ALLOWLIST_PATH`, `MCP_SERVERS_PATH`, `SKILLS_PATH`, `MAX_TOOL_TURNS` (20), `SUMMARIZE_TRIGGER_TOKENS` (4000), `SUMMARIZE_KEEP_MESSAGES` (20), `LOG_LEVEL`.

## Review Focus

- Exceeding `MAX_TOOL_TURNS` mid-conversation must stop the run cleanly (raise, not hang), and the caller (WebSocket/task) must turn that into a clean error, not a crash.
- One misconfigured or unreachable MCP server must not prevent other configured servers — or the app itself — from starting.
- A REST tool's non-2xx HTTP response must propagate as an exception for `ToolErrorMiddleware` to catch, never be silently swallowed or returned as if it were a valid result.
- A WebSocket client that sends a bad first frame, skips the token handshake, or triggers an unreachable-LLM error mid-turn must be handled cleanly — the connection must not die uncaught.
- A background task's own exception (agent failure, anything) must always be caught and recorded as `status="failed"` with `error` populated — never stuck at `"running"` forever, never crashing the server process.

---

### Task 1: Settings

**Files:**
- Create: `settings.py`
- Create: `.env.example`
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: nothing new (`pydantic-settings`).
- Produces: `Settings` (a `pydantic_settings.BaseSettings` subclass) with every field listed in Global Constraints above. Every later task that touches config imports this.

- [ ] **Step 1: Add dependencies**

Run: `uv add fastapi "uvicorn[standard]" pydantic pydantic-settings aiosqlite httpx`
Run: `uv add --dev pytest pytest-asyncio respx`

Add to `pyproject.toml` under a new `[tool.pytest.ini_options]` section (append, don't remove anything `uv init` generated):

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_settings.py`:

```python
import pytest

from settings import Settings


def test_settings_loads_required_fields(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("API_TOKEN", "dev-token")

    settings = Settings(_env_file=None)

    assert settings.LLM_BASE_URL == "https://api.groq.com/openai/v1"
    assert settings.LLM_API_KEY == "test-key"
    assert settings.LLM_MODEL == "llama-3.3-70b-versatile"
    assert settings.API_TOKEN == "dev-token"


def test_settings_applies_defaults(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("API_TOKEN", "dev-token")

    settings = Settings(_env_file=None)

    assert settings.SANDBOX_ROOT == "./data/sandbox"
    assert settings.DB_PATH == "./data/storage_agent.db"
    assert settings.SYSTEM_PROMPT_PATH == "./config/system_prompt.md"
    assert settings.API_ALLOWLIST_PATH == "./config/api_allowlist.json"
    assert settings.MCP_SERVERS_PATH == "./config/mcp_servers.json"
    assert settings.SKILLS_PATH == "./skills"
    assert settings.MAX_TOOL_TURNS == 20
    assert settings.SUMMARIZE_TRIGGER_TOKENS == 4000
    assert settings.SUMMARIZE_KEEP_MESSAGES == 20
    assert settings.LOG_LEVEL == "INFO"


def test_settings_raises_when_required_field_missing(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("API_TOKEN", raising=False)

    with pytest.raises(Exception):
        Settings(_env_file=None)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'settings'`

- [ ] **Step 4: Write the implementation**

Create `settings.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_BASE_URL: str
    LLM_API_KEY: str
    LLM_MODEL: str
    API_TOKEN: str

    SANDBOX_ROOT: str = "./data/sandbox"
    DB_PATH: str = "./data/storage_agent.db"
    SYSTEM_PROMPT_PATH: str = "./config/system_prompt.md"
    API_ALLOWLIST_PATH: str = "./config/api_allowlist.json"
    MCP_SERVERS_PATH: str = "./config/mcp_servers.json"
    SKILLS_PATH: str = "./skills"

    MAX_TOOL_TURNS: int = 20
    SUMMARIZE_TRIGGER_TOKENS: int = 4000
    SUMMARIZE_KEEP_MESSAGES: int = 20

    LOG_LEVEL: str = "INFO"
```

Create `.env.example`:

```bash
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=changeme
LLM_MODEL=llama-3.3-70b-versatile
API_TOKEN=changeme
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_settings.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add settings.py .env.example pyproject.toml uv.lock tests/test_settings.py
git commit -m "Add Settings (pydantic-settings)"
```

---

### Task 2: Auth helpers

**Files:**
- Create: `auth.py`
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `parse_bearer_token(header: str | None) -> str | None`, `verify_token(token: str | None, expected: str) -> bool`. Tasks 9 and 10 import both.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_auth.py`:

```python
from auth import parse_bearer_token, verify_token


def test_parse_bearer_token_extracts_token():
    assert parse_bearer_token("Bearer abc123") == "abc123"


def test_parse_bearer_token_case_insensitive_scheme():
    assert parse_bearer_token("bearer abc123") == "abc123"


def test_parse_bearer_token_returns_none_for_missing_header():
    assert parse_bearer_token(None) is None


def test_parse_bearer_token_returns_none_for_wrong_scheme():
    assert parse_bearer_token("Basic abc123") is None


def test_parse_bearer_token_returns_none_for_malformed_header():
    assert parse_bearer_token("abc123") is None


def test_verify_token_accepts_matching_token():
    assert verify_token("abc123", "abc123") is True


def test_verify_token_rejects_mismatched_token():
    assert verify_token("wrong", "abc123") is False


def test_verify_token_rejects_none():
    assert verify_token(None, "abc123") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_auth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auth'`

- [ ] **Step 3: Write the implementation**

Create `auth.py`:

```python
def parse_bearer_token(header: str | None) -> str | None:
    if not header:
        return None
    parts = header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def verify_token(token: str | None, expected: str) -> bool:
    return token is not None and token == expected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_auth.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add auth.py tests/test_auth.py
git commit -m "Add bearer token auth helpers"
```

---

### Task 3: Sandboxed file tools

**Files:**
- Create: `agent/__init__.py`, `agent/tools/__init__.py`, `agent/tools/files.py`
- Test: `tests/test_sandbox_traversal.py`, `tests/test_files_tool.py`

**Interfaces:**
- Consumes: `langchain.tools.tool`.
- Produces: `resolve_in_sandbox(sandbox_root: Path, user_path: str) -> Path`, `SandboxViolation` (Exception), `build_file_tools(sandbox_root: Path) -> list[BaseTool]` (tools: `list_dir`, `search_files`, `read_file`, `write_file`, `edit_file`, `delete_file`, `move_file`). Task 12 imports `build_file_tools`; Task 4 of the (separate, already-written) skills plan imports `resolve_in_sandbox`.

- [ ] **Step 1: Add the `langchain` dependency**

Run: `uv add langchain langchain-openai`

- [ ] **Step 2: Write the failing sandbox-traversal tests**

Create `tests/test_sandbox_traversal.py`:

```python
import pytest

from agent.tools.files import SandboxViolation, resolve_in_sandbox


def test_resolve_in_sandbox_allows_normal_relative_path(tmp_path):
    result = resolve_in_sandbox(tmp_path, "notes.txt")
    assert result == tmp_path / "notes.txt"


def test_resolve_in_sandbox_rejects_absolute_path(tmp_path):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(tmp_path, "/etc/passwd")


def test_resolve_in_sandbox_rejects_dotdot(tmp_path):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(tmp_path, "../outside.txt")


def test_resolve_in_sandbox_rejects_dotdot_in_middle(tmp_path):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(tmp_path, "subdir/../../outside.txt")


def test_resolve_in_sandbox_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("secret")

    escape_link = tmp_path / "escape"
    escape_link.symlink_to(outside)

    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(tmp_path, "escape/secret.txt")


def test_resolve_in_sandbox_allows_new_file_that_does_not_exist_yet(tmp_path):
    result = resolve_in_sandbox(tmp_path, "new/nested/file.txt")
    assert result == tmp_path / "new" / "nested" / "file.txt"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_sandbox_traversal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 4: Write `resolve_in_sandbox`**

Create `agent/__init__.py` (empty) and `agent/tools/__init__.py` (empty).

Create `agent/tools/files.py`:

```python
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool


class SandboxViolation(Exception):
    pass


def resolve_in_sandbox(sandbox_root: Path, user_path: str) -> Path:
    candidate = Path(user_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SandboxViolation(f"path escapes sandbox: {user_path!r}")

    resolved = (sandbox_root / candidate).resolve(strict=False)
    if not resolved.is_relative_to(sandbox_root):
        raise SandboxViolation(f"path escapes sandbox: {user_path!r}")

    return resolved
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_sandbox_traversal.py -v`
Expected: 6 passed

- [ ] **Step 6: Write the failing file-tool tests**

Create `tests/test_files_tool.py`:

```python
import pytest

from agent.tools.files import build_file_tools


def make_tools(tmp_path):
    return {t.name: t for t in build_file_tools(tmp_path)}


async def test_write_then_read_round_trip(tmp_path):
    tools = make_tools(tmp_path)

    await tools["write_file"].ainvoke({"path": "notes.txt", "content": "hello"})
    result = await tools["read_file"].ainvoke({"path": "notes.txt"})

    assert result == "hello"


async def test_write_file_creates_parent_directories(tmp_path):
    tools = make_tools(tmp_path)

    await tools["write_file"].ainvoke({"path": "a/b/c.txt", "content": "deep"})

    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


async def test_read_file_missing_raises(tmp_path):
    tools = make_tools(tmp_path)

    with pytest.raises(FileNotFoundError):
        await tools["read_file"].ainvoke({"path": "missing.txt"})


async def test_list_dir_lists_entries(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    tools = make_tools(tmp_path)

    result = await tools["list_dir"].ainvoke({"path": "."})

    assert "a.txt" in result
    assert "sub/" in result


async def test_list_dir_empty_directory(tmp_path):
    tools = make_tools(tmp_path)

    result = await tools["list_dir"].ainvoke({"path": "."})

    assert result == "(empty directory)"


async def test_search_files_matches_glob(tmp_path):
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    tools = make_tools(tmp_path)

    result = await tools["search_files"].ainvoke({"pattern": "*.md", "path": "."})

    assert "a.md" in result
    assert "b.txt" not in result


async def test_edit_file_replaces_first_occurrence(tmp_path):
    (tmp_path / "f.txt").write_text("foo bar foo")
    tools = make_tools(tmp_path)

    await tools["edit_file"].ainvoke({"path": "f.txt", "old_text": "foo", "new_text": "baz"})

    assert (tmp_path / "f.txt").read_text() == "baz bar foo"


async def test_edit_file_raises_when_text_not_found(tmp_path):
    (tmp_path / "f.txt").write_text("foo")
    tools = make_tools(tmp_path)

    with pytest.raises(ValueError):
        await tools["edit_file"].ainvoke({"path": "f.txt", "old_text": "missing", "new_text": "x"})


async def test_delete_file_removes_file(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    tools = make_tools(tmp_path)

    await tools["delete_file"].ainvoke({"path": "f.txt"})

    assert not (tmp_path / "f.txt").exists()


async def test_move_file_renames(tmp_path):
    (tmp_path / "src.txt").write_text("x")
    tools = make_tools(tmp_path)

    await tools["move_file"].ainvoke({"src": "src.txt", "dest": "dest.txt"})

    assert not (tmp_path / "src.txt").exists()
    assert (tmp_path / "dest.txt").read_text() == "x"


async def test_move_file_missing_source_raises(tmp_path):
    tools = make_tools(tmp_path)

    with pytest.raises(FileNotFoundError):
        await tools["move_file"].ainvoke({"src": "missing.txt", "dest": "dest.txt"})


def test_build_file_tools_returns_all_seven_tools(tmp_path):
    names = {t.name for t in build_file_tools(tmp_path)}
    assert names == {
        "list_dir", "search_files", "read_file",
        "write_file", "edit_file", "delete_file", "move_file",
    }
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `uv run pytest tests/test_files_tool.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_file_tools'`

- [ ] **Step 8: Write `build_file_tools`**

Append to `agent/tools/files.py`:

```python
def build_file_tools(sandbox_root: Path) -> list[BaseTool]:
    async def list_dir(path: str = ".") -> str:
        """List files and directories at a path relative to the sandbox root."""
        target = resolve_in_sandbox(sandbox_root, path)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return "\n".join(entries) if entries else "(empty directory)"

    async def search_files(pattern: str, path: str = ".") -> str:
        """Search for files matching a glob pattern under a path relative to the sandbox root."""
        target = resolve_in_sandbox(sandbox_root, path)
        matches = sorted(
            str(p.relative_to(sandbox_root)) for p in target.rglob(pattern) if p.is_file()
        )
        return "\n".join(matches) if matches else "(no matches)"

    async def read_file(path: str) -> str:
        """Read a file's full text content."""
        target = resolve_in_sandbox(sandbox_root, path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path}")
        return target.read_text()

    async def write_file(path: str, content: str) -> str:
        """Create or overwrite a file with the given text content."""
        target = resolve_in_sandbox(sandbox_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    async def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace the first occurrence of old_text with new_text in a file."""
        target = resolve_in_sandbox(sandbox_root, path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path}")
        content = target.read_text()
        if old_text not in content:
            raise ValueError(f"text not found in {path}: {old_text!r}")
        target.write_text(content.replace(old_text, new_text, 1))
        return f"edited {path}"

    async def delete_file(path: str) -> str:
        """Delete a file."""
        target = resolve_in_sandbox(sandbox_root, path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path}")
        target.unlink()
        return f"deleted {path}"

    async def move_file(src: str, dest: str) -> str:
        """Move or rename a file."""
        src_target = resolve_in_sandbox(sandbox_root, src)
        dest_target = resolve_in_sandbox(sandbox_root, dest)
        if not src_target.is_file():
            raise FileNotFoundError(f"not a file: {src}")
        dest_target.parent.mkdir(parents=True, exist_ok=True)
        src_target.rename(dest_target)
        return f"moved {src} to {dest}"

    return [
        tool(list_dir),
        tool(search_files),
        tool(read_file),
        tool(write_file),
        tool(edit_file),
        tool(delete_file),
        tool(move_file),
    ]
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/test_files_tool.py -v`
Expected: 12 passed

- [ ] **Step 10: Commit**

```bash
git add agent/ pyproject.toml uv.lock tests/test_sandbox_traversal.py tests/test_files_tool.py
git commit -m "Add sandboxed file tools"
```

---

### Task 4: REST allowlist tools

**Files:**
- Create: `agent/tools/rest.py`, `config/api_allowlist.json`
- Test: `tests/test_rest_tool.py`

**Interfaces:**
- Consumes: `langchain.tools.tool` (Task 3's import pattern).
- Produces: `load_api_configs(path: Path) -> list[dict]`, `build_rest_tools(configs: list[dict], stack: AsyncExitStack) -> list[BaseTool]` (async — it opens HTTP clients). Task 12 imports both.

- [ ] **Step 1: Add the `respx` dependency (test-only, already added in Task 1) and create the empty allowlist config**

Create `config/api_allowlist.json`:

```json
[]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_rest_tool.py`:

```python
import json
from contextlib import AsyncExitStack

import httpx
import pytest
import respx

from agent.tools.rest import build_rest_tools, load_api_configs

VOLUMES_CONFIG = [
    {
        "name": "purestorage",
        "description": "PureStorage FlashArray API",
        "base_url": "https://flasharray.example.com",
        "auth_type": "bearer",
        "auth_value": "${TEST_PURESTORAGE_TOKEN}",
        "operations": [
            {
                "name": "list_volumes",
                "method": "GET",
                "path": "/api/2.x/volumes",
                "description": "List volumes.",
                "params_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_volume",
                "method": "GET",
                "path": "/api/2.x/volumes/{name}",
                "description": "Get a volume by name.",
                "params_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "create_volume",
                "method": "POST",
                "path": "/api/2.x/volumes",
                "description": "Create a volume.",
                "params_schema": {
                    "type": "object",
                    "properties": {
                        "names": {"type": "array", "items": {"type": "string"}},
                        "provisioned": {"type": "integer"},
                    },
                    "required": ["names", "provisioned"],
                },
            },
        ],
    }
]


def test_load_api_configs_returns_empty_list_for_missing_file(tmp_path):
    assert load_api_configs(tmp_path / "does-not-exist.json") == []


def test_load_api_configs_interpolates_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PURESTORAGE_TOKEN", "secret-token")
    config_path = tmp_path / "api_allowlist.json"
    config_path.write_text(json.dumps(VOLUMES_CONFIG))

    configs = load_api_configs(config_path)

    assert configs[0]["auth_value"] == "secret-token"


async def test_build_rest_tools_returns_one_tool_per_operation(monkeypatch):
    monkeypatch.setenv("TEST_PURESTORAGE_TOKEN", "secret-token")
    async with AsyncExitStack() as stack:
        configs = json.loads(json.dumps(VOLUMES_CONFIG))
        configs[0]["auth_value"] = "secret-token"
        tools = await build_rest_tools(configs, stack)

        names = {t.name for t in tools}
        assert names == {
            "purestorage_list_volumes",
            "purestorage_get_volume",
            "purestorage_create_volume",
        }


@respx.mock
async def test_get_operation_substitutes_path_param_and_sends_bearer_auth():
    route = respx.get("https://flasharray.example.com/api/2.x/volumes/prod-erp-data").mock(
        return_value=httpx.Response(200, json={"name": "prod-erp-data"})
    )
    configs = json.loads(json.dumps(VOLUMES_CONFIG))
    configs[0]["auth_value"] = "secret-token"

    async with AsyncExitStack() as stack:
        tools = {t.name: t for t in await build_rest_tools(configs, stack)}
        result = await tools["purestorage_get_volume"].ainvoke({"name": "prod-erp-data"})

    assert json.loads(result) == {"name": "prod-erp-data"}
    assert route.calls[0].request.headers["Authorization"] == "Bearer secret-token"


@respx.mock
async def test_post_operation_sends_remaining_args_as_json_body():
    route = respx.post("https://flasharray.example.com/api/2.x/volumes").mock(
        return_value=httpx.Response(201, json={"created": True})
    )
    configs = json.loads(json.dumps(VOLUMES_CONFIG))
    configs[0]["auth_value"] = "secret-token"

    async with AsyncExitStack() as stack:
        tools = {t.name: t for t in await build_rest_tools(configs, stack)}
        await tools["purestorage_create_volume"].ainvoke(
            {"names": ["prod-erp-data-500g"], "provisioned": 500_000_000_000}
        )

    body = json.loads(route.calls[0].request.content)
    assert body == {"names": ["prod-erp-data-500g"], "provisioned": 500_000_000_000}


@respx.mock
async def test_non_2xx_response_raises():
    respx.get("https://flasharray.example.com/api/2.x/volumes").mock(
        return_value=httpx.Response(500, text="internal error")
    )
    configs = json.loads(json.dumps(VOLUMES_CONFIG))
    configs[0]["auth_value"] = "secret-token"

    async with AsyncExitStack() as stack:
        tools = {t.name: t for t in await build_rest_tools(configs, stack)}
        with pytest.raises(httpx.HTTPStatusError):
            await tools["purestorage_list_volumes"].ainvoke({})


async def test_builds_one_http_client_per_config_shared_across_operations(monkeypatch):
    monkeypatch.setenv("TEST_PURESTORAGE_TOKEN", "secret-token")
    configs = json.loads(json.dumps(VOLUMES_CONFIG))
    configs[0]["auth_value"] = "secret-token"

    calls = []
    original_init = httpx.AsyncClient.__init__

    def spy_init(self, *args, **kwargs):
        calls.append(1)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", spy_init)

    async with AsyncExitStack() as stack:
        await build_rest_tools(configs, stack)

    assert len(calls) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_rest_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.tools.rest'`

- [ ] **Step 4: Write the implementation**

Create `agent/tools/rest.py`:

```python
import base64
import json
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import httpx
from langchain.tools import tool
from langchain_core.tools import BaseTool

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
_PATH_PARAM_PATTERN = re.compile(r"\{(\w+)\}")


def _interpolate_env(value: str) -> str:
    return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)


def load_api_configs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    configs = json.loads(path.read_text())
    for config in configs:
        if config.get("auth_value"):
            config["auth_value"] = _interpolate_env(config["auth_value"])
    return configs


def _build_auth_headers(config: dict[str, Any]) -> dict[str, str]:
    auth_type = config.get("auth_type", "none")
    if auth_type == "none":
        return {}
    if auth_type == "bearer":
        return {"Authorization": f"Bearer {config['auth_value']}"}
    if auth_type == "api_key_header":
        return {config["auth_header_name"]: config["auth_value"]}
    if auth_type == "basic":
        encoded = base64.b64encode(config["auth_value"].encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    raise ValueError(f"unknown auth_type: {auth_type!r}")


async def build_rest_tools(configs: list[dict[str, Any]], stack: AsyncExitStack) -> list[BaseTool]:
    tools: list[BaseTool] = []

    for config in configs:
        client = httpx.AsyncClient(base_url=config["base_url"], headers=_build_auth_headers(config))
        await stack.enter_async_context(client)

        for operation in config["operations"]:
            tools.append(_build_operation_tool(config["name"], operation, client))

    return tools


def _build_operation_tool(config_name: str, operation: dict[str, Any], client: httpx.AsyncClient) -> BaseTool:
    method = operation["method"]
    path_template = operation["path"]

    async def call(**kwargs: Any) -> str:
        path = path_template
        remaining = dict(kwargs)
        for name in _PATH_PARAM_PATTERN.findall(path_template):
            if name not in remaining:
                raise ValueError(f"missing path parameter: {name}")
            path = path.replace(f"{{{name}}}", str(remaining.pop(name)))

        if method in ("GET", "DELETE"):
            response = await client.request(method, path, params=remaining)
        else:
            response = await client.request(method, path, json=remaining)

        response.raise_for_status()
        return response.text

    return tool(
        call,
        name=f"{config_name}_{operation['name']}",
        description=operation["description"],
        args_schema=operation["params_schema"],
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_rest_tool.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add agent/tools/rest.py config/api_allowlist.json tests/test_rest_tool.py
git commit -m "Add REST allowlist tools"
```

---

### Task 5: MCP tools

**Files:**
- Create: `agent/tools/mcp.py`, `config/mcp_servers.json`, `tests/fixtures/__init__.py`, `tests/fixtures/dummy_mcp_server.py`
- Test: `tests/test_mcp_tool.py`

**Interfaces:**
- Consumes: `langchain.mcp.MCPAdapter`.
- Produces: `load_mcp_server_configs(path: Path) -> list[dict]`, `build_mcp_tools(configs: list[dict], stack: AsyncExitStack) -> tuple[list[BaseTool], list[dict]]`. Task 12 imports both.

**Deliberate deviation from the spec, noted here rather than silently:** the merged spec described a separate `summarize_connections(configs, tools)` that inferred per-server tool lists from an `mcp_{server}_{tool}` name prefix. `MCPAdapter` returns tools with their own unprefixed names (that's the whole point — no manual wrapping), so there is no prefix left to infer from after the fact. `build_mcp_tools` here returns the per-server status alongside the tool list instead, computed at connection time, before that association would otherwise be lost. `summarize_connections` does not exist in this implementation.

- [ ] **Step 1: Add dependencies**

Run: `uv add "langchain[mcp]"`

If `fastmcp` isn't already importable after that (check with `uv run python -c "import fastmcp"`), run: `uv add fastmcp`

Create `config/mcp_servers.json`:

```json
[]
```

- [ ] **Step 2: Write the dummy MCP server fixture**

Create `tests/fixtures/__init__.py` (empty).

Create `tests/fixtures/dummy_mcp_server.py`:

```python
from fastmcp import FastMCP

mcp = FastMCP("dummy")


@mcp.tool()
def ping() -> str:
    """Respond with pong."""
    return "pong"


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_mcp_tool.py`:

```python
import json
from contextlib import AsyncExitStack
from pathlib import Path

from agent.tools.mcp import build_mcp_tools, load_mcp_server_configs

FIXTURE_SERVER = str(Path(__file__).parent / "fixtures" / "dummy_mcp_server.py")


def test_load_mcp_server_configs_returns_empty_list_for_missing_file(tmp_path):
    assert load_mcp_server_configs(tmp_path / "does-not-exist.json") == []


def test_load_mcp_server_configs_parses_json(tmp_path):
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(json.dumps([{"name": "dummy", "transport": "stdio", "command": FIXTURE_SERVER}]))

    configs = load_mcp_server_configs(config_path)

    assert configs[0]["name"] == "dummy"


async def test_build_mcp_tools_connects_to_real_stdio_server():
    configs = [{"name": "dummy", "transport": "stdio", "command": FIXTURE_SERVER}]

    async with AsyncExitStack() as stack:
        tools, status = await build_mcp_tools(configs, stack)

        tool_names = {t.name for t in tools}
        assert "ping" in tool_names

        ping_tool = next(t for t in tools if t.name == "ping")
        result = await ping_tool.ainvoke({})
        assert result == "pong"

    assert status == [{"name": "dummy", "transport": "stdio", "connected": True, "tools": ["ping"]}]


async def test_build_mcp_tools_skips_broken_server_without_crashing(caplog):
    configs = [{"name": "broken", "transport": "stdio", "command": "/no/such/script.py"}]

    async with AsyncExitStack() as stack:
        tools, status = await build_mcp_tools(configs, stack)

    assert tools == []
    assert status == [{"name": "broken", "transport": "stdio", "connected": False, "tools": []}]
    assert "broken" in caplog.text


async def test_build_mcp_tools_one_broken_server_does_not_block_a_good_one():
    configs = [
        {"name": "broken", "transport": "stdio", "command": "/no/such/script.py"},
        {"name": "dummy", "transport": "stdio", "command": FIXTURE_SERVER},
    ]

    async with AsyncExitStack() as stack:
        tools, status = await build_mcp_tools(configs, stack)

    assert {t.name for t in tools} == {"ping"}
    assert {s["name"]: s["connected"] for s in status} == {"broken": False, "dummy": True}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.tools.mcp'`

- [ ] **Step 5: Write the implementation**

Create `agent/tools/mcp.py`:

```python
import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from fastmcp.client import Client
from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def load_mcp_server_configs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def _adapter_target(config: dict[str, Any]):
    transport = config["transport"]
    if transport == "stdio":
        return Path(config["command"])
    if transport == "streamable_http":
        return config["url"]
    if transport == "sse":
        return Client(config["url"], mode="legacy")
    raise ValueError(f"unknown transport: {transport!r}")


async def build_mcp_tools(
    configs: list[dict[str, Any]], stack: AsyncExitStack
) -> tuple[list[BaseTool], list[dict[str, Any]]]:
    tools: list[BaseTool] = []
    status: list[dict[str, Any]] = []

    for config in configs:
        try:
            adapter = MCPAdapter(_adapter_target(config))
            await stack.enter_async_context(adapter)
            server_tools = await adapter.list_tools()
            tools.extend(server_tools)
            status.append({
                "name": config["name"],
                "transport": config["transport"],
                "connected": True,
                "tools": [t.name for t in server_tools],
            })
        except Exception:
            logger.warning("failed to connect to MCP server %r", config["name"], exc_info=True)
            status.append({
                "name": config["name"],
                "transport": config["transport"],
                "connected": False,
                "tools": [],
            })

    return tools, status
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_tool.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add agent/tools/mcp.py config/mcp_servers.json tests/fixtures/ tests/test_mcp_tool.py pyproject.toml uv.lock
git commit -m "Add MCP client tools"
```

---

### Task 6: Agent core (`build_agent`)

**Files:**
- Create: `agent/core.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: any `list[BaseTool]` (Tasks 3–5 produce these; Task 4 of the skills plan produces skill tools).
- Produces: `DuplicateToolName` (Exception), `build_agent(model, tools, system_prompt, max_tool_turns, checkpointer, summarize_trigger_tokens, summarize_keep_messages)`. Task 12 imports both.

**Verify-at-runtime note:** `ToolCallLimitExceededError`'s import path is assumed below to be `langchain.agents.middleware` (same module as `ToolCallLimitMiddleware`) — this wasn't directly confirmed in available docs. If Step 3's test fails on that import specifically (not on the behavior), check `python -c "from langchain.agents.middleware import ToolCallLimitExceededError"`; if it's actually elsewhere (e.g. `langgraph.errors`), fix the import in both the test and this note and continue — the rest of the plan is unaffected.

**Verify-at-runtime note:** `AsyncSqliteSaver.from_conn_string(...)` used as an async context manager followed by `await checkpointer.setup()` is the confirmed pattern for `AsyncPostgresSaver`; `AsyncSqliteSaver` (same `langgraph.checkpoint` family, `langgraph-checkpoint-sqlite` package) is assumed to follow the identical pattern. Step 6's test exercises this directly — if the constructor or method names differ, adjust the test and `main.py` (Task 12) together, and note the real API here for the next person.

- [ ] **Step 1: Add dependencies**

Run: `uv add langgraph-checkpoint-sqlite`

- [ ] **Step 2: Write the failing wiring tests**

Create `tests/test_core.py`:

```python
import pytest
from langchain.messages import AIMessage, ToolCall
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver

from agent.core import DuplicateToolName, build_agent


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
    model = GenericFakeChatModel(messages=iter([
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
    model = GenericFakeChatModel(messages=iter([
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
    from langchain.agents.middleware import ToolCallLimitExceededError

    def always_call_echo():
        while True:
            yield AIMessage(content="", tool_calls=[ToolCall(name="echo", args={"text": "x"}, id="call")])

    model = GenericFakeChatModel(messages=always_call_echo())
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.core'`

- [ ] **Step 4: Write the implementation**

Create `agent/core.py`:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
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
            ToolErrorMiddleware(on_error=lambda exc, request: f"Error: {exc}"),
            ToolCallLimitMiddleware(run_limit=max_tool_turns, exit_behavior="error"),
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", summarize_trigger_tokens),
                keep=("messages", summarize_keep_messages),
            ),
        ],
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_core.py -v`
Expected: 4 passed (adjusting the `ToolCallLimitExceededError` import first if the verify-at-runtime note above applies)

- [ ] **Step 6: Add and verify the SQLite checkpointer**

Append to `tests/test_core.py`:

```python
async def test_sqlite_checkpointer_persists_conversation_across_separate_invocations(tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = tmp_path / "checkpoints.db"
    model = GenericFakeChatModel(messages=iter([
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
```

Run: `uv run pytest tests/test_core.py -v`
Expected: 5 passed. If `AsyncSqliteSaver.from_conn_string`/`.setup()` don't exist under those exact names, run `uv run python -c "from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver; help(AsyncSqliteSaver)"` to find the real constructor/setup call, fix this test and the note above, and continue — same for `agent.aget_state`, LangGraph's documented way to inspect a checkpointed graph's current state.

- [ ] **Step 7: Commit**

```bash
git add agent/core.py tests/test_core.py pyproject.toml uv.lock
git commit -m "Add build_agent: tool wiring, error/turn-limit/summarization middleware, checkpointer"
```

---

### Task 7: SQLite storage (conversations, tasks)

**Files:**
- Create: `storage/__init__.py`, `storage/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `aiosqlite`.
- Produces: `init_db(db_path: Path) -> aiosqlite.Connection`, `create_conversation(db, source: str) -> str`, `create_task(db, conversation_id: str, input_text: str) -> str`, `update_task(db, task_id: str, **fields) -> None`, `get_task(db, task_id: str) -> dict | None`. Tasks 9 and 10 import all five.

- [ ] **Step 1: Write the failing tests**

Create `storage/__init__.py` (empty).

Create `tests/test_db.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'storage'`

- [ ] **Step 3: Write the implementation**

Create `storage/db.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add storage/ tests/test_db.py
git commit -m "Add SQLite storage layer (conversations, tasks)"
```

---

### Task 8: Chat WebSocket API

**Files:**
- Create: `api/__init__.py`, `api/chat.py`
- Test: `tests/test_chat_ws.py`

**Interfaces:**
- Consumes: `create_conversation` (Task 7), `build_agent`'s output (an object with `.astream_events()` — Task 6).
- Produces: `router` (a `fastapi.APIRouter` with the `/ws/chat` WebSocket route). Task 12 includes this router.

**Verify-at-runtime note:** the exact attributes on `stream.tool_calls`' items (`.tool_name`, `.input`, and how to obtain the final `.output` once the call completes) are taken from LangGraph's `v3` streaming docs but not exhaustively confirmed. Step 3's test exercises this against a real fake-model-backed agent — if an attribute name is wrong, the test's failure message will show the real object's attributes; fix the handler and the test together and move on.

- [ ] **Step 1: Write the failing test**

Create `api/__init__.py` (empty).

Create `tests/test_chat_ws.py`:

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.messages import AIMessage, ToolCall
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver

from agent.core import build_agent
from api.chat import router
from storage.db import init_db


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


def test_websocket_streams_tool_call_and_final_answer(db):
    model = GenericFakeChatModel(messages=iter([
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
        # connection must still be open for a second message
        ws.send_json({"type": "message", "conversation_id": None, "content": "hi again"})
        reply2 = ws.receive_json()
        assert reply2["type"] == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat_ws.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.chat'`

- [ ] **Step 3: Write the implementation**

Create `api/chat.py`:

```python
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
            output = await call.output
            await websocket.send_json({
                "type": "tool_result",
                "name": call.tool_name,
                "output": str(output),
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
```

- [ ] **Step 4: Run tests, adjusting the streaming attribute access if needed**

Run: `uv run pytest tests/test_chat_ws.py -v`
Expected: 4 passed. If `test_websocket_streams_tool_call_and_final_answer` fails on an `AttributeError` from the `stream.tool_calls`/`stream.messages` projections, print `dir(call)` / `dir(message)` in a throwaway script to find the real attribute names, fix `_run_turn` accordingly, and re-run.

- [ ] **Step 5: Commit**

```bash
git add api/__init__.py api/chat.py tests/test_chat_ws.py
git commit -m "Add chat WebSocket API"
```

---

### Task 9: Trigger REST API

**Files:**
- Create: `api/tasks.py`, `api/schemas.py`
- Test: `tests/test_tasks_api.py`

**Interfaces:**
- Consumes: `create_conversation`, `create_task`, `update_task`, `get_task` (Task 7); `verify_token`, `parse_bearer_token` (Task 2); the built agent's `.ainvoke()` (Task 6).
- Produces: `router` (a `fastapi.APIRouter` with `POST /tasks` and `GET /tasks/{task_id}`). Task 12 includes this router.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tasks_api.py`:

```python
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
    client = TestClient(app)

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
    class FailingModel:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("LLM unreachable")

    app = make_app(GenericFakeChatModel(messages=iter([])), db)
    app.state.agent = build_agent(
        model=FailingModel(),
        tools=[],
        system_prompt="test",
        max_tool_turns=20,
        checkpointer=InMemorySaver(),
        summarize_trigger_tokens=4000,
        summarize_keep_messages=20,
    )
    # build_agent wraps a real LangChain model type; to force a failure at the
    # ainvoke boundary for this test, monkeypatch the compiled agent's ainvoke directly.
    real_agent = app.state.agent

    async def failing_ainvoke(*args, **kwargs):
        raise RuntimeError("LLM unreachable")

    real_agent.ainvoke = failing_ainvoke

    client = TestClient(app)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tasks_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.tasks'`

- [ ] **Step 3: Write the implementation**

Create `api/schemas.py`:

```python
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
```

Create `api/tasks.py`:

```python
import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request

from api.schemas import CreateTaskRequest, TaskResponse
from auth import parse_bearer_token, verify_token
from storage.db import create_conversation, create_task, get_task, update_task

router = APIRouter()


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

    asyncio.create_task(_run_task(request.app, task_id, conversation_id, body.input))

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tasks_api.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add api/tasks.py api/schemas.py tests/test_tasks_api.py
git commit -m "Add trigger REST API (POST /tasks, GET /tasks/{id})"
```

---

### Task 10: MCP status API

**Files:**
- Create: `api/mcp_status.py`
- Test: `tests/test_mcp_status_api.py`

**Interfaces:**
- Consumes: `parse_bearer_token`, `verify_token` (Task 2); `app.state.mcp_status` (a `list[dict]`, produced by `build_mcp_tools` in Task 5, stored in Task 12).
- Produces: `router` (a `fastapi.APIRouter` with `GET /mcp/status`). Task 12 includes this router.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_status_api.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.mcp_status import router


def make_app(mcp_status):
    app = FastAPI()
    app.include_router(router)
    app.state.settings = type("S", (), {"API_TOKEN": "dev-token"})()
    app.state.mcp_status = mcp_status
    return app


def test_mcp_status_without_token_returns_401():
    app = make_app([])
    client = TestClient(app)

    response = client.get("/mcp/status")

    assert response.status_code == 401


def test_mcp_status_returns_configured_servers():
    status = [{"name": "dummy", "transport": "stdio", "connected": True, "tools": ["ping"]}]
    app = make_app(status)
    client = TestClient(app)

    response = client.get("/mcp/status", headers={"Authorization": "Bearer dev-token"})

    assert response.status_code == 200
    assert response.json() == status
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_status_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.mcp_status'`

- [ ] **Step 3: Write the implementation**

Create `api/mcp_status.py`:

```python
from fastapi import APIRouter, Header, HTTPException, Request

from auth import parse_bearer_token, verify_token

router = APIRouter()


@router.get("/mcp/status")
async def mcp_status(request: Request, authorization: str | None = Header(default=None)) -> list[dict]:
    token = parse_bearer_token(authorization)
    if not verify_token(token, request.app.state.settings.API_TOKEN):
        raise HTTPException(status_code=401, detail="unauthorized")

    return request.app.state.mcp_status
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_status_api.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add api/mcp_status.py tests/test_mcp_status_api.py
git commit -m "Add MCP status API"
```

---

### Task 11: Wire everything into `main.py`

**Files:**
- Create: `main.py` (replacing the `uv init` placeholder), `config/system_prompt.md`

**Interfaces:**
- Consumes: every task above.
- Produces: the running FastAPI app, `app.state.settings`, `app.state.db`, `app.state.agent`, `app.state.mcp_status`, `app.state.resources_stack`. This is the final integration point for this plan; the (separate) skills plan's Task 4 extends this same `lifespan()`.

- [ ] **Step 1: Create the base system prompt**

Create `config/system_prompt.md`:

```markdown
You are storage-agent, an assistant for a storage infrastructure team. You have tools to manipulate files in a sandbox, call allowlisted REST APIs, and use connected MCP servers. Use them precisely, and never guess at a resource name or path — ask if you're unsure.
```

- [ ] **Step 2: Write `main.py`**

```python
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.core import build_agent
from agent.tools.files import build_file_tools
from agent.tools.mcp import build_mcp_tools, load_mcp_server_configs
from agent.tools.rest import build_rest_tools, load_api_configs
from api.chat import router as chat_router
from api.mcp_status import router as mcp_status_router
from api.tasks import router as tasks_router
from settings import Settings
from storage.db import init_db

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    logging.getLogger().setLevel(settings.LOG_LEVEL)

    sandbox_root = Path(settings.SANDBOX_ROOT).resolve(strict=False)
    sandbox_root.mkdir(parents=True, exist_ok=True)

    app.state.db = await init_db(Path(settings.DB_PATH))

    model = init_chat_model(
        model=settings.LLM_MODEL,
        model_provider="openai",
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )

    base_prompt = Path(settings.SYSTEM_PROMPT_PATH).read_text()

    resources_stack = AsyncExitStack()
    app.state.resources_stack = resources_stack

    checkpointer = await resources_stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string(settings.DB_PATH + ".checkpoints")
    )
    await checkpointer.setup()

    file_tools = build_file_tools(sandbox_root)

    api_configs = load_api_configs(Path(settings.API_ALLOWLIST_PATH))
    rest_tools = await build_rest_tools(api_configs, resources_stack)

    mcp_servers = load_mcp_server_configs(Path(settings.MCP_SERVERS_PATH))
    mcp_tools, mcp_status = await build_mcp_tools(mcp_servers, resources_stack)
    app.state.mcp_status = mcp_status

    all_tools = file_tools + rest_tools + mcp_tools
    app.state.agent = build_agent(
        model,
        all_tools,
        base_prompt,
        settings.MAX_TOOL_TURNS,
        checkpointer,
        settings.SUMMARIZE_TRIGGER_TOKENS,
        settings.SUMMARIZE_KEEP_MESSAGES,
    )

    yield

    await resources_stack.aclose()
    await app.state.db.close()


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)
app.include_router(tasks_router)
app.include_router(mcp_status_router)


@app.get("/")
async def root() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 3: Live smoke test — app starts cleanly with empty allowlist/MCP configs**

```bash
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill 2>/dev/null
rm -f data/storage_agent.db data/storage_agent.db.checkpoints
uv run uvicorn main:app --port 8000 &> /tmp/storage-agent-run.log &
sleep 3
curl -s -o /dev/null -w "readiness: %{http_code}\n" http://localhost:8000/
grep -i "error\|traceback" /tmp/storage-agent-run.log || echo "no errors in startup log"
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill
```

Expected: `readiness: 200`, no errors in the log.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests from Tasks 1–10 pass.

- [ ] **Step 5: Live end-to-end verification against the real Groq endpoint**

```bash
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill 2>/dev/null
rm -f data/storage_agent.db data/storage_agent.db.checkpoints
uv run uvicorn main:app --port 8000 &> /tmp/storage-agent-run.log &
sleep 3
uv run python -c "
import asyncio, json
import websockets

async def main():
    async with websockets.connect('ws://127.0.0.1:8000/ws/chat') as ws:
        await ws.send(json.dumps({'token': 'dev-token'}))
        await ws.send(json.dumps({'type': 'message', 'conversation_id': None, 'content': 'Write a file named hello.txt containing the word hello, then read it back to confirm.'}))
        while True:
            reply = json.loads(await ws.recv())
            print(reply)
            if reply['type'] in ('final', 'error'):
                break

asyncio.run(main())
"
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill
```

Expected: `tool_call`/`tool_result` events for `write_file` then `read_file`, and a `final` answer confirming the content. Requires `websockets` as a dev dependency: `uv add --dev websockets` if not already resolvable.

- [ ] **Step 6: Commit**

```bash
git add main.py config/system_prompt.md
git commit -m "Wire core system together in main.py"
```

---

### Task 12: Webview

**Files:**
- Create: `web/index.html`, `web/chat.js`, `web/style.css`
- Modify: `main.py` (mount static files)
- Test: `tests/test_webview.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: the browser chat UI. Nothing later depends on this task; it's the last piece for interactive use (the skills plan's Task 5 live-verification step drives the WebSocket directly, not through this UI).

- [ ] **Step 1: Write the failing test**

Create `tests/test_webview.py`:

```python
from pathlib import Path

from fastapi.testclient import TestClient

from main import app


def test_root_serves_index_html(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("API_TOKEN", "dev-token")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("SANDBOX_ROOT", str(tmp_path / "sandbox"))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "storage-agent" in response.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_webview.py -v`
Expected: FAIL (root currently returns JSON, not HTML)

- [ ] **Step 3: Write the webview files**

Create `web/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>storage-agent</title>
  <link rel="stylesheet" href="/web/style.css">
</head>
<body>
  <div id="sidebar">
    <h2>MCP Status</h2>
    <div id="mcp-status">Loading…</div>
  </div>
  <div id="main">
    <div id="messages"></div>
    <div id="thinking" class="hidden">thinking…</div>
    <form id="chat-form">
      <input id="chat-input" type="text" autocomplete="off" placeholder="Ask storage-agent…">
      <button type="submit">Send</button>
    </form>
  </div>
  <script src="/web/chat.js"></script>
</body>
</html>
```

Create `web/style.css`:

```css
body { display: flex; margin: 0; font-family: system-ui, sans-serif; height: 100vh; }
#sidebar { width: 260px; border-right: 1px solid #ddd; padding: 1rem; overflow-y: auto; }
#main { flex: 1; display: flex; flex-direction: column; padding: 1rem; }
#messages { flex: 1; overflow-y: auto; }
.message { margin: 0.5rem 0; }
.message.user { font-weight: 600; }
.message.tool { color: #888; font-size: 0.85rem; font-family: monospace; }
#thinking { color: #888; font-style: italic; }
#thinking.hidden { display: none; }
#chat-form { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
#chat-input { flex: 1; padding: 0.5rem; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.4rem; }
.status-dot.connected { background: #2ecc71; }
.status-dot.disconnected { background: #ccc; }
```

Create `web/chat.js`:

```javascript
let token = null;
let conversationId = null;
let ws = null;

function appendMessage(text, className) {
  const div = document.createElement("div");
  div.className = `message ${className}`;
  div.textContent = text;
  document.getElementById("messages").appendChild(div);
  div.scrollIntoView();
}

function setThinking(visible) {
  document.getElementById("thinking").classList.toggle("hidden", !visible);
}

function connect() {
  ws = new WebSocket(`ws://${window.location.host}/ws/chat`);
  ws.onopen = () => ws.send(JSON.stringify({ token }));
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "tool_call") {
      appendMessage(`→ ${data.name}(${JSON.stringify(data.input)})`, "tool");
    } else if (data.type === "tool_result") {
      appendMessage(`← ${data.name}: ${data.output}`, "tool");
      setThinking(true);
    } else if (data.type === "final") {
      setThinking(false);
      conversationId = data.conversation_id;
      appendMessage(data.content, "assistant");
    } else if (data.type === "error") {
      setThinking(false);
      appendMessage(`Error: ${data.message}`, "tool");
    }
  };
}

async function loadMcpStatus() {
  const response = await fetch("/mcp/status", { headers: { Authorization: `Bearer ${token}` } });
  const servers = await response.json();
  renderMcpStatus(servers);
}

function renderMcpStatus(servers) {
  const el = document.getElementById("mcp-status");
  el.innerHTML = "";
  for (const server of servers) {
    const div = document.createElement("div");
    const dot = server.connected ? "connected" : "disconnected";
    div.innerHTML = `<span class="status-dot ${dot}"></span>${server.name}`;
    if (server.connected && server.tools.length) {
      const list = document.createElement("ul");
      for (const toolName of server.tools) {
        const li = document.createElement("li");
        li.textContent = toolName;
        list.appendChild(li);
      }
      div.appendChild(list);
    }
    el.appendChild(div);
  }
}

document.getElementById("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = document.getElementById("chat-input");
  const content = input.value.trim();
  if (!content) return;
  appendMessage(content, "user");
  setThinking(true);
  ws.send(JSON.stringify({ type: "message", conversation_id: conversationId, content }));
  input.value = "";
});

token = window.prompt("API token:");
connect();
loadMcpStatus();
```

- [ ] **Step 4: Mount static files and serve `index.html` at `/`**

In `main.py`, replace the `root()` handler and add the static mount:

```python
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
```

```python
app.mount("/web", StaticFiles(directory="web"), name="web")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("web/index.html")
```

(Remove the old JSON-returning `root()` — this replaces it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_webview.py -v`
Expected: 1 passed

- [ ] **Step 6: Run the full test suite one more time**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add web/ main.py tests/test_webview.py
git commit -m "Add chat webview with MCP status sidebar"
```
