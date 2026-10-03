# storage-agent: Complete Specification

This file merges what were three separate documents into one: the core system design, the skills system design, and the skills system's step-by-step implementation plan. The three originals (`docs/superpowers/specs/2026-09-24-storage-agent-design.md`, `docs/superpowers/specs/2026-09-25-skills-design.md`, `docs/superpowers/plans/2026-09-25-skills-system.md`) have been removed; this file is now the single source of truth.

**Status:** Implemented. Both Part 1 (core system: sandboxed file tools, allowlisted REST tool, MCP client, chat webview, trigger REST API, LangChain-based agent loop with checkpointer persistence) and Part 2/3 (skills system: `agent/skills.py`, `agent/tools/skills.py`, `agent/prompt.py`, wired into `main.py`) are built, tested, and verified live against the real LLM endpoint.

---

## Part 1: Core System Design

### Context

storage-agent is a self-hosted AI agent (Python, server-hosted) that can:

- manipulate files inside one configured sandbox subdirectory,
- call REST APIs, restricted to an admin-configured allowlist (not arbitrary URLs),
- connect to MCP servers — both local stdio subprocesses and remote HTTP/SSE — and use their tools,
- be used interactively via a browser chat webview,
- and be triggered programmatically by external scripts/apps via a REST endpoint (submit a task, poll for status/result), separate from the human chat flow.

The project was built from scratch. The guiding constraint was that the code stay **easy to read and extend**: every capability — file ops, REST-allowlist calls, MCP tools, and skill tools — is a LangChain tool, and `agent/core.py::build_agent()` is the single place that assembles the full tool list and hands it to `create_agent()`. Adding a new capability later means adding one small module/config entry and appending to that tool list, not touching the core loop.

**Skills** (Claude-Code-style: named instructions injected into the system prompt when relevant) were originally deferred, then built — see Part 2 and Part 3 below. `agent/prompt.py::build_system_prompt()` was deliberately kept as its own discrete function specifically so this addition wouldn't require restructuring, and that's exactly how it was added: extended in place, no separate registry class needed.

One thing remains intentionally deferred:

- **System prompt editing** — loads `config/system_prompt.md` as a static file at startup. No runtime/admin API to edit it.

### Architecture

Single Python/FastAPI monolith, one process, one SQLite file. Both entry points (webview and trigger API) share the same LangChain agent (built once by `build_agent()`); they only differ in how they feed it messages and how they return results (streaming over WebSocket vs. async task + polling).

```
Browser ──WS──▶ Chat Webview (WebSocket) ─┐
                                            ├──▶ LangChain Agent (create_agent) ──▶ File tools (sandboxed)
External ──REST─▶ Trigger API (POST/GET) ─┘         │                            ├─▶ REST tools (allowlist)
Script                                              ▼                            ├─▶ MCP tools (stdio/HTTP)
                                     Checkpointer (conversation history)          └─▶ Skill tools
                                              │
                                     SQLite (conversations, tasks)
                                              │
                                     OpenAI-compatible chat completions
                                     endpoint (user's own base_url + key)
```

Auth: a single shared bearer token, required on both the WebSocket (sent as the first frame after connect, since browsers can't set custom headers on a WS upgrade) and the trigger REST API (`Authorization: Bearer <token>` header).

### Directory structure (as built)

```
storage-agent/
  pyproject.toml                 # deps via uv
  .env.example
  .gitignore                     # data/, .env, __pycache__, .venv
  main.py                        # FastAPI app + lifespan (startup/shutdown)
  settings.py                    # pydantic-settings, reads env/.env
  auth.py                        # parse_bearer_token / verify_token (pure logic, no framework coupling)

  config/
    system_prompt.md             # static system prompt
    api_allowlist.json           # read directly at startup, no DB round-trip (starts empty: [])
    mcp_servers.json              # read directly at startup, no DB round-trip (starts empty: [])

  skills/                         # one subdirectory per skill (SKILL.md + optional reference/) - see Part 2

  agent/
    core.py                      # build_agent(): create_agent() + inline duplicate-tool-name check + middleware wiring (errors, turn limit, summarization)
    prompt.py                    # build_system_prompt() - injects always-on/on-demand skill instructions (see Part 2)
    skills.py                    # load_skills() (see Part 2)
    tools/
      files.py                   # sandboxed file tools (LangChain @tool-wrapped) + resolve_in_sandbox()
      rest.py                    # allowlisted REST tools (LangChain @tool-wrapped)
      mcp.py                     # builds one langchain.mcp.MCPAdapter per configured MCP server
      skills.py                  # load_skill/read_skill_file tools (LangChain @tool-wrapped, see Part 2)

  api/
    chat.py                      # WebSocket /ws/chat
    tasks.py                     # POST /tasks, GET /tasks/{id}
    mcp_status.py                # GET /mcp/status
    schemas.py                   # pydantic request/response models

  storage/
    db.py                        # schema, init, seeding, CRUD helpers (aiosqlite)

  web/
    index.html                   # chat UI shell + MCP status panel
    chat.js                      # WebSocket connect + auth frame + streaming render + thinking indicator + MCP status fetch
    style.css

  data/                          # gitignored at runtime
    storage_agent.db
    sandbox/                     # default SANDBOX_ROOT

  tests/                         # 110 tests, all passing
    test_sandbox_traversal.py
    test_files_tool.py
    test_db.py
    test_settings.py
    test_auth.py
    test_core.py                  # tests build_agent(): duplicate-tool-name check, and the middleware list (error handling, turn limit, summarization) it wires onto create_agent()
    test_prompt.py
    test_chat_ws.py
    test_tasks_api.py
    test_rest_tool.py
    test_mcp_tool.py              # now tests agent/tools/mcp.py's MCPAdapter wiring, not a hand-rolled stdio/streamable_http/sse client
    test_mcp_status_api.py
    test_skills.py
    test_skills_tool.py
    fixtures/dummy_mcp_server.py  # tiny real MCP stdio server used by test_mcp_tool.py
```

### Core interface: LangChain tools

Every capability — hand-written file tools, dynamically generated REST-allowlist tools, discovered MCP tools, and skill tools — is a LangChain `BaseTool`, built via the `langchain.tools.tool` decorator/factory (`tool(fn, name=..., description=..., args_schema=...)`) applied to a closure. This is the whole extensibility mechanism; there is no custom `Tool` type or class hierarchy per tool source. `args_schema` accepts a plain JSON-schema dict directly, which lets `api_allowlist.json`'s existing `params_schema` format plug straight in for REST tools with no dynamic-pydantic-model step.

Error safety is centralized instead of per-tool: `langchain.agents.middleware.ToolErrorMiddleware` — a LangChain *built-in*, not project code — is registered once on the agent (see below) with a one-line `on_error` callback (`lambda exc, request: f"Error: {exc}"`) that turns any tool's exception into a `ToolMessage`. A bad path, a failed HTTP call, or an unreachable MCP tool never crashes the agent loop, and no individual tool — nor any project-owned middleware module — needs its own try/except.

- `agent/tools/files.py::build_file_tools(sandbox_root)` — builds `list_dir`, `search_files`, `read_file`, `write_file`, `edit_file`, `delete_file`, `move_file` as `tool(...)`-wrapped closures over `sandbox_root`.
- `agent/tools/rest.py::build_rest_tools(configs, stack)` — one tool per `(api_config, operation)` pair, named `{config_name}_{operation_name}`, with `args_schema=operation.params_schema` taken directly from the allowlist JSON. **Optimization**: builds exactly one pooled `httpx.AsyncClient` per `api_config` (entered into the shared `resources_stack`, same lifecycle pattern as MCP connections) and reuses it across every operation/call for that config, instead of opening a fresh connection (and TLS handshake, for HTTPS APIs) on every single tool invocation. Covered by `tests/test_rest_tool.py::test_builds_one_http_client_per_config_shared_across_operations`, which spies on `httpx.AsyncClient.__init__` to prove construction count stays at 1 regardless of operation count.
- `agent/tools/mcp.py::build_mcp_tools(configs, stack)` — for each configured MCP server, constructs a `langchain.mcp.MCPAdapter` (target is an HTTP(S) URL for `streamable_http`, a script `Path` for `stdio`, or a `fastmcp.client.Client(..., mode="legacy")` for `sse`), enters it into `stack`, and calls `await adapter.list_tools()` — the adapter returns ready-to-use LangChain tools directly, already carrying their own name/description/schema (no manual `mcp_{server_name}_{tool_name}` wrapping needed, though names are still checked for collisions in `build_agent()` below).
- `agent/core.py::build_agent(...)` concatenates every tool list (file/REST/MCP/skill) into one flat list for `create_agent(tools=...)`, with a small inline loop that raises `DuplicateToolName` on any repeated `.name` before constructing the agent (fail fast, not silent shadowing) — a few lines inside `build_agent()` itself, not a separate registry module or class. No dispatch-by-name or spec precompute is needed — LangChain's agent graph builds and caches each tool's model-facing spec internally once, from the tool list passed to `create_agent()`.
- `agent/tools/mcp.py::summarize_connections(configs, tools)` — pure function, no new connection attempts. For each configured MCP server, checks whether any tool in the already-built `tools` list carries that server's name prefix, and reports `{name, transport, connected, tools}`. Computed once at startup and stored on `app.state.mcp_status` for `GET /mcp/status` to serve.

### The agent core loop

`agent/core.py::build_agent(model, tools, system_prompt, max_tool_turns, checkpointer)` constructs one `langchain.agents.create_agent(...)` graph, once at startup (stored on `app.state`, not rebuilt per request — same "build once, reuse for the process lifetime" principle the old precomputed `tool_specs` followed):

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)

class DuplicateToolName(Exception):
    pass


def build_agent(model, tools, system_prompt, max_tool_turns, checkpointer, summarize_trigger_tokens, summarize_keep_messages):
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

- `model` comes from `langchain.chat_models.init_chat_model(model=settings.LLM_MODEL, model_provider="openai", base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY)` — this is the direct replacement for the old hand-rolled `LLMClient`/`OpenAICompatibleClient`, and supports the user's custom OpenAI-compatible endpoint (e.g. Ollama) with no adapter code needed.
- Per message, `agent.astream_events({"messages": [...]}, config={"configurable": {"thread_id": conversation_id}}, version="v3")` is called with just the new user message — the `checkpointer` (see "Data model" below) reloads that conversation's prior history automatically, keyed by `thread_id`; there is no hand-rolled history-reconstruction step. The event stream is translated into the same WebSocket frame shapes the app already emits (`tool_call`, `tool_result`, `final`, `error`) so `api/chat.py`'s wire protocol and `web/chat.js` need no changes — only the internals of what produces those frames change.
- `ToolErrorMiddleware(on_error=lambda exc, request: f"Error: {exc}")` is the built-in replacement for the old hand-rolled error-guard — see "Core interface" above.
- `ToolCallLimitMiddleware(run_limit=max_tool_turns, exit_behavior="error")` replaces the old `MAX_TOOL_TURNS`/`ToolTurnLimitExceeded`: exceeding the per-invocation tool-call budget raises `ToolCallLimitExceededError`, caught the same way `ToolTurnLimitExceeded` was caught before.
- `SummarizationMiddleware(...)` compresses older turns once a conversation's token count crosses `summarize_trigger_tokens` (settings: `SUMMARIZE_TRIGGER_TOKENS`, default 4000), keeping the most recent `summarize_keep_messages` (settings: `SUMMARIZE_KEEP_MESSAGES`, default 20) verbatim — this directly addresses the prompt-size/latency concern already noted for this project's CPU-bound local model (see Part 2), for any conversation that runs long, not just skills. It reuses the same `model` the agent already talks to for the summarization call itself, since this project points at a single self-hosted OpenAI-compatible endpoint rather than a mix of cheap/expensive providers — there's no separate "smaller model" to delegate to.
- An unknown tool name can no longer occur at this layer — `create_agent` only ever routes to tools present in the `tools` list it was built with.
- **Considered, declined: human-in-the-loop tool approval.** LangChain offers an interrupt-based middleware that pauses execution before a destructive tool call until a human approves it — a reasonable general best practice, but it requires a pause/resume flow with someone present to approve. This project's trigger API (`POST /tasks`) is explicitly designed for *unattended* execution by external scripts; an interrupted background task would hang forever with no one to approve it. Declined for that reason — destructive-action safety stays prompt-based, via the `team-safety-rules` always-on skill (see Part 2), applied uniformly across both the webview and trigger-API entry points.

### Sandboxing (file tools)

At startup, `SANDBOX_ROOT = Path(settings.SANDBOX_ROOT).resolve(strict=True)` (fails fast if missing). Every incoming path from the LLM goes through `resolve_in_sandbox()` before touching the filesystem:

1. Reject if the path is absolute, or `".."` appears in its parts.
2. `resolved = (sandbox_root / user_path).resolve(strict=False)` — resolves symlinks on existing components (defeats a symlink planted inside the sandbox pointing outside it) while still allowing new files to be created.
3. Reject unless `resolved.is_relative_to(sandbox_root)`.

`move_file` validates both `src` and `dest` independently. Covered by `tests/test_sandbox_traversal.py`, including a real symlink-escape case.

### REST allowlist tools

`config/api_allowlist.json` defines named APIs: `{name, description, base_url, auth_type, auth_value, auth_header_name, operations: [{name, method, path, description, params_schema}]}`. `auth_value` (and MCP `headers`/`env` values) support `${ENV_VAR}` interpolation resolved at seed time, so secrets aren't committed to the JSON files.

Only the declared `operations` become callable tools — the LLM never gets a generic "call any URL" tool. `auth_type` supports `none`, `bearer`, `api_key_header`, and `basic`. Path placeholders (`{param}`) in `operation.path` are substituted from the tool's arguments; any remaining arguments become query params (`GET`/`DELETE`) or a JSON body (other methods). A non-2xx response is caught by the shared `ToolErrorMiddleware` (see "Core interface" above) and returned as an `"Error: ..."` string.

### MCP client

Uses `langchain.mcp.MCPAdapter` (`langchain[mcp]>=1.4.0`; **this namespace is explicitly marked beta by LangChain — the API may change between releases, pin the version and re-check this section on upgrade**). One adapter is constructed per configured MCP server (`transport: stdio | streamable_http | sse`):

- `stdio`: `MCPAdapter(Path(command_script))` — launched as a subprocess, one per adapter.
- `streamable_http`: `MCPAdapter(url)` — a plain HTTP(S) URL string, reached over streamable HTTP.
- `sse`: `MCPAdapter(fastmcp.client.Client(url, mode="legacy"))` — kept only as a legacy fallback, pinning the older protocol era so it doesn't get dropped by auto-negotiation.

Each is used as an async context manager: `async with MCPAdapter(target) as adapter: tools = await adapter.list_tools()` — the returned tools are already-wrapped LangChain `BaseTool`s (name, description, and schema derived from the server's own advertised tool metadata), so there's no manual attribute-mapping step from the raw MCP SDK types the way the old hand-rolled client needed.

**TLS verification — open item, not yet re-verified under this library**: the previous hand-rolled client (built directly on the `mcp` SDK's `httpx2`-based transport) had a deliberate, user-confirmed `verify=False` override disabling TLS certificate verification on remote `streamable_http`/`sse` connections, after tracing a `CERTIFICATE_VERIFY_FAILED` error to an incomplete OS trust store on one deployment machine (not a real certificate problem). That requirement — the ability to bypass or otherwise correctly configure TLS verification for this same class of trust-store issue — must carry over, but `MCPAdapter`'s internal transport/TLS configuration was not something available reference docs covered explicitly. **Before relying on this in a real deployment**, check `langchain-mcp`/`fastmcp`'s actual transport code for the equivalent verification-control hook, and document the finding here (either a scoped, safer replacement, or a re-confirmation that the same blanket bypass is still needed and accepted).

All adapters are long-lived async context managers entered into `app.state.resources_stack` (an `AsyncExitStack` created during FastAPI's `lifespan` in `main.py`) and closed together on shutdown — this same stack also holds the pooled REST `httpx.AsyncClient`s, see the REST allowlist section above. Each server connection is attempted independently with its own try/except in `build_mcp_tools` — a failing server is logged (`logger.warning(..., exc_info=True)`) and skipped, it never blocks the rest of the app or other servers from starting. Verified with `tests/test_mcp_tool.py` against a real local stdio server (`tests/fixtures/dummy_mcp_server.py`) and against a deliberately-broken server config in the same run.

### Chat WebSocket resilience (bug found while running the app live)

The original `api/chat.py` only caught `ToolCallLimitExceededError` around the agent's event stream. Driving the app against a real (but unreachable) LLM endpoint surfaced a connection error from the chat model provider that propagated out of the WebSocket handler uncaught, killing the connection outright (`ConnectionClosedError: no close frame received or sent` on the client side) instead of reporting a clean error. Fixed by widening the `except` to catch any `Exception` from consuming `agent.astream_events(...)`, matching the resilience `api/tasks.py` already had — the connection now sends `{"type": "error", "message": ...}` and stays open for the next message. Covered by `tests/test_chat_ws.py::test_llm_failure_sends_error_event_without_crashing_connection`, and re-verified live.

### Webview UX additions

- **Thinking indicator** (`web/chat.js`): local models can take tens of seconds per turn with no intermediate output, which looked indistinguishable from the app being broken. A pulsing "thinking…" line now appears immediately after sending a message and after each `tool_result` (since the loop goes back to the model), and disappears the instant any server event arrives.
- **MCP status side panel** (`web/index.html` `#sidebar` / `#mcp-status`, rendered by `chat.js::loadMcpStatus()`/`renderMcpStatus()`): the page layout is a fixed-width left sidebar plus a chat main panel. On page load, the sidebar fetches `GET /mcp/status` (using the same cached token as the WebSocket) and renders one entry per configured MCP server — a status dot (● connected / ○ disconnected) plus name, with the full list of its discovered tools shown underneath when connected. Display-only by design (no enable/disable toggle) — an earlier design question confirmed this narrower scope over live connect/disconnect, which would need per-server connection lifecycle management instead of the current startup-only shared `AsyncExitStack`.

### Data model (SQLite)

- `conversations(id, source['webview'|'task'], created_at)` — just enough to know a conversation exists and where it came from; slimmed relative to earlier versions of this design now that message content itself lives elsewhere (below).
- `tasks(id, status['pending'|'running'|'completed'|'failed'], input, result, error, conversation_id, created_at, started_at, finished_at)`

Conversation *message history* is no longer a hand-rolled table — it's a LangGraph **checkpointer** (`checkpointer=` on `create_agent`, see "The agent core loop" above), keyed by `thread_id=conversation_id`. This is LangGraph's own documented mechanism for exactly this (per-conversation state across separate calls), so there's no `messages` table or `get_conversation_messages()`-style reconstruction function to write or maintain. **Flagged, not asserted:** a SQLite-file-backed checkpointer (so history survives a server restart, matching the DB's own `data/storage_agent.db` file-based durability) is the intended choice — expected to be `langgraph-checkpoint-sqlite`'s `AsyncSqliteSaver`, by analogy with the confirmed `PostgresSaver` pattern in LangGraph's docs, but the exact package/class needs confirming against LangGraph's current docs at implementation time. If it doesn't hold up, the fallback is a hand-rolled `messages` table exactly as in the pre-LangChain design.

`conversations` and `tasks` remain hand-rolled because a checkpointer doesn't provide either: `conversations.source` and `tasks`' full status/result/error/timestamp lifecycle are application-level bookkeeping a generic conversation-state store has no concept of.

**Neither REST allowlist configs nor MCP servers are DB-backed** — both were originally designed that way (mirroring `config/*.json` into a SQLite table, `seed_*()` upserting on every startup), and both were later simplified to read their JSON file directly at startup with no DB round-trip at all: `agent/tools/mcp.py::load_mcp_server_configs(path)` and `agent/tools/rest.py::load_api_configs(path)`. Asked directly why each one had a DB table, the honest answer both times was the same: it existed only to leave a seam for a future runtime admin/toggle API that was never built, so the table was dead weight — always fully overwritten from the JSON file on every startup, with no code path that ever let the DB diverge from the file. MCP was simplified first; the REST allowlist had the identical pattern and got the same treatment shortly after, once the same question was asked about it. If a real runtime admin API for either is ever actually built, this is the seam to add back at that point — not before it's needed.

### API surface

- **`WebSocket /ws/chat`**: client's first frame must be `{"token": "..."}` (else `{"type": "error", "message": "unauthorized"}` then closed with code 4401). Then exchanges `{"type": "message", "conversation_id": "<uuid|null>", "content": "..."}` for streamed `{"type": "tool_call"|"tool_result"|"final"|"error", ...}` frames. A `null` `conversation_id` creates a new conversation; the server's own `final` event (not the core loop's) carries the `conversation_id` so the client can persist it for the next message.
- **`POST /tasks`** (`Authorization: Bearer <token>`, body `{"input": "..."}`) → creates a task + conversation row, runs the agent loop as a background `asyncio.create_task`, returns `202` `{"task_id", "status": "pending"}` immediately.
- **`GET /tasks/{task_id}`** (same auth) → `{"task_id", "status", "input", "result", "error", "created_at", "started_at", "finished_at"}`, `404` if unknown. A task interrupted by a server restart stays `running` rather than resuming (acceptable for v1; not retried automatically). A background task's own exceptions are always caught and recorded as `status="failed"`, `error=str(exc)` — verified live against an unreachable LLM endpoint.
- **`GET /mcp/status`** (same auth) → `list[{"name", "transport", "connected", "tools"}]`, one entry per configured MCP server (regardless of whether it connected successfully), computed once at startup from `summarize_connections()`. Surfaced in the webview's sidebar (see Webview UX additions above).

### Config

Env vars (`settings.py`, `pydantic-settings`): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `SANDBOX_ROOT`, `API_TOKEN`, `DB_PATH`, `SYSTEM_PROMPT_PATH`, `API_ALLOWLIST_PATH`, `MCP_SERVERS_PATH`, `SKILLS_PATH` (default `./skills`), `MAX_TOOL_TURNS` (default 20), `SUMMARIZE_TRIGGER_TOKENS` (default 4000), `SUMMARIZE_KEEP_MESSAGES` (default 20), `LOG_LEVEL`.

Config files: `config/api_allowlist.json`, `config/mcp_servers.json` (both start as `[]`), `config/system_prompt.md`.

**Optional, zero code:** LangChain agents pick up LangSmith tracing automatically from `LANGCHAIN_TRACING_V2=true` + `LANGSMITH_API_KEY` env vars, if the team ever wants tool-call observability on the local model — not a requirement here, just a free option worth knowing about.

### Dependencies (via `uv`, `pyproject.toml`)

Runtime: `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `aiosqlite`, `httpx` (for allowlisted REST calls), `langchain` + `langchain-openai` (agent loop, tool-calling, and the custom `base_url`/`api_key` chat model), `langchain[mcp]` (MCP integration — **beta**, pin the exact version), a SQLite-backed LangGraph checkpointer package (**flagged, unconfirmed** — expected `langgraph-checkpoint-sqlite`, see "Data model" above; verify at implementation time), `pyyaml` (skills frontmatter parsing, see Part 2). `openai`/`mcp`/`httpx2` are no longer imported directly anywhere in this project's own code; they may still appear as transitive dependencies of the `langchain-openai`/`langchain[mcp]` packages.
Dev/test: `pytest`, `pytest-asyncio` (`asyncio_mode = "auto"` in `pyproject.toml`), `respx` (httpx mocking).

Target Python 3.12 via `uv venv --python 3.12` (managed automatically by `uv`).

### Verification performed

All 110 automated tests pass (`uv run pytest`), written test-first throughout. In addition:

- **Milestone 1**: server starts, static webview serves, WebSocket auth handshake rejects a wrong token and accepts the correct one.
- **Milestone 2**: `POST /tasks` without a token → `401`; with a token → `202` + `task_id`; polling `GET /tasks/{id}` showed the real `pending → running → failed` lifecycle (failure expected — the smoke test used a placeholder, unreachable `LLM_BASE_URL`) with the connection error captured in `error`, proving the background task never crashes the server.
- **Milestone 3**: server starts cleanly with the (empty) REST allowlist wired into the agent's tool list.
- **Milestone 4**: server starts cleanly with a real local MCP stdio server (the same `dummy_mcp_server.py` fixture used in tests) configured in `mcp_servers.json`, with no connection warnings logged. Later re-verified against a real remote `streamable_http` MCP server (a public test/demo server), discovering 4 real tools with no code changes needed.
- **Real end-to-end LLM round trip**: verified against a locally-run Ollama instance (model `qwen3:4b`, served over its OpenAI-compatible endpoint at `http://localhost:11434/v1`) — the agent correctly called `write_file` then `read_file` as real tool calls in response to a natural-language instruction, and the file was confirmed to actually exist in the sandbox afterward.
- **`GET /mcp/status`**: verified live, returning the real connected server and its 4 discovered tool names.

### Deferred (not built, by design)

- **Runtime-editable system prompt** — currently a static file read once at startup.
- **Runtime skill management** (create/edit/toggle a skill via API or UI) — skills are files on disk, loaded once at startup; see Part 2's own "Out of scope" section.

---

## Part 2: Skills System — Design

### Context

storage-agent is a self-hosted AI agent dedicated to a storage infrastructure team, covering PureStorage, NetApp, Dell ECS, and general NAS/SAN/object storage administration — shares, DFS, export policies, naming conventions, and vendor-specific procedures. The core system (sandboxed file tools, an allowlisted REST-API tool, an MCP client, a chat webview, and a trigger REST API) is already built and running (see Part 1 above).

"Skills" — named, loadable domain knowledge — were deferred during the original build, with `agent/prompt.py::build_system_prompt()` deliberately left as a discrete function specifically so this could be added later without restructuring. This spec designs that addition.

A skill in this project is **instructional content, not executable code**: vendor-specific conventions, safety rules, API/CLI syntax references, and procedures — text that shapes how the agent uses the tools it already has (REST, MCP, file tools), not a new way for it to act. The project deliberately has no code-execution tool, and this design doesn't add one; that was an explicit scope decision, not an oversight.

### Goals

- Let the team encode storage-domain knowledge (per vendor: PureStorage, NetApp, ECS, etc., plus cross-cutting rules) as versioned, reviewable files the agent loads into context.
- Keep small, universally-relevant knowledge ("always confirm before deleting a volume") available on every request without extra latency.
- Keep large, situational reference material (a full ONTAP CLI reference, a REST API cheat sheet) out of every request's prompt by default — this matters concretely here, since the team runs this against a local, CPU-bound model where prompt size directly costs response time.
- Fit the existing extensibility pattern (LangChain `tool(...)`-wrapped closures + `build_X_tools()`, combined into `create_agent()`'s tool list by `agent/core.py::build_agent()`) rather than introducing a parallel mechanism.

### Architecture

```
skills/
  team-safety-rules/
    SKILL.md              # always_on: true — short, universal
  purestorage/
    SKILL.md              # always_on: false
    reference/
      rest-api-cheatsheet.md
      naming-conventions.md
  netapp/
    SKILL.md
    reference/
      ontap-cli-reference.md
```

Two triggering modes, chosen per skill via frontmatter:

- **Always-on**: the skill's full instructions are appended to the system prompt once at startup. For short, universally-relevant knowledge.
- **On-demand**: only the skill's name + one-line description appear in the system prompt (a short catalog). The LLM calls a `load_skill(name)` tool to pull in the full instructions when a task actually needs them — the same progressive-disclosure pattern Claude Code's own skills use, and one the team is already directly familiar with from using this very tool.

### Components

- **`agent/skills.py`** (new):
  - `Skill` dataclass: `name: str`, `description: str`, `always_on: bool`, `instructions: str`, `reference_dir: Path | None`.
  - `load_skills(skills_dir: Path) -> list[Skill]` — scans `skills/*/SKILL.md`. For each, parses YAML frontmatter (`name`, `description`, `always_on`, via `pyyaml`'s `yaml.safe_load`) plus the Markdown body as `instructions`. Sets `reference_dir` to the skill's own directory if a `reference/` subdirectory exists, else `None`.
    - A malformed `SKILL.md` (missing `name` or `description`) is skipped with a logged warning — the same resilience pattern `build_mcp_tools()` already uses for a misconfigured MCP server: one bad skill folder doesn't block the others or the app.
    - A duplicate `name` across skill folders raises `DuplicateSkillName` at startup — a real config bug, fail fast (mirrors the `DuplicateToolName` check inside `agent/core.py::build_agent()`).
  - Returns the full list; callers partition into always-on vs. on-demand by the `always_on` flag.

- **`agent/tools/skills.py`** (new) — `build_skill_tools(on_demand_skills: list[Skill]) -> list[BaseTool]`. Follows the exact same shape as `build_file_tools`/`build_rest_tools`/`build_mcp_tools`: each tool is a LangChain `tool(...)`-wrapped closure over `skills_by_name`, not a custom `Tool` type. Returns `[]` if `on_demand_skills` is empty (no pointless tools registered, same convention as REST/MCP with empty config). Otherwise returns two tools:
  - **`load_skill(name)`** — looks up the skill by name among on-demand skills. Returns its `instructions`, followed by a listing of any reference files under `reference_dir` (relative paths) with a note to use `read_skill_file` for one. Unknown name → raises `ValueError`, caught by `ToolErrorMiddleware` (LangChain built-in, registered once on the agent — see Part 1's "Core interface" section) and turned into `"Error: unknown skill 'x'"`, never a crash.
  - **`read_skill_file(skill, path)`** — resolves `path` within that skill's `reference_dir` using `resolve_in_sandbox()` (imported from `agent/tools/files.py`, reused as-is — the containment/traversal-safety requirement is identical, so this isn't reimplemented). Returns the file's content, size-capped and truncated using the same convention as `read_file` (`DEFAULT_MAX_READ_BYTES`). Unknown skill, unknown path, or a traversal attempt → raises (`ValueError`/`FileNotFoundError`), converted to an `"Error: ..."` string by the same `ToolErrorMiddleware`, not a crash.

- **`agent/prompt.py::build_system_prompt(base_prompt, always_on_skills, on_demand_skills)`** (signature extended — this is the seam the original design left for exactly this): appends each always-on skill's full `instructions` as its own section, then (if any on-demand skills exist) a short catalog: one line per skill, `name — description`, with a note that `load_skill` loads one's full instructions.

- **`main.py`**: new `SKILLS_PATH` setting (default `./skills`, matching `SANDBOX_ROOT`/`API_ALLOWLIST_PATH`/`MCP_SERVERS_PATH`'s convention). At startup: `load_skills(Path(settings.SKILLS_PATH))`, partition by `always_on`, pass both lists into `build_system_prompt()`, and pass `build_skill_tools(on_demand_skills)`'s output alongside the file/REST/MCP tool lists into `agent/core.py::build_agent()`, which concatenates them (and checks for name collisions) before calling `create_agent()` (see Part 1's "Core interface" and "The agent core loop" sections).

### Data flow example

1. A user asks the agent to provision a PureStorage volume.
2. The LLM sees `purestorage — PureStorage FlashArray/FlashBlade provisioning...` in the system prompt's on-demand catalog and calls `load_skill({"name": "purestorage"})`.
3. The tool result gives it the skill's conventions/procedures, plus: `Reference files available (use read_skill_file to view): rest-api-cheatsheet.md, naming-conventions.md` — paths relative to the skill's own `reference/` directory, not including the `reference/` prefix.
4. If the task needs exact REST syntax, the LLM calls `read_skill_file({"skill": "purestorage", "path": "rest-api-cheatsheet.md"})` to pull that specific file rather than having it in context by default.
5. The LLM proceeds using the (already-existing) REST-allowlist tools, now informed by the loaded skill's conventions.

### Error handling summary

| Condition | Behavior |
|---|---|
| Malformed `SKILL.md` (missing `name`/`description`) | Skipped, logged warning, app starts normally |
| Duplicate skill `name` | `DuplicateSkillName` raised at startup (fail fast) |
| `load_skill` with unknown name | Raises `ValueError`; `ToolErrorMiddleware` turns it into `"Error: unknown skill '...'"`, not a crash |
| `read_skill_file` with unknown skill/path, or a traversal attempt | Raises (reuses `resolve_in_sandbox`); same `ToolErrorMiddleware` turns it into an `"Error: ..."` string, not a crash |
| Zero on-demand skills configured | `load_skill`/`read_skill_file` not registered as tools at all |
| Zero skills configured at all (`skills/` empty or missing) | System prompt unchanged from today; app starts normally |

### Testing

- `tests/test_skills.py`: `load_skills()` — parses a valid `SKILL.md` correctly (including `reference_dir` detection), skips a malformed one with a warning (doesn't block loading the rest), raises `DuplicateSkillName` on a repeated `name`. `load_skill`/`read_skill_file` tool behavior, including every row of the error table above, using `tmp_path` fixtures (never the real project tree) — same pattern as `tests/test_files_tool.py`.
- `tests/test_prompt.py`: extend for the new `build_system_prompt(base_prompt, always_on_skills, on_demand_skills)` signature — always-on instructions appended, on-demand catalog appended with correct name/description lines, no catalog section when there are no on-demand skills.
- Live verification: create real example skill folders (an always-on `team-safety-rules` and an on-demand `purestorage` with a reference file), run the local Ollama-backed server, and drive an actual chat message that should trigger `load_skill` then `read_skill_file`, confirming the full round trip end-to-end (matching how every other feature in this project was verified during its own build).

### Dependencies

Adds `pyyaml` (frontmatter parsing) — a standard, minimal, well-known dependency; avoids a hand-rolled parser being wrong on an edge case like a `description` field containing a colon.

### Out of scope (explicit)

- **Script execution.** Considered and explicitly declined — storage-agent has no code-execution tool today, and adding one was judged a meaningfully bigger, more security-sensitive addition than this design warrants for a storage-infra agent. Skills are instructions + reference data only.
- **Runtime skill management (create/edit/toggle via API or UI).** Skills are managed as files on disk, loaded once at startup — consistent with how `config/api_allowlist.json` and `config/mcp_servers.json` already work (edit the file, restart). Not addressed here; would be a separate, later design if actually needed.

---

## Part 3: Skills System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Skills system letting the storage team encode vendor/domain knowledge (PureStorage, NetApp, ECS, NAS/SAN/object conventions) as always-on or on-demand instructions the agent loads into context.

**Architecture:** Skills are directories under `skills/` with a `SKILL.md` (YAML frontmatter + Markdown body) and optional `reference/` files. Always-on skills are appended to the system prompt at startup; on-demand skills appear as a name+description catalog and are pulled in via new `load_skill`/`read_skill_file` tools, following the exact same LangChain `tool(...)`/`build_X_tools()` pattern every other tool source in this project already uses.

**Tech Stack:** Python, `pyyaml` (new dependency, frontmatter parsing), `langchain` (tool wrapping via `langchain.tools.tool` — already a project dependency per Part 1; `langchain[mcp]` is beta, pin the version), existing FastAPI/pytest/uv toolchain.

**Spec:** Part 2 above.

### Global Constraints

- Skills are instructions + reference data only — no script execution (explicit scope decision in the spec).
- Skill directory convention: `skills/<name>/SKILL.md` + optional `skills/<name>/reference/`.
- Frontmatter parsed with `pyyaml`'s `yaml.safe_load`, not hand-rolled parsing.
- A malformed `SKILL.md` is skipped with a logged warning; a duplicate skill `name` raises at startup (fail fast) — same resilience/fail-fast split as MCP servers, and as `build_agent()`'s own duplicate-tool-name check, elsewhere in this project.
- `read_skill_file`'s path containment reuses `resolve_in_sandbox()` from `agent/tools/files.py` — not reimplemented.
- Zero on-demand skills → `load_skill`/`read_skill_file` are not registered as tools at all.
- New `SKILLS_PATH` setting, default `./skills`, matching the existing `SANDBOX_ROOT`/`API_ALLOWLIST_PATH`/`MCP_SERVERS_PATH` convention.

### Review Focus

- A skill folder present but with no `SKILL.md` file at all (not malformed — just absent) → `load_skills()` must skip it silently, not crash.
- A skill's `reference/` directory exists but is empty → `load_skill` must not claim reference files are available when there are none.
- Reading a reference file larger than the size cap → must truncate with a note, same convention as `read_file`, not silently cut off unremarked.
- An always-on skill with unusually large instructions → logged warning at startup (not a hard block) — this directly matters here since it inflates every single prompt on the team's CPU-bound local model.
- `always_on` written as a quoted YAML string (`"false"`) instead of a real boolean → must not silently evaluate as truthy just because it's a non-empty string.

---

#### Task 1: `agent/skills.py` — Skill loading

**Files:**
- Create: `agent/skills.py`
- Test: `tests/test_skills.py`

**Interfaces:**
- Consumes: nothing new (stdlib `pathlib`, `logging`, `dataclasses`; `yaml` from the new `pyyaml` dependency).
- Produces: `Skill` (frozen dataclass: `name: str`, `description: str`, `always_on: bool`, `instructions: str`, `reference_dir: Path | None`), `DuplicateSkillName` (Exception), `load_skills(skills_dir: Path) -> list[Skill]`. Later tasks import all three from `agent.skills`.

- [ ] **Step 1: Add the `pyyaml` dependency**

Run: `uv add pyyaml`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_skills.py`:

```python
import logging

import pytest

from agent.skills import DuplicateSkillName, load_skills


def write_skill(base, folder_name, name=None, description="A skill.", always_on=False, body="Do the thing.", reference_files=None):
    name = name or folder_name
    skill_dir = base / folder_name
    skill_dir.mkdir()
    always_on_line = str(always_on).lower()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nalways_on: {always_on_line}\n---\n{body}\n"
    )
    if reference_files:
        ref_dir = skill_dir / "reference"
        ref_dir.mkdir()
        for filename, content in reference_files.items():
            (ref_dir / filename).write_text(content)
    return skill_dir


def test_load_skills_parses_valid_skill(tmp_path):
    write_skill(tmp_path, "purestorage", description="PureStorage conventions.", always_on=False, body="Full instructions here.")

    skills = load_skills(tmp_path)

    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "purestorage"
    assert skill.description == "PureStorage conventions."
    assert skill.always_on is False
    assert skill.instructions == "Full instructions here."
    assert skill.reference_dir is None


def test_load_skills_detects_reference_dir(tmp_path):
    write_skill(tmp_path, "purestorage", reference_files={"api.md": "cheatsheet"})

    skills = load_skills(tmp_path)

    assert skills[0].reference_dir == tmp_path / "purestorage" / "reference"


def test_load_skills_returns_empty_list_for_missing_dir(tmp_path):
    assert load_skills(tmp_path / "does-not-exist") == []


def test_load_skills_skips_folder_without_skill_md(tmp_path):
    (tmp_path / "empty-folder").mkdir()
    write_skill(tmp_path, "purestorage")

    skills = load_skills(tmp_path)

    assert [s.name for s in skills] == ["purestorage"]


def test_load_skills_skips_missing_frontmatter(tmp_path, caplog):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("just some text, no frontmatter")
    write_skill(tmp_path, "purestorage")

    with caplog.at_level(logging.WARNING):
        skills = load_skills(tmp_path)

    assert [s.name for s in skills] == ["purestorage"]
    assert "broken" in caplog.text


def test_load_skills_skips_missing_name_or_description(tmp_path, caplog):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: broken\n---\nbody")
    write_skill(tmp_path, "purestorage")

    with caplog.at_level(logging.WARNING):
        skills = load_skills(tmp_path)

    assert [s.name for s in skills] == ["purestorage"]


def test_load_skills_skips_non_boolean_always_on(tmp_path, caplog):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        '---\nname: broken\ndescription: d\nalways_on: "false"\n---\nbody'
    )
    write_skill(tmp_path, "purestorage")

    with caplog.at_level(logging.WARNING):
        skills = load_skills(tmp_path)

    assert [s.name for s in skills] == ["purestorage"]


def test_load_skills_raises_on_duplicate_name(tmp_path):
    write_skill(tmp_path, "purestorage", name="purestorage")
    write_skill(tmp_path, "purestorage-2", name="purestorage", description="dup")

    with pytest.raises(DuplicateSkillName):
        load_skills(tmp_path)


def test_load_skills_warns_on_large_always_on_instructions(tmp_path, caplog):
    write_skill(tmp_path, "big", always_on=True, body="x" * 25_000)

    with caplog.at_level(logging.WARNING):
        skills = load_skills(tmp_path)

    assert len(skills) == 1
    assert "big" in caplog.text
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_skills.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.skills'`

- [ ] **Step 4: Write the implementation**

Create `agent/skills.py`:

```python
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

MAX_ALWAYS_ON_INSTRUCTIONS_CHARS = 20_000


class DuplicateSkillName(Exception):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    always_on: bool
    instructions: str
    reference_dir: Path | None


def load_skills(skills_dir: Path) -> list["Skill"]:
    skills: dict[str, Skill] = {}

    if not skills_dir.is_dir():
        return []

    for skill_dir in sorted(skills_dir.iterdir()):
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            continue

        skill = _parse_skill_file(skill_file, skill_dir)
        if skill is None:
            continue

        if skill.name in skills:
            raise DuplicateSkillName(f"duplicate skill name: {skill.name!r}")
        skills[skill.name] = skill

    return list(skills.values())


def _parse_skill_file(skill_file: Path, skill_dir: Path) -> "Skill | None":
    text = skill_file.read_text()
    if not text.startswith("---"):
        logger.warning("skipping skill %r: missing frontmatter", skill_dir.name)
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("skipping skill %r: missing frontmatter", skill_dir.name)
        return None

    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        logger.warning("skipping skill %r: invalid YAML frontmatter", skill_dir.name, exc_info=True)
        return None

    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not name or not description:
        logger.warning("skipping skill %r: missing name or description", skill_dir.name)
        return None

    always_on = frontmatter.get("always_on", False)
    if not isinstance(always_on, bool):
        logger.warning(
            "skipping skill %r: always_on must be a YAML boolean, got %r", skill_dir.name, always_on
        )
        return None

    instructions = parts[2].strip()
    if always_on and len(instructions) > MAX_ALWAYS_ON_INSTRUCTIONS_CHARS:
        logger.warning(
            "skill %r is always_on with %d chars of instructions (over %d) -- "
            "this is appended to every request's prompt",
            skill_dir.name, len(instructions), MAX_ALWAYS_ON_INSTRUCTIONS_CHARS,
        )

    reference_dir = skill_dir / "reference"
    if not reference_dir.is_dir():
        reference_dir = None

    return Skill(
        name=name,
        description=description,
        always_on=always_on,
        instructions=instructions,
        reference_dir=reference_dir,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_skills.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add agent/skills.py tests/test_skills.py pyproject.toml uv.lock
git commit -m "Add skill loading (agent/skills.py)"
```

---

#### Task 2: `agent/tools/skills.py` — `load_skill` and `read_skill_file` tools

**Files:**
- Create: `agent/tools/skills.py`
- Test: `tests/test_skills_tool.py`

**Interfaces:**
- Consumes: `Skill` from `agent.skills` (Task 1); `tool` from `langchain.tools`; `resolve_in_sandbox` from `agent.tools.files` (existing).
- Produces: `build_skill_tools(on_demand_skills: list[Skill], max_reference_bytes: int = 1_000_000) -> list[BaseTool]`. Task 4 imports this. Unknown-name/path errors are raised as plain exceptions (`ValueError`/`FileNotFoundError`), not pre-formatted into `"Error: ..."` strings — LangChain's built-in `ToolErrorMiddleware` (Task 4 wires this onto the agent, via `agent/core.py::build_agent()`) does that conversion once, for every tool, with no project-owned middleware module.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_skills_tool.py`:

```python
import pytest

from agent.skills import Skill
from agent.tools.skills import build_skill_tools


def make_skill(tmp_path, name="purestorage", always_on=False, reference_files=None):
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    reference_dir = None
    if reference_files:
        reference_dir = skill_dir / "reference"
        reference_dir.mkdir()
        for filename, content in reference_files.items():
            (reference_dir / filename).write_text(content)
    return Skill(
        name=name,
        description="A skill.",
        always_on=always_on,
        instructions="Full instructions.",
        reference_dir=reference_dir,
    )


def test_build_skill_tools_empty_list_returns_no_tools():
    assert build_skill_tools([]) == []


def test_build_skill_tools_returns_load_and_read_tools(tmp_path):
    skill = make_skill(tmp_path)
    tools = {t.name for t in build_skill_tools([skill])}
    assert tools == {"load_skill", "read_skill_file"}


async def test_load_skill_returns_instructions(tmp_path):
    skill = make_skill(tmp_path)
    tools = {t.name: t for t in build_skill_tools([skill])}

    result = await tools["load_skill"].ainvoke({"name": "purestorage"})

    assert result == "Full instructions."


async def test_load_skill_lists_reference_files(tmp_path):
    skill = make_skill(tmp_path, reference_files={"cheatsheet.md": "content"})
    tools = {t.name: t for t in build_skill_tools([skill])}

    result = await tools["load_skill"].ainvoke({"name": "purestorage"})

    assert "Full instructions." in result
    assert "cheatsheet.md" in result
    assert "Reference files available" in result


async def test_load_skill_omits_reference_note_when_dir_empty(tmp_path):
    skill_dir = tmp_path / "purestorage"
    skill_dir.mkdir()
    (skill_dir / "reference").mkdir()
    skill = Skill(
        name="purestorage", description="d", always_on=False,
        instructions="Full instructions.", reference_dir=skill_dir / "reference",
    )
    tools = {t.name: t for t in build_skill_tools([skill])}

    result = await tools["load_skill"].ainvoke({"name": "purestorage"})

    assert result == "Full instructions."


async def test_load_skill_unknown_name_raises(tmp_path):
    skill = make_skill(tmp_path)
    tools = {t.name: t for t in build_skill_tools([skill])}

    with pytest.raises(ValueError, match="unknown skill"):
        await tools["load_skill"].ainvoke({"name": "nonexistent"})


async def test_read_skill_file_returns_content(tmp_path):
    skill = make_skill(tmp_path, reference_files={"cheatsheet.md": "the content"})
    tools = {t.name: t for t in build_skill_tools([skill])}

    result = await tools["read_skill_file"].ainvoke({"skill": "purestorage", "path": "cheatsheet.md"})

    assert result == "the content"


async def test_read_skill_file_truncates_large_files(tmp_path):
    skill = make_skill(tmp_path, reference_files={"big.md": "x" * 100})
    tools = {t.name: t for t in build_skill_tools([skill], max_reference_bytes=50)}

    result = await tools["read_skill_file"].ainvoke({"skill": "purestorage", "path": "big.md"})

    assert "truncated at 50 bytes" in result


async def test_read_skill_file_rejects_traversal(tmp_path):
    skill = make_skill(tmp_path, reference_files={"cheatsheet.md": "content"})
    tools = {t.name: t for t in build_skill_tools([skill])}

    with pytest.raises(Exception):
        await tools["read_skill_file"].ainvoke({"skill": "purestorage", "path": "../../SKILL.md"})


async def test_read_skill_file_unknown_skill_raises(tmp_path):
    skill = make_skill(tmp_path)
    tools = {t.name: t for t in build_skill_tools([skill])}

    with pytest.raises(ValueError, match="unknown skill"):
        await tools["read_skill_file"].ainvoke({"skill": "nonexistent", "path": "x.md"})


async def test_read_skill_file_unknown_path_raises(tmp_path):
    skill = make_skill(tmp_path, reference_files={"cheatsheet.md": "content"})
    tools = {t.name: t for t in build_skill_tools([skill])}

    with pytest.raises(FileNotFoundError):
        await tools["read_skill_file"].ainvoke({"skill": "purestorage", "path": "missing.md"})
```

Note: these tests exercise the tools directly and assert they *raise* rather than return `"Error: ..."` strings — the string conversion now happens once, via LangChain's built-in `ToolErrorMiddleware` (its wiring is asserted in `test_core.py`, Task 4 — there's no project-owned middleware module to write a dedicated test file for), not inside each tool.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_skills_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.tools.skills'`

- [ ] **Step 3: Write the implementation**

Create `agent/tools/skills.py`:

```python
from langchain_core.tools import BaseTool
from langchain.tools import tool

from agent.skills import Skill
from agent.tools.files import resolve_in_sandbox

DEFAULT_MAX_REFERENCE_FILE_BYTES = 1_000_000


def build_skill_tools(
    on_demand_skills: list[Skill],
    max_reference_bytes: int = DEFAULT_MAX_REFERENCE_FILE_BYTES,
) -> list[BaseTool]:
    if not on_demand_skills:
        return []

    skills_by_name = {skill.name: skill for skill in on_demand_skills}

    async def load_skill(name: str) -> str:
        """Load a storage-domain skill's instructions by name."""
        skill = skills_by_name.get(name)
        if skill is None:
            raise ValueError(f"unknown skill: {name!r}")

        result = skill.instructions

        if skill.reference_dir is not None:
            files = sorted(
                str(p.relative_to(skill.reference_dir))
                for p in skill.reference_dir.rglob("*")
                if p.is_file()
            )
            if files:
                file_list = "\n".join(f"- {f}" for f in files)
                result += (
                    f"\n\nReference files available (use read_skill_file to view one):\n{file_list}"
                )

        return result

    async def read_skill_file(skill: str, path: str) -> str:
        """Read one reference file belonging to a loaded skill. `path` is relative to the skill's reference/ directory."""
        skill_obj = skills_by_name.get(skill)
        if skill_obj is None:
            raise ValueError(f"unknown skill: {skill!r}")
        if skill_obj.reference_dir is None:
            raise ValueError(f"skill {skill!r} has no reference files")

        target = resolve_in_sandbox(skill_obj.reference_dir, path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path}")

        data = target.read_bytes()
        text = data[:max_reference_bytes].decode(errors="replace")
        if len(data) > max_reference_bytes:
            text += f"\n... truncated at {max_reference_bytes} bytes"
        return text

    return [
        tool(load_skill, name="load_skill"),
        tool(read_skill_file, name="read_skill_file"),
    ]
```

Unknown name/path and traversal attempts raise plain exceptions here (`ValueError`, `FileNotFoundError`) — there is no per-tool `guard_errors`-style wrapping anymore. LangChain's built-in `ToolErrorMiddleware`, wired onto the agent in Task 4 (`agent/core.py::build_agent()`), is what turns any tool's exception into an `"Error: ..."` string; that's a single, project-wide behavior rather than something each tool — or any project-owned middleware module — implements for itself.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_skills_tool.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add agent/tools/skills.py tests/test_skills_tool.py
git commit -m "Add load_skill and read_skill_file tools"
```

---

#### Task 3: Extend `agent/prompt.py::build_system_prompt()`

**Files:**
- Modify: `agent/prompt.py` (entire file — currently 3 lines)
- Modify: `tests/test_prompt.py` (add tests; keep the existing one)

**Interfaces:**
- Consumes: `Skill` from `agent.skills` (Task 1).
- Produces: `build_system_prompt(base_prompt: str, always_on_skills: list[Skill] | None = None, on_demand_skills: list[Skill] | None = None) -> str`. Task 4 calls this with both lists.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prompt.py` (the existing `test_build_system_prompt_returns_base_prompt_unchanged` stays as-is — new signature is backward compatible via defaults):

```python
from agent.skills import Skill


def make_skill(name, description="d", always_on=False, instructions="do the thing"):
    return Skill(
        name=name, description=description, always_on=always_on,
        instructions=instructions, reference_dir=None,
    )


def test_build_system_prompt_appends_always_on_skill_instructions():
    skill = make_skill("safety", instructions="Always confirm before deleting a volume.")

    result = build_system_prompt("base prompt", always_on_skills=[skill])

    assert "base prompt" in result
    assert "Always confirm before deleting a volume." in result


def test_build_system_prompt_appends_on_demand_catalog():
    skill = make_skill("purestorage", description="PureStorage conventions.")

    result = build_system_prompt("base prompt", on_demand_skills=[skill])

    assert "purestorage" in result
    assert "PureStorage conventions." in result
    assert "load_skill" in result


def test_build_system_prompt_no_catalog_section_when_no_on_demand_skills():
    always_on = make_skill("safety", instructions="Confirm before deleting.")

    result = build_system_prompt("base prompt", always_on_skills=[always_on])

    assert "Available skills" not in result


def test_build_system_prompt_combines_both_kinds():
    always_on = make_skill("safety", instructions="Confirm before deleting.")
    on_demand = make_skill("purestorage", description="PureStorage conventions.")

    result = build_system_prompt("base prompt", always_on_skills=[always_on], on_demand_skills=[on_demand])

    assert "Confirm before deleting." in result
    assert "purestorage" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: FAIL — `TypeError: build_system_prompt() got an unexpected keyword argument 'always_on_skills'`

- [ ] **Step 3: Write the implementation**

Replace `agent/prompt.py` entirely:

```python
from agent.skills import Skill


def build_system_prompt(
    base_prompt: str,
    always_on_skills: list[Skill] | None = None,
    on_demand_skills: list[Skill] | None = None,
) -> str:
    always_on_skills = always_on_skills or []
    on_demand_skills = on_demand_skills or []

    sections = [base_prompt]

    for skill in always_on_skills:
        sections.append(f"## Skill: {skill.name}\n\n{skill.instructions}")

    if on_demand_skills:
        catalog_lines = "\n".join(f"- {s.name} — {s.description}" for s in on_demand_skills)
        sections.append(
            "## Available skills\n\n"
            "Call load_skill(name) to load full instructions for one of these "
            "when relevant to the task:\n\n" + catalog_lines
        )

    return "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: 5 passed (the original test plus the 4 new ones)

- [ ] **Step 5: Commit**

```bash
git add agent/prompt.py tests/test_prompt.py
git commit -m "Extend build_system_prompt to inject skills"
```

---

#### Task 4: Wire skills into `main.py`

**Files:**
- Modify: `settings.py`
- Modify: `main.py`

**Interfaces:**
- Consumes: `load_skills` (Task 1), `build_skill_tools` (Task 2), `build_system_prompt` new signature (Task 3); `agent/core.py::build_agent()` (already present from the LangChain migration described in Part 1 — it concatenates the tool lists itself, with its own inline duplicate-name check, so there's no separate `agent/registry.py` to import).
- Produces: nothing new for later tasks — this is the integration point. Task 5 relies on `SKILLS_PATH` defaulting to `./skills`.

- [ ] **Step 1: Add `SKILLS_PATH` setting**

In `settings.py`, add this line among the other optional path settings (after `MCP_SERVERS_PATH`):

```python
    SKILLS_PATH: str = "./skills"
```

- [ ] **Step 2: Wire skill loading into the lifespan**

In `main.py`, add to the imports:

```python
from agent.prompt import build_system_prompt  # already imported — no change if already present
from agent.skills import load_skills
from agent.tools.skills import build_skill_tools
```

Replace the body of `lifespan()` from the `base_prompt = ...` line through the `app.state.agent = ...` line with:

```python
    skills = load_skills(Path(settings.SKILLS_PATH))
    always_on_skills = [s for s in skills if s.always_on]
    on_demand_skills = [s for s in skills if not s.always_on]

    base_prompt = Path(settings.SYSTEM_PROMPT_PATH).read_text()
    system_prompt = build_system_prompt(base_prompt, always_on_skills, on_demand_skills)

    file_tools = build_file_tools(sandbox_root)
    skill_tools = build_skill_tools(on_demand_skills)

    # long-lived resources (pooled HTTP clients, MCP connections) that must
    # outlive the request that created them and be closed together at shutdown
    resources_stack = AsyncExitStack()
    app.state.resources_stack = resources_stack

    await seed_api_configs(app.state.db, Path(settings.API_ALLOWLIST_PATH))
    api_configs = await list_enabled_api_configs(app.state.db)
    rest_tools = await build_rest_tools(api_configs, resources_stack)

    mcp_servers = load_mcp_server_configs(Path(settings.MCP_SERVERS_PATH))
    mcp_tools = await build_mcp_tools(mcp_servers, resources_stack)
    app.state.mcp_status = summarize_connections(mcp_servers, mcp_tools)

    app.state.agent = build_agent(
        model,
        file_tools + rest_tools + mcp_tools + skill_tools,
        system_prompt,
        settings.MAX_TOOL_TURNS,
        checkpointer,
        settings.SUMMARIZE_TRIGGER_TOKENS,
        settings.SUMMARIZE_KEEP_MESSAGES,
    )
```

(The rest of `lifespan()` — building `model` via `init_chat_model(...)` and `checkpointer` (see Part 1's "Data model" section), `yield`, and shutdown — is unchanged. `model`, `checkpointer`, and `build_agent` already exist from the LangChain migration described in Part 1; `build_agent()` itself concatenates the tool lists and checks for name collisions, so this task just adds `skill_tools` to the list passed in and threads skill instructions into `system_prompt`.)

- [ ] **Step 3: Run the full test suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass (no existing test touches `main.py` directly, so this is a regression check, not new coverage)

- [ ] **Step 4: Live smoke test — app still starts cleanly with an empty `skills/` directory**

```bash
mkdir -p skills
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill 2>/dev/null
rm -f data/storage_agent.db
uv run uvicorn main:app --port 8000 &> /tmp/storage-agent-run.log &
sleep 3
curl -s -o /dev/null -w "readiness: %{http_code}\n" http://localhost:8000/
grep -i "error\|traceback" /tmp/storage-agent-run.log || echo "no errors in startup log"
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill
```

Expected: `readiness: 200`, no errors in the log (an empty `skills/` directory is valid — `load_skills` returns `[]`).

- [ ] **Step 5: Commit**

```bash
git add settings.py main.py
git commit -m "Wire skills into app startup"
```

---

#### Task 5: Example skills and end-to-end verification

**Files:**
- Create: `skills/team-safety-rules/SKILL.md`
- Create: `skills/purestorage/SKILL.md`
- Create: `skills/purestorage/reference/rest-api-cheatsheet.md`
- Modify: `README.md`
- Modify: this document's Part 2 (status line only)

**Interfaces:**
- Consumes: the fully wired app from Task 4.
- Produces: nothing further downstream — this is the final task.

- [ ] **Step 1: Create the always-on safety-rules skill**

Create `skills/team-safety-rules/SKILL.md`:

```markdown
---
name: team-safety-rules
description: Universal safety rules for all storage operations.
always_on: true
---

- Always confirm with the user in plain language before deleting, unprovisioning, or resizing down any volume, share, or export — these actions can be destructive and are not easily reversible.
- Never disable an export policy or share access rule without explicit confirmation of the exact policy/rule name and target.
- When unsure which array, filer, or cluster a request applies to, ask rather than guessing.
```

- [ ] **Step 2: Create the on-demand PureStorage skill with a reference file**

Create `skills/purestorage/SKILL.md`:

```markdown
---
name: purestorage
description: PureStorage FlashArray/FlashBlade provisioning, naming conventions, and REST API usage.
always_on: false
---

Volume names follow the pattern `<env>-<app>-<purpose>-<size>` (e.g. `prod-erp-data-500g`).

For FlashArray REST API calls, use the `purestorage_*` tools if configured in the REST allowlist. See reference/rest-api-cheatsheet.md for endpoint details.
```

Create `skills/purestorage/reference/rest-api-cheatsheet.md`:

```markdown
# PureStorage FlashArray REST API quick reference

- `GET /api/2.x/volumes` — list volumes
- `POST /api/2.x/volumes` — create a volume, body: `{"names": ["..."], "provisioned": <bytes>}`
- `DELETE /api/2.x/volumes/<name>` — destroy (soft-delete, recoverable for a period)
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (these example skills aren't referenced by any automated test, only used for the live check below)

- [ ] **Step 4: Live end-to-end verification against the real local model**

```bash
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill 2>/dev/null
rm -f data/storage_agent.db
uv run uvicorn main:app --port 8000 &> /tmp/storage-agent-run.log &
sleep 3
curl -s http://localhost:8000/mcp/status -H "Authorization: Bearer dev-token" > /dev/null
```

Then drive a real chat message that should trigger `load_skill` (adjust the token if `.env`'s `API_TOKEN` differs from `dev-token`):

```bash
uv run python -c "
import asyncio, json
import websockets

async def main():
    async with websockets.connect('ws://127.0.0.1:8000/ws/chat') as ws:
        await ws.send(json.dumps({'token': 'dev-token'}))
        await ws.send(json.dumps({'type': 'message', 'conversation_id': None, 'content': 'What naming convention do we use for PureStorage volumes?'}))
        while True:
            reply = json.loads(await ws.recv())
            print(reply)
            if reply['type'] in ('final', 'error'):
                break

asyncio.run(main())
"
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill
```

Expected: a `tool_call` event for `load_skill` with `{"name": "purestorage"}`, its `tool_result` containing the naming-convention text, and a `final` answer reflecting it. (A small local model may also call `read_skill_file` for the cheatsheet, or may answer directly from `load_skill`'s result — both are correct; the naming convention is in `SKILL.md` itself, so `load_skill` alone is sufficient.)

- [ ] **Step 5: Update README**

In `README.md`, add a row to the environment variables table (after the `MCP_SERVERS_PATH` row):

```markdown
| `SKILLS_PATH` | no | `./skills` | Directory of skill folders (domain knowledge injected into the agent's context). |
```

Add a new subsection after "### MCP servers (`config/mcp_servers.json`)":

```markdown
### Skills (`skills/`)

Each subdirectory of `skills/` is one skill: a `SKILL.md` with YAML frontmatter (`name`, `description`, `always_on`) and a Markdown body, plus an optional `reference/` directory of additional files.

```markdown
---
name: purestorage
description: PureStorage FlashArray/FlashBlade provisioning, naming conventions, and REST API usage.
always_on: false
---

Full instructions here...
```

- `always_on: true` skills are appended to the system prompt on every request — keep these short (safety rules, universal conventions).
- `always_on: false` skills appear only as a name + description in the system prompt; the agent calls `load_skill(name)` to pull in the full instructions when relevant, and `read_skill_file(skill, path)` to read a specific file under that skill's `reference/` directory. This keeps large reference material (API cheat-sheets, CLI references) out of every prompt by default.

Changes require a restart.
```

- [ ] **Step 6: Update this document's status**

This document's Part 2 status line already reads "Status: implemented." (this step is already satisfied — noted here only so the task-by-task record stays complete).

- [ ] **Step 7: Commit**

```bash
git add skills/ README.md docs/superpowers/storage-agent-spec.md
git commit -m "Add example skills, README docs, mark skills spec implemented"
```
