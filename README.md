# storage-agent

A self-hosted AI agent for a storage infrastructure team (PureStorage, NetApp, Dell ECS, general NAS/SAN/object storage). Built on FastAPI and LangChain: a tool-calling agent loop with sandboxed file access, allowlisted REST API calls, MCP server integration, a skills system for domain knowledge, and a chat webview — plus a REST endpoint for triggering the agent from external scripts.

See `docs/storage-agent-spec.md` for the full design and implementation plan.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible LLM endpoint (tested against Groq's API)

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` with your LLM credentials and a token for the trigger REST API:

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_BASE_URL` | yes | — | OpenAI-compatible base URL (e.g. `https://api.groq.com/openai/v1`) |
| `LLM_API_KEY` | yes | — | API key for the LLM endpoint |
| `LLM_MODEL` | yes | — | Model name (e.g. `openai/gpt-oss-120b`) |
| `API_TOKEN` | yes | — | Bearer token required by the trigger REST API (`/tasks`). The chat webview is unauthenticated. |
| `SANDBOX_ROOT` | no | `./data/sandbox` | Root directory the file tools are confined to |
| `DB_PATH` | no | `./data/storage_agent.db` | SQLite database path (conversations, tasks) |
| `SYSTEM_PROMPT_PATH` | no | `./config/system_prompt.md` | Base system prompt file |
| `API_ALLOWLIST_PATH` | no | `./config/api_allowlist.json` | Allowlisted REST API configs |
| `MCP_SERVERS_PATH` | no | `./config/mcp_servers.json` | MCP server configs |
| `SKILLS_PATH` | no | `./skills` | Directory of skill folders (domain knowledge injected into the agent's context) |
| `MAX_TOOL_TURNS` | no | `20` | Hard cap on tool-calling turns per message |
| `SUMMARIZE_TRIGGER_TOKENS` | no | `4000` | Token count that triggers conversation summarization |
| `SUMMARIZE_KEEP_MESSAGES` | no | `20` | Number of recent messages kept verbatim after summarization |
| `LOG_LEVEL` | no | `INFO` | Logging level |

## Running

```bash
make run
```

This starts the app at `http://localhost:8000` — the chat webview at `/`, the chat WebSocket at `/ws/chat`, and the REST API below. Changes to `.env` or any config file require a restart.

## API surface

The chat webview (`/`, `/ws/chat`, `/mcp/status`) has no authentication — anyone who can reach the port can use it. The trigger REST API (`/tasks`) requires `Authorization: Bearer <API_TOKEN>`.

- **`GET /`** — chat webview.
- **`WS /ws/chat`** — no handshake; just send `{"type": "message", "conversation_id": null_or_id, "content": "..."}` per turn. Streams `tool_call` / `tool_result` / `final` / `error` frames back.
- **`GET /mcp/status`** — `[{"name", "transport", "connected", "tools"}]`, one entry per configured MCP server.
- **`POST /tasks`** (requires `Authorization: Bearer <API_TOKEN>`) — body `{"input": "..."}`. Runs the agent in the background; returns `202 {"task_id", "status": "pending"}` immediately.
- **`GET /tasks/{task_id}`** (requires `Authorization: Bearer <API_TOKEN>`) — `{"task_id", "status", "input", "result", "error", "created_at", "started_at", "finished_at"}`.

## Configuration files

### System prompt (`config/system_prompt.md`)

Plain text, read verbatim as the base of the agent's system prompt. Skill content (see below) is appended to it at startup.

### REST allowlist (`config/api_allowlist.json`)

A list of REST API configs the agent is allowed to call. Each config becomes one tool per operation, named `<config.name>_<operation.name>`:

```json
[
  {
    "name": "purestorage",
    "base_url": "https://flasharray.example.com",
    "auth_type": "bearer",
    "auth_value": "${PURESTORAGE_API_TOKEN}",
    "operations": [
      {
        "name": "list_volumes",
        "method": "GET",
        "path": "/api/2.x/volumes",
        "description": "List volumes",
        "params_schema": {}
      }
    ]
  }
]
```

`auth_type` is one of `none`, `bearer`, `api_key_header` (needs `auth_header_name`), or `basic`. `auth_value` supports `${ENV_VAR}` interpolation. `path` may contain `{param}` placeholders, filled from the tool call's arguments (URL-encoded) and matched against `params_schema`; any remaining arguments go in the query string (`GET`/`DELETE`) or JSON body (other methods).

### MCP servers (`config/mcp_servers.json`)

A list of MCP server configs. Each server that fails to connect is skipped independently — the rest still load.

```json
[
  { "name": "example-stdio", "transport": "stdio", "command": "/path/to/server" },
  { "name": "example-http", "transport": "streamable_http", "url": "https://mcp.example.com" }
]
```

`transport` is `stdio`, `streamable_http`, or `sse` (legacy).

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
- `always_on: false` skills appear only as a name + description in the system prompt; the agent calls `load_skill(name)` to pull in the full instructions when relevant, and `read_skill_file(skill, path)` to read a specific file under that skill's `reference/` directory.

## Testing

```bash
uv run pytest
```
