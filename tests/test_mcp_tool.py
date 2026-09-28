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
        # MCPAdapter-wrapped tools return content blocks, not plain strings.
        assert result[0]["text"] == "pong"

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
