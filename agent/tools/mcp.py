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
