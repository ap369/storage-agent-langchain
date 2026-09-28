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
        f"{config_name}_{operation['name']}",
        description=operation["description"],
        args_schema=operation["params_schema"],
    )(call)
