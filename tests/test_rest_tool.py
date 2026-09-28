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
