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
