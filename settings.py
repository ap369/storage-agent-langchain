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
