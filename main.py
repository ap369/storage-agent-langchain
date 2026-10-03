import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.core import build_agent
from agent.prompt import build_system_prompt
from agent.skills import load_skills
from agent.tools.files import build_file_tools
from agent.tools.mcp import build_mcp_tools, load_mcp_server_configs
from agent.tools.rest import build_rest_tools, load_api_configs
from agent.tools.skills import build_skill_tools
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

    skills = load_skills(Path(settings.SKILLS_PATH))
    always_on_skills = [s for s in skills if s.always_on]
    on_demand_skills = [s for s in skills if not s.always_on]

    base_prompt = Path(settings.SYSTEM_PROMPT_PATH).read_text()
    system_prompt = build_system_prompt(base_prompt, always_on_skills, on_demand_skills)

    resources_stack = AsyncExitStack()
    app.state.resources_stack = resources_stack

    checkpointer = await resources_stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string(settings.DB_PATH + ".checkpoints")
    )
    await checkpointer.setup()

    file_tools = build_file_tools(sandbox_root)
    skill_tools = build_skill_tools(on_demand_skills)

    api_configs = load_api_configs(Path(settings.API_ALLOWLIST_PATH))
    rest_tools = await build_rest_tools(api_configs, resources_stack)

    mcp_servers = load_mcp_server_configs(Path(settings.MCP_SERVERS_PATH))
    mcp_tools, mcp_status = await build_mcp_tools(mcp_servers, resources_stack)
    app.state.mcp_status = mcp_status

    all_tools = file_tools + rest_tools + mcp_tools + skill_tools
    app.state.agent = build_agent(
        model,
        all_tools,
        system_prompt,
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
app.mount("/web", StaticFiles(directory="web"), name="web")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("web/index.html")
