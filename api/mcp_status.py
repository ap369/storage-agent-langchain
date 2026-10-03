from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/mcp/status")
async def mcp_status(request: Request) -> list[dict]:
    return request.app.state.mcp_status
