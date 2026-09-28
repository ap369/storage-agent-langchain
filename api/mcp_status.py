from fastapi import APIRouter, Header, HTTPException, Request

from auth import parse_bearer_token, verify_token

router = APIRouter()


@router.get("/mcp/status")
async def mcp_status(request: Request, authorization: str | None = Header(default=None)) -> list[dict]:
    token = parse_bearer_token(authorization)
    if not verify_token(token, request.app.state.settings.API_TOKEN):
        raise HTTPException(status_code=401, detail="unauthorized")

    return request.app.state.mcp_status
