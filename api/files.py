import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.tools.files import SandboxViolation, resolve_in_sandbox

router = APIRouter()


class CreateDirRequest(BaseModel):
    path: str


def _resolve(request: Request, path: str) -> Path:
    try:
        return resolve_in_sandbox(request.app.state.sandbox_root, path)
    except SandboxViolation as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/files")
def list_files(request: Request, path: str = ".") -> list[dict]:
    target = _resolve(request, path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="not a directory")

    entries = [
        {
            "name": p.name,
            "type": "dir" if p.is_dir() else "file",
            "size": None if p.is_dir() else p.stat().st_size,
        }
        for p in target.iterdir()
    ]
    return sorted(entries, key=lambda e: (e["type"] != "dir", e["name"].lower()))


@router.post("/files/dir", status_code=201)
def create_dir(body: CreateDirRequest, request: Request) -> dict:
    target = _resolve(request, body.path)
    if target.exists():
        raise HTTPException(status_code=409, detail="already exists")
    target.mkdir(parents=True)
    return {"path": body.path}


@router.post("/files/upload", status_code=201)
def upload_files(request: Request, files: list[UploadFile], path: str = ".") -> dict:
    folder = _resolve(request, path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail="not a directory")

    # Keep only the final name component, so an upload can't pick its own
    # destination folder (e.g. a filename like "../../x").
    targets = [(upload, _resolve(request, str(Path(path) / Path(upload.filename or "").name))) for upload in files]
    for upload, target in targets:
        if target == folder:
            raise HTTPException(status_code=400, detail="missing filename")
        if target.exists():
            raise HTTPException(status_code=409, detail=f"already exists: {target.name}")

    for upload, target in targets:
        with target.open("xb") as out:
            shutil.copyfileobj(upload.file, out)
    return {"uploaded": [target.name for _, target in targets]}


@router.get("/files/download")
def download_file(request: Request, path: str) -> FileResponse:
    target = _resolve(request, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(target, filename=target.name)
