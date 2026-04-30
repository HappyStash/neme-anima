"""REST routes for /api/projects/{slug}/refs."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from neme_extractor.server.paths import normalize_input_path
from neme_extractor.storage.project import Project

router = APIRouter(prefix="/api/projects", tags=["refs"])


class AddRefsBody(BaseModel):
    paths: list[str]


class RemoveRefBody(BaseModel):
    path: str


def _load(request: Request, slug: str) -> Project:
    entry = request.app.state.registry.get(slug)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown project: {slug}")
    try:
        return Project.load(Path(entry.folder))
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"project files missing for {slug!r} at {entry.folder}",
        )


@router.post("/{slug}/refs")
async def add_refs(request: Request, slug: str, body: AddRefsBody) -> dict:
    project = _load(request, slug)
    added: list[str] = []
    skipped: list[str] = []
    for p in body.paths:
        try:
            normalized = normalize_input_path(p)
        except ValueError:
            skipped.append(p)
            continue
        try:
            r = project.add_ref(normalized)
            added.append(r.path)
        except ValueError:
            skipped.append(str(normalized.resolve()))
    return {"added": added, "skipped": skipped}


@router.post("/{slug}/refs/upload")
async def upload_refs(
    request: Request, slug: str, files: list[UploadFile]
) -> dict:
    """Accept multipart-uploaded image bytes and store them in the project."""
    project = _load(request, slug)
    added: list[str] = []
    skipped: list[str] = []
    for f in files:
        try:
            data = await f.read()
            if not data:
                skipped.append(f.filename or "<empty>")
                continue
            r = project.add_ref_bytes(f.filename or "ref", data)
            added.append(r.path)
        finally:
            await f.close()
    return {"added": added, "skipped": skipped}


@router.get("/{slug}/refs/{name}/image")
async def get_ref_image(request: Request, slug: str, name: str) -> FileResponse:
    """Serve the bytes of a reference image stored under ``<project>/refs/``."""
    project = _load(request, slug)
    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="invalid ref name")
    refs_root = (project.root / "refs").resolve()
    target = (refs_root / name).resolve()
    # Defense in depth — ensure ``name`` didn't escape the refs/ folder.
    try:
        target.relative_to(refs_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid ref name")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="ref not found")
    return FileResponse(target)


@router.delete("/{slug}/refs", status_code=204)
async def remove_ref(request: Request, slug: str, body: RemoveRefBody) -> Response:
    project = _load(request, slug)
    project.remove_ref(body.path)
    return Response(status_code=204)
