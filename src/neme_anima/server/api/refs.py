"""REST routes for /api/projects/{slug}/refs."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from neme_anima.server.paths import normalize_input_path
from neme_anima.storage.project import Project

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


def _resolve_character_slug(project: Project, raw: str | None) -> str | None:
    """Validate ``raw`` against the project's characters.

    ``None`` (the default when the query string is omitted) means "use the
    default character" and is passed through unchanged. An unknown slug is
    a 404 — callers should know what character they're targeting; silent
    fallback to the default would mask UI bugs.
    """
    if raw is None or raw == "":
        return None
    if project.character_by_slug(raw) is None:
        raise HTTPException(status_code=404, detail=f"unknown character: {raw}")
    return raw


@router.post("/{slug}/refs")
async def add_refs(
    request: Request,
    slug: str,
    body: AddRefsBody,
    character_slug: str | None = None,
) -> dict:
    project = _load(request, slug)
    cslug = _resolve_character_slug(project, character_slug)
    added: list[str] = []
    skipped: list[str] = []
    for p in body.paths:
        try:
            normalized = normalize_input_path(p)
        except ValueError:
            skipped.append(p)
            continue
        try:
            r = project.add_ref(normalized, character_slug=cslug)
            added.append(r.path)
        except ValueError:
            skipped.append(str(normalized.resolve()))
    return {"added": added, "skipped": skipped}


@router.post("/{slug}/refs/upload")
async def upload_refs(
    request: Request,
    slug: str,
    files: list[UploadFile],
    character_slug: str | None = None,
) -> dict:
    """Accept multipart-uploaded image bytes and store them in the project."""
    project = _load(request, slug)
    cslug = _resolve_character_slug(project, character_slug)
    added: list[str] = []
    skipped: list[str] = []
    for f in files:
        try:
            data = await f.read()
            if not data:
                skipped.append(f.filename or "<empty>")
                continue
            r = project.add_ref_bytes(
                f.filename or "ref", data, character_slug=cslug,
            )
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
