"""REST routes for /api/projects."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from neme_extractor.storage.project import Project

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectBody(BaseModel):
    name: str
    folder: str


class RegisterBody(BaseModel):
    folder: str


class PatchProjectBody(BaseModel):
    name: str | None = None
    thresholds_overrides: dict | None = None


class DeleteProjectBody(BaseModel):
    delete_files: bool = False


def _project_view(project: Project) -> dict:
    return {
        "slug": project.slug,
        "name": project.name,
        "folder": str(project.root.resolve()),
        "created_at": project.created_at.isoformat(),
        "sources": [asdict(s) for s in project.sources],
        "refs": [asdict(r) for r in project.refs],
        "thresholds_overrides": project.thresholds_overrides,
        "source_root": project.source_root,
    }


@router.get("")
async def list_projects(request: Request) -> list[dict]:
    rows = request.app.state.registry.list()
    out: list[dict] = []
    for r in rows:
        try:
            project = Project.load(Path(r.folder))
            out.append({
                "slug": r.slug,
                "name": r.name,
                "folder": r.folder,
                "missing": False,
                "source_count": len(project.sources),
                "ref_count": len(project.refs),
                "last_opened_at": r.last_opened_at,
            })
        except FileNotFoundError:
            out.append({
                "slug": r.slug, "name": r.name, "folder": r.folder,
                "missing": True, "source_count": 0, "ref_count": 0,
                "last_opened_at": r.last_opened_at,
            })
    return out


@router.post("", status_code=201)
async def create_project(request: Request, body: CreateProjectBody) -> dict:
    target = Path(body.folder).expanduser()
    try:
        project = Project.create(target, name=body.name)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    request.app.state.registry.register(project)
    return _project_view(project)


@router.post("/register")
async def register_existing(request: Request, body: RegisterBody) -> dict:
    folder = Path(body.folder).expanduser()
    try:
        project = Project.load(folder)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    request.app.state.registry.register(project)
    return _project_view(project)


def _load_or_404(request: Request, slug: str) -> Project:
    entry = request.app.state.registry.get(slug)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown project: {slug}")
    try:
        return Project.load(Path(entry.folder))
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"project files missing for {slug!r} at {entry.folder} — "
                   "folder was moved or deleted; remove the registry entry or restore the files",
        )


@router.get("/{slug}")
async def get_project(request: Request, slug: str) -> dict:
    project = _load_or_404(request, slug)
    request.app.state.registry.touch(slug)
    return _project_view(project)


@router.patch("/{slug}")
async def patch_project(request: Request, slug: str, body: PatchProjectBody) -> dict:
    project = _load_or_404(request, slug)
    if body.name is not None:
        project.name = body.name
    if body.thresholds_overrides is not None:
        project.thresholds_overrides = body.thresholds_overrides
    project.save()
    request.app.state.registry.register(project)  # refresh name
    return _project_view(project)


@router.delete("/{slug}", status_code=204)
async def delete_project(
    request: Request, slug: str, body: DeleteProjectBody = DeleteProjectBody(),
) -> Response:
    entry = request.app.state.registry.get(slug)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown project: {slug}")
    request.app.state.registry.unregister(slug)
    if body.delete_files:
        shutil.rmtree(entry.folder, ignore_errors=True)
    return Response(status_code=204)
