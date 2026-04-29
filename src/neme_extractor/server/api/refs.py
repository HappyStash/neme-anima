"""REST routes for /api/projects/{slug}/refs."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

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
    return Project.load(Path(entry.folder))


@router.post("/{slug}/refs")
async def add_refs(request: Request, slug: str, body: AddRefsBody) -> dict:
    project = _load(request, slug)
    added: list[str] = []
    skipped: list[str] = []
    for p in body.paths:
        try:
            r = project.add_ref(Path(p).expanduser())
            added.append(r.path)
        except ValueError:
            skipped.append(str(Path(p).expanduser().resolve()))
    return {"added": added, "skipped": skipped}


@router.delete("/{slug}/refs", status_code=204)
async def remove_ref(request: Request, slug: str, body: RemoveRefBody) -> Response:
    project = _load(request, slug)
    project.remove_ref(body.path)
    return Response(status_code=204)
