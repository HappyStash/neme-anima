"""REST routes for /api/projects/{slug}/characters.

Adds the multi-character CRUD surface; the existing refs/sources/training
endpoints remain mono-character (default character) for backwards
compatibility while the UI is still single-character. New character-aware
clients can pass ``?character_slug=`` to those endpoints — see
``refs.py`` and ``sources.py``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from neme_anima.storage.project import Project

router = APIRouter(prefix="/api/projects", tags=["characters"])


class CreateCharacterBody(BaseModel):
    name: str
    slug: str | None = None


class PatchCharacterBody(BaseModel):
    name: str | None = None
    trigger_token: str | None = None


def _load(request: Request, slug: str) -> Project:
    entry = request.app.state.registry.get(slug)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown project: {slug}")
    try:
        return Project.load(Path(entry.folder))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"project files missing for {slug!r} at {entry.folder}",
        ) from e


def _character_view(project: Project, character_slug: str) -> dict:
    c = project.character_by_slug(character_slug)
    if c is None:
        raise HTTPException(status_code=404, detail=f"unknown character: {character_slug}")
    return {
        "slug": c.slug,
        "name": c.name,
        "trigger_token": c.trigger_token,
        "refs": [asdict(r) for r in c.refs],
        "ref_count": len(c.refs),
    }


@router.get("/{slug}/characters")
async def list_characters(request: Request, slug: str) -> list[dict]:
    project = _load(request, slug)
    return [_character_view(project, c.slug) for c in project.characters]


@router.post("/{slug}/characters", status_code=201)
async def create_character(
    request: Request, slug: str, body: CreateCharacterBody,
) -> dict:
    project = _load(request, slug)
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    c = project.add_character(name=name, slug=body.slug)
    return _character_view(project, c.slug)


@router.patch("/{slug}/characters/{character_slug}")
async def update_character(
    request: Request, slug: str, character_slug: str, body: PatchCharacterBody,
) -> dict:
    project = _load(request, slug)
    c = project.character_by_slug(character_slug)
    if c is None:
        raise HTTPException(status_code=404, detail=f"unknown character: {character_slug}")
    if body.name is not None:
        new_name = body.name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="name must not be empty")
        c.name = new_name
    if body.trigger_token is not None:
        c.trigger_token = body.trigger_token.strip()
    project.save()
    return _character_view(project, c.slug)


@router.delete("/{slug}/characters/{character_slug}", status_code=204)
async def delete_character(
    request: Request, slug: str, character_slug: str,
) -> Response:
    project = _load(request, slug)
    if project.character_by_slug(character_slug) is None:
        raise HTTPException(status_code=404, detail=f"unknown character: {character_slug}")
    try:
        project.remove_character(character_slug)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return Response(status_code=204)
