"""REST routes for /api/projects/{slug}/sources."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from neme_extractor.server.paths import normalize_input_path
from neme_extractor.storage.project import Project

router = APIRouter(prefix="/api/projects", tags=["sources"])


class AddSourcesBody(BaseModel):
    paths: list[str]


class PatchSourceBody(BaseModel):
    excluded_refs: list[str] | None = None


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


@router.post("/{slug}/sources")
async def add_sources(request: Request, slug: str, body: AddSourcesBody) -> dict:
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
            s = project.add_source(normalized)
            added.append(s.path)
        except ValueError:
            skipped.append(str(normalized.resolve()))
    return {"added": added, "skipped": skipped}


@router.delete("/{slug}/sources/{idx}", status_code=204)
async def remove_source(request: Request, slug: str, idx: int) -> Response:
    project = _load(request, slug)
    if idx < 0 or idx >= len(project.sources):
        raise HTTPException(status_code=404, detail="source index out of range")
    project.remove_source(idx)
    return Response(status_code=204)


@router.patch("/{slug}/sources/{idx}")
async def patch_source(
    request: Request, slug: str, idx: int, body: PatchSourceBody
) -> dict:
    project = _load(request, slug)
    if idx < 0 or idx >= len(project.sources):
        raise HTTPException(status_code=404, detail="source index out of range")
    if body.excluded_refs is not None:
        project.set_excluded_refs(idx, body.excluded_refs)
    return {"excluded_refs": project.sources[idx].excluded_refs}


@router.post("/{slug}/sources/{idx}/extract", status_code=202)
async def extract(request: Request, slug: str, idx: int) -> dict:
    project = _load(request, slug)
    if idx < 0 or idx >= len(project.sources):
        raise HTTPException(status_code=404, detail="source index out of range")
    job_id = await request.app.state.queue.submit({
        "kind": "extract",
        "project_folder": str(project.root.resolve()),
        "source_idx": idx,
    })
    return {"job_id": job_id}


@router.post("/{slug}/sources/{idx}/rerun", status_code=202)
async def rerun(request: Request, slug: str, idx: int) -> dict:
    project = _load(request, slug)
    if idx < 0 or idx >= len(project.sources):
        raise HTTPException(status_code=404, detail="source index out of range")
    video_stem = project.video_stem(idx)
    job_id = await request.app.state.queue.submit({
        "kind": "rerun",
        "project_folder": str(project.root.resolve()),
        "video_stem": video_stem,
    })
    return {"job_id": job_id}
