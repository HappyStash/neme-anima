"""REST routes for /api/projects/{slug}/frames."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from neme_extractor.storage.metadata import MetadataLog
from neme_extractor.storage.project import Project

router = APIRouter(prefix="/api/projects", tags=["frames"])


class PutTagsBody(BaseModel):
    text: str


class BulkDeleteBody(BaseModel):
    filenames: list[str]


class BulkReplaceBody(BaseModel):
    filenames: list[str]
    pattern: str
    replacement: str
    case_insensitive: bool = False


def _load(request: Request, slug: str) -> Project:
    entry = request.app.state.registry.get(slug)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown project: {slug}")
    return Project.load(Path(entry.folder))


def _frame_paths(project: Project, filename: str) -> tuple[Path, Path]:
    return (project.kept_dir / f"{filename}.png",
            project.kept_dir / f"{filename}.txt")


@router.get("/{slug}/frames")
async def list_frames(
    request: Request, slug: str,
    source: str | None = Query(None),
    kept_only: bool = Query(True),
    offset: int = Query(0),
    limit: int = Query(500),
) -> dict:
    project = _load(request, slug)
    log = MetadataLog(project.metadata_path)
    items = []
    for rec in log.iter_records(video_stem=source):
        if kept_only and not rec.kept:
            continue
        items.append({
            "filename": rec.filename,
            "kept": rec.kept,
            "video_stem": rec.video_stem,
            "scene_idx": rec.scene_idx,
            "tracklet_id": rec.tracklet_id,
            "frame_idx": rec.frame_idx,
            "timestamp_seconds": rec.timestamp_seconds,
            "ccip_distance": rec.ccip_distance,
            "score": rec.score,
        })
    return {"count": len(items), "items": items[offset: offset + limit]}


@router.get("/{slug}/frames/{filename}/image")
async def get_frame_image(request: Request, slug: str, filename: str) -> FileResponse:
    project = _load(request, slug)
    png, _ = _frame_paths(project, filename)
    if not png.exists():
        raise HTTPException(status_code=404, detail="frame not found")
    return FileResponse(png, media_type="image/png")


@router.get("/{slug}/frames/{filename}/tags")
async def get_tags(request: Request, slug: str, filename: str) -> dict:
    project = _load(request, slug)
    _, txt = _frame_paths(project, filename)
    if not txt.exists():
        return {"text": ""}
    return {"text": txt.read_text(encoding="utf-8").rstrip("\n")}


@router.put("/{slug}/frames/{filename}/tags")
async def put_tags(request: Request, slug: str, filename: str, body: PutTagsBody) -> dict:
    project = _load(request, slug)
    png, txt = _frame_paths(project, filename)
    if not png.exists():
        raise HTTPException(status_code=404, detail="frame not found")
    txt.write_text(body.text + "\n", encoding="utf-8")
    return {"text": body.text}


@router.delete("/{slug}/frames/{filename}", status_code=204)
async def delete_frame(request: Request, slug: str, filename: str) -> Response:
    project = _load(request, slug)
    png, txt = _frame_paths(project, filename)
    if png.exists():
        png.unlink()
    if txt.exists():
        txt.unlink()
    return Response(status_code=204)


@router.post("/{slug}/frames/bulk-delete")
async def bulk_delete(request: Request, slug: str, body: BulkDeleteBody) -> dict:
    project = _load(request, slug)
    deleted = 0
    for filename in body.filenames:
        png, txt = _frame_paths(project, filename)
        if png.exists():
            png.unlink(); deleted += 1
        if txt.exists():
            txt.unlink()
    return {"deleted": deleted}


@router.post("/{slug}/frames/bulk-tags-replace")
async def bulk_tags_replace(
    request: Request, slug: str, body: BulkReplaceBody,
) -> dict:
    flags = re.IGNORECASE if body.case_insensitive else 0
    try:
        regex = re.compile(body.pattern, flags)
    except re.error as e:
        raise HTTPException(status_code=422, detail=f"invalid regex: {e}")
    project = _load(request, slug)
    changed = 0
    for filename in body.filenames:
        _, txt = _frame_paths(project, filename)
        if not txt.exists():
            continue
        before = txt.read_text(encoding="utf-8")
        after, n = regex.subn(body.replacement, before)
        if n > 0 and after != before:
            txt.write_text(after, encoding="utf-8")
            changed += n
    return {"changed": changed}
