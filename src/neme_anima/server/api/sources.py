"""REST routes for /api/projects/{slug}/sources."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from neme_anima.server.paths import normalize_input_path
from neme_anima.storage.project import Project

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["sources"])


class AddSourcesBody(BaseModel):
    paths: list[str]


class ImportFolderBody(BaseModel):
    folder: str


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


@router.post("/{slug}/sources/import-folder")
async def import_folder(request: Request, slug: str, body: ImportFolderBody) -> dict:
    project = _load(request, slug)
    try:
        folder = normalize_input_path(body.folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {folder}")
    added, skipped = project.import_videos_from_folder(folder)
    return {
        "added": [s.path for s in added],
        "skipped": skipped,
        "source_root": project.source_root,
    }


@router.post("/{slug}/sources/reimport")
async def reimport(request: Request, slug: str) -> dict:
    project = _load(request, slug)
    if not project.source_root:
        raise HTTPException(
            status_code=400,
            detail="no source folder has been imported yet — pick a folder first",
        )
    folder = Path(project.source_root)
    if not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"source folder is missing: {folder}",
        )
    added, skipped = project.import_videos_from_folder(folder)
    return {
        "added": [s.path for s in added],
        "skipped": skipped,
        "source_root": project.source_root,
    }


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
        "project_slug": project.slug,
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
        "project_slug": project.slug,
        "source_idx": idx,
        "video_stem": video_stem,
    })
    return {"job_id": job_id}


@router.get("/{slug}/sources/{idx}/thumbnail")
async def get_thumbnail(request: Request, slug: str, idx: int) -> FileResponse:
    """Return a cached JPEG thumbnail for the source's video.

    The first request grabs one frame near 10 % of the video's duration via
    OpenCV, saves it under ``<project>/.thumbs/<stem>.jpg``, and serves it.
    Subsequent requests serve the cached file directly.
    """
    project = _load(request, slug)
    if idx < 0 or idx >= len(project.sources):
        raise HTTPException(status_code=404, detail="source index out of range")
    video_path = Path(project.sources[idx].path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"video file missing: {video_path}")

    thumbs_dir = project.root / ".thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    cache_path = thumbs_dir / f"{video_path.stem}.jpg"
    if not cache_path.exists():
        try:
            await asyncio.to_thread(_extract_thumbnail, video_path, cache_path)
        except Exception as e:
            # Surface the real error to the server log, then to the client.
            logger.exception("thumbnail extraction failed for %s", video_path)
            raise HTTPException(
                status_code=500,
                detail=f"thumbnail extraction failed: {type(e).__name__}: {e}",
            )
    return FileResponse(cache_path, media_type="image/jpeg")


def _extract_thumbnail(video_path: Path, dest: Path, *, max_side: int = 320) -> None:
    """Grab one frame near 10 % of the video and save it as a JPEG via ffmpeg.

    We shell out to ffmpeg/ffprobe rather than using cv2 or decord because:
      * ffmpeg handles every container/codec combination the user is likely to
        have and is already installed wherever decord works;
      * a clean subprocess avoids CPython/opencv install fragility (we hit a
        broken cv2 install in development);
      * a single fast seek (`-ss` before `-i`) is essentially instant even on
        large files.
    """
    import shutil as _shutil
    import subprocess

    if _shutil.which("ffmpeg") is None or _shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")

    duration = _probe_duration_seconds(video_path)
    seek = max(0.0, duration * 0.10) if duration > 10 else 0.0

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{seek:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale='min({max_side},iw)':-2",
        "-q:v", "4",
        str(dest),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    # Some files don't honour the fast seek (e.g. very short clips, broken
    # indexes); retry from frame 0 if the first attempt produced nothing.
    if res.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        cmd_retry = cmd.copy()
        cmd_retry[cmd_retry.index("-ss") + 1] = "0"
        res = subprocess.run(cmd_retry, capture_output=True, text=True, timeout=30)

    if res.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        stderr = (res.stderr or "").strip().splitlines()[-1:] or [""]
        raise RuntimeError(f"ffmpeg failed (rc={res.returncode}): {stderr[0][:300]}")


def _probe_duration_seconds(video_path: Path) -> float:
    """Return the video duration in seconds via ffprobe, or 0.0 if unknown."""
    import subprocess

    res = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=nw=1:nokey=1",
         str(video_path)],
        capture_output=True, text=True, timeout=15,
    )
    try:
        return float(res.stdout.strip())
    except (TypeError, ValueError):
        return 0.0
