"""REST routes for /api/projects/{slug}/frames."""

from __future__ import annotations

import asyncio
import io
import re
import secrets
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from neme_extractor.storage.metadata import FrameRecord, MetadataLog
from neme_extractor.storage.project import Project
from neme_extractor.tag import join_sidecar, split_sidecar

router = APIRouter(prefix="/api/projects", tags=["frames"])

# All drag-and-drop uploaded frames live under this synthetic video stem so
# they can be filtered/grouped just like extracted frames.
CUSTOM_VIDEO_STEM = "custom_uploads"

# Largest longest-side we keep on disk for uploaded images. The trainer
# handles bucketing/resizing itself, so cropping is wasteful, but a hard
# downscale ceiling keeps storage and tagging latency bounded.
MAX_UPLOAD_LONGEST_SIDE = 2048


class PutTagsBody(BaseModel):
    text: str


class PutDescriptionBody(BaseModel):
    text: str


class BulkDeleteBody(BaseModel):
    filenames: list[str]


class BulkReplaceBody(BaseModel):
    filenames: list[str]
    pattern: str
    replacement: str
    case_insensitive: bool = False


class BulkRetagBody(BaseModel):
    filenames: list[str]


class CropBody(BaseModel):
    x: int
    y: int
    width: int
    height: int


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

    # The metadata log is append-only — a delete or rerun leaves orphan rows
    # behind. We dedupe by filename (keep the most recent record) and filter
    # to entries whose image still exists on disk so the UI never shows a
    # row with a broken thumbnail.
    by_filename: dict[str, FrameRecord] = {}
    for rec in log.iter_records(video_stem=source):
        if kept_only and not rec.kept:
            continue
        by_filename[rec.filename] = rec

    kept_dir = project.kept_dir
    rejected_dir = project.rejected_dir
    items = []
    for rec in by_filename.values():
        on_disk = kept_dir / f"{rec.filename}.png" if rec.kept else rejected_dir / f"{rec.filename}.png"
        if not on_disk.is_file():
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
            "has_description": _has_description(kept_dir / f"{rec.filename}.txt"),
        })
    return {"count": len(items), "items": items[offset: offset + limit]}


def _has_description(txt_path: Path) -> bool:
    """True if the sidecar has a non-empty second row.

    Reads the file rather than just stat'ing — a stale .txt with whitespace
    on row 2 should still count as "no description" so the grid badge stays
    honest. The files are tiny (a few hundred bytes) so this is cheap.
    """
    if not txt_path.is_file():
        return False
    try:
        _, description = split_sidecar(txt_path.read_text(encoding="utf-8"))
    except OSError:
        return False
    return bool(description)


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


@router.get("/{slug}/frames/{filename}/description")
async def get_description(request: Request, slug: str, filename: str) -> dict:
    """Return only the LLM description (line 2 of the sidecar)."""
    project = _load(request, slug)
    _, txt = _frame_paths(project, filename)
    if not txt.exists():
        return {"text": ""}
    _, description = split_sidecar(txt.read_text(encoding="utf-8"))
    return {"text": description}


@router.put("/{slug}/frames/{filename}/description")
async def put_description(
    request: Request, slug: str, filename: str, body: PutDescriptionBody,
) -> dict:
    """Replace only the description line; the danbooru tag line is preserved."""
    project = _load(request, slug)
    png, txt = _frame_paths(project, filename)
    if not png.exists():
        raise HTTPException(status_code=404, detail="frame not found")
    danbooru = ""
    if txt.exists():
        danbooru, _ = split_sidecar(txt.read_text(encoding="utf-8"))
    txt.write_text(join_sidecar(danbooru, body.text), encoding="utf-8")
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
    """Run a regex over the *danbooru* line only — the LLM description on row
    two stays untouched so the user can rewrite tags without disturbing
    captions written by a separate model.

    Tip for adding tags: use a pattern like ``^`` with replacement ``new_tag, ``
    to prepend, or ``$`` with replacement ``, new_tag`` to append. To replace
    the whole tag set, use ``.*`` with the new comma-separated string.
    """
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
        danbooru, description = split_sidecar(before)
        new_danbooru, n = regex.subn(body.replacement, danbooru)
        if n > 0 and new_danbooru != danbooru:
            txt.write_text(join_sidecar(new_danbooru, description), encoding="utf-8")
            changed += n
    return {"changed": changed}


@router.post("/{slug}/frames/bulk-retag-danbooru")
async def bulk_retag_danbooru(
    request: Request, slug: str, body: BulkRetagBody,
) -> dict:
    """Re-run the WD14 tagger on selected frames; preserves the LLM line."""
    import numpy as np
    from PIL import Image

    project = _load(request, slug)
    tagger = _get_or_make_tagger(request)

    def _tag_one(filename: str) -> bool:
        png, txt = _frame_paths(project, filename)
        if not png.exists():
            return False
        with Image.open(png) as im:
            arr = np.array(im.convert("RGB"))
        new_danbooru = tagger.tag(arr).text
        old_text = txt.read_text(encoding="utf-8") if txt.exists() else ""
        _, description = split_sidecar(old_text)
        txt.write_text(join_sidecar(new_danbooru, description), encoding="utf-8")
        return True

    retagged = 0
    for filename in body.filenames:
        try:
            ok = await asyncio.to_thread(_tag_one, filename)
        except Exception:
            ok = False
        if ok:
            retagged += 1
    return {"retagged": retagged, "total": len(body.filenames)}


@router.post("/{slug}/frames/bulk-retag-llm")
async def bulk_retag_llm(
    request: Request, slug: str, body: BulkRetagBody,
) -> dict:
    """Re-run the LLM description on selected frames; preserves the danbooru
    line. Returns 422 if the project has no LLM model configured — the
    frontend won't show the button in that state, but a stale tab might still
    fire it.
    """
    from neme_extractor.llm import DEFAULT_PROMPT, LLMUnavailable, describe_image

    project = _load(request, slug)
    if not project.llm.model:
        raise HTTPException(
            status_code=422,
            detail="LLM tagging not configured: pick a model in Settings first",
        )
    endpoint = project.llm.endpoint
    model = project.llm.model
    prompt = project.llm.prompt or DEFAULT_PROMPT

    def _describe_one(filename: str) -> tuple[bool, str | None]:
        png, txt = _frame_paths(project, filename)
        if not png.exists():
            return False, None
        old_text = txt.read_text(encoding="utf-8") if txt.exists() else ""
        danbooru, _ = split_sidecar(old_text)
        try:
            description = describe_image(
                endpoint=endpoint, model=model, image_path=png,
                prompt=prompt, danbooru_tags=danbooru or None,
            )
        except LLMUnavailable as exc:
            return False, str(exc)
        txt.write_text(join_sidecar(danbooru, description), encoding="utf-8")
        return True, None

    described = 0
    last_error: str | None = None
    for filename in body.filenames:
        ok, err = await asyncio.to_thread(_describe_one, filename)
        if ok:
            described += 1
        elif err:
            last_error = err
    return {
        "described": described,
        "total": len(body.filenames),
        "error": last_error,
    }


def _record_to_dict(rec: FrameRecord) -> dict:
    return {
        "filename": rec.filename,
        "kept": rec.kept,
        "video_stem": rec.video_stem,
        "scene_idx": rec.scene_idx,
        "tracklet_id": rec.tracklet_id,
        "frame_idx": rec.frame_idx,
        "timestamp_seconds": rec.timestamp_seconds,
        "ccip_distance": rec.ccip_distance,
        "score": rec.score,
    }


def _find_record(project: Project, filename: str) -> FrameRecord | None:
    """Walk the metadata log and return the most recent record for ``filename``."""
    found: FrameRecord | None = None
    log = MetadataLog(project.metadata_path)
    for rec in log.iter_records():
        if rec.filename == filename:
            found = rec
    return found


def _next_crop_filename(project: Project, base: str) -> str:
    """Generate ``<base>_crop<n>`` not colliding with anything on disk."""
    n = 1
    while (project.kept_dir / f"{base}_crop{n}.png").exists():
        n += 1
        if n > 9999:
            raise RuntimeError(f"too many crops of {base!r}")
    return f"{base}_crop{n}"


@router.post("/{slug}/frames/{filename}/crop")
async def crop_frame_endpoint(
    request: Request, slug: str, filename: str, body: CropBody,
) -> dict:
    """Save a cropped derivative as a NEW frame; the original is preserved."""
    from PIL import Image

    project = _load(request, slug)
    src_png, src_txt = _frame_paths(project, filename)
    if not src_png.exists():
        raise HTTPException(status_code=404, detail="frame not found")

    original = _find_record(project, filename)
    if original is None:
        raise HTTPException(status_code=404, detail="frame metadata not found")

    with Image.open(src_png) as im:
        im_w, im_h = im.size
        x = max(0, min(int(body.x), im_w))
        y = max(0, min(int(body.y), im_h))
        w = max(1, min(int(body.width), im_w - x))
        h = max(1, min(int(body.height), im_h - y))
        cropped = im.crop((x, y, x + w, y + h))
        new_base = _next_crop_filename(project, filename)
        out_png = project.kept_dir / f"{new_base}.png"
        cropped.save(out_png)

    # Carry the original's tags forward; the crop usually keeps roughly the
    # same subject and the user can re-edit / re-tag on demand.
    out_txt = project.kept_dir / f"{new_base}.txt"
    if src_txt.exists():
        out_txt.write_bytes(src_txt.read_bytes())
    else:
        out_txt.write_text("\n", encoding="utf-8")

    new_rec = FrameRecord(
        filename=new_base,
        kept=True,
        scene_idx=original.scene_idx,
        tracklet_id=original.tracklet_id,
        frame_idx=original.frame_idx,
        timestamp_seconds=original.timestamp_seconds,
        bbox=(x, y, x + w, y + h),
        ccip_distance=original.ccip_distance,
        sharpness=original.sharpness,
        visibility=original.visibility,
        aspect=(w / h) if h else 1.0,
        score=original.score,
        video_stem=original.video_stem,
    )
    MetadataLog(project.metadata_path).append(new_rec)
    return _record_to_dict(new_rec)


def _get_or_make_tagger(request: Request):
    """Cache a single Tagger instance on app.state — WD14 model load is slow."""
    cached = getattr(request.app.state, "_tagger", None)
    if cached is not None:
        return cached
    from neme_extractor.tag import Tagger
    cached = Tagger()
    request.app.state._tagger = cached
    return cached


def _process_uploaded_image(
    project: Project, data: bytes, filename_hint: str,
) -> tuple[Path, int, int, str]:
    """Decode, downscale-if-huge, save PNG with a unique name. No tagging here.

    Returns (png_path, width, height, base_filename).
    """
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        if max(im.width, im.height) > MAX_UPLOAD_LONGEST_SIDE:
            scale = MAX_UPLOAD_LONGEST_SIDE / max(im.width, im.height)
            new_size = (max(1, int(im.width * scale)),
                        max(1, int(im.height * scale)))
            im = im.resize(new_size, Image.LANCZOS)
        # 8-char random suffix is plenty to avoid collisions with concurrent drops.
        token = secrets.token_hex(4)
        base = f"{CUSTOM_VIDEO_STEM}__{token}"
        png_path = project.kept_dir / f"{base}.png"
        # Vanishingly unlikely, but keep the loop tight just in case.
        while png_path.exists():
            token = secrets.token_hex(4)
            base = f"{CUSTOM_VIDEO_STEM}__{token}"
            png_path = project.kept_dir / f"{base}.png"
        im.save(png_path)
        return png_path, im.width, im.height, base


@router.post("/{slug}/frames/upload")
async def upload_frames(
    request: Request, slug: str, files: list[UploadFile],
) -> dict:
    """Accept dropped image files, store + auto-tag them as custom-source frames."""
    import numpy as np
    from PIL import Image

    project = _load(request, slug)
    project.kept_dir.mkdir(parents=True, exist_ok=True)
    log = MetadataLog(project.metadata_path)

    added: list[dict] = []
    skipped: list[str] = []

    tagger = _get_or_make_tagger(request)

    for f in files:
        try:
            data = await f.read()
            if not data:
                skipped.append(f.filename or "<empty>")
                continue
            try:
                png_path, w, h, base = await asyncio.to_thread(
                    _process_uploaded_image, project, data, f.filename or "drop",
                )
            except Exception:
                skipped.append(f.filename or "<unknown>")
                continue

            def _do_tag(p: Path) -> str:
                with Image.open(p) as pim:
                    arr = np.array(pim.convert("RGB"))
                return tagger.tag(arr).text

            try:
                tag_text = await asyncio.to_thread(_do_tag, png_path)
            except Exception:
                tag_text = ""

            description = ""
            if project.llm.enabled and project.llm.model:
                from neme_extractor.llm import (
                    DEFAULT_PROMPT, LLMUnavailable, describe_image,
                )

                def _do_describe() -> str:
                    return describe_image(
                        endpoint=project.llm.endpoint,
                        model=project.llm.model,
                        image_path=png_path,
                        prompt=project.llm.prompt or DEFAULT_PROMPT,
                        danbooru_tags=tag_text or None,
                    )

                try:
                    description = await asyncio.to_thread(_do_describe)
                except (LLMUnavailable, Exception):
                    description = ""
            png_path.with_suffix(".txt").write_text(
                join_sidecar(tag_text, description), encoding="utf-8",
            )

            rec = FrameRecord(
                filename=base,
                kept=True,
                scene_idx=0,
                tracklet_id=0,
                frame_idx=0,
                timestamp_seconds=0.0,
                bbox=(0, 0, w, h),
                ccip_distance=0.0,
                sharpness=0.0,
                visibility=0.0,
                aspect=(w / h) if h else 1.0,
                score=0.0,
                video_stem=CUSTOM_VIDEO_STEM,
            )
            log.append(rec)
            added.append(_record_to_dict(rec))
        finally:
            await f.close()

    return {"added": added, "skipped": skipped}
