"""REST routes for /api/projects/{slug}/frames."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import CROP_SUFFIX, Project
from neme_anima.tag import join_sidecar, split_sidecar

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


def _resolve_tag_target(
    project: Project, filename: str,
) -> tuple[Path, Path, str]:
    """Pick the (image, sidecar, effective_filename) the WD14/LLM taggers
    should operate on.

    Why this exists: the user may have cropped a frame to focus the trained-
    against region. The crop derivative lives at ``{filename}_crop.png`` and
    is what the dataset actually carries forward. Tagging the original wide
    shot would mis-describe pixels that the model never sees, so when a
    derivative is on disk for an original filename we transparently retarget
    the *image* to it. The sidecar always stays at the original filename —
    there is only ever one ``.txt`` per kept frame, and training pairs the
    crop's pixels with that original sidecar at staging time. The effective
    filename returned is also the original's so the UI updates the row the
    user actually sees (crop derivatives are hidden from the grid).

    If ``filename`` is itself a ``_crop`` derivative we strip the suffix and
    treat it as the original — there's no ``_crop_crop`` ghost and the .txt
    still belongs to the original.
    """
    base = filename[: -len(CROP_SUFFIX)] if filename.endswith(CROP_SUFFIX) else filename
    orig_png, txt = _frame_paths(project, base)
    deriv_png, _deriv_txt, _spec = _crop_paths(project, base)
    img = deriv_png if deriv_png.is_file() else orig_png
    return img, txt, base


@router.get("/{slug}/frames")
async def list_frames(
    request: Request, slug: str,
    source: str | None = Query(None),
    kept_only: bool = Query(True),
    query: str | None = Query(None),
    offset: int = Query(0),
    limit: int = Query(500),
) -> dict:
    project = _load(request, slug)
    log = MetadataLog(project.metadata_path)

    # The metadata log is append-only — a delete or rerun leaves orphan rows
    # behind. We dedupe by filename (keep the most recent record) and filter
    # to entries whose image still exists on disk so the UI never shows a
    # row with a broken thumbnail. Crop derivatives (`*_crop`) are hidden
    # from the grid: they're an internal training-target replacement for
    # their original, not a separate frame the user picked.
    by_filename: dict[str, FrameRecord] = {}
    for rec in log.iter_records(video_stem=source):
        if kept_only and not rec.kept:
            continue
        if rec.filename.endswith(CROP_SUFFIX):
            continue
        by_filename[rec.filename] = rec

    kept_dir = project.kept_dir
    rejected_dir = project.rejected_dir
    tokens = _parse_tag_query(query) if query else []
    items = []
    total_in_view = 0
    for rec in by_filename.values():
        on_disk = kept_dir / f"{rec.filename}.png" if rec.kept else rejected_dir / f"{rec.filename}.png"
        if not on_disk.is_file():
            continue
        # `total` is the count in the current source/kept_only view *before*
        # the tag query is applied, so the UI can render "X / Y" when a
        # search is active without firing a second listFrames call.
        total_in_view += 1
        if tokens and not _frame_matches_tag_query(
            kept_dir / f"{rec.filename}.txt", tokens,
        ):
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
    return {
        "count": len(items),
        "total": total_in_view,
        "items": items[offset: offset + limit],
    }


def _parse_tag_query(query: str) -> list[tuple[bool, str]]:
    """Split a search query into ``(is_negation, lowercased_substring)`` tokens.

    Whitespace separates tokens; a leading ``~`` flips a token to negation.
    Empty tokens are dropped. Substring (not exact) match against the
    danbooru tag line — typing ``red`` matches ``red eyes`` and ``red hair``,
    and ``~hat`` filters out anything whose tag line contains ``hat``.
    """
    out: list[tuple[bool, str]] = []
    for raw in query.split():
        if raw.startswith("~"):
            t = raw[1:].strip().lower()
            if t:
                out.append((True, t))
        else:
            t = raw.strip().lower()
            if t:
                out.append((False, t))
    return out


def _frame_matches_tag_query(
    txt_path: Path, tokens: list[tuple[bool, str]],
) -> bool:
    """Return True iff every positive token is present and every negation is absent."""
    if not tokens:
        return True
    haystack = ""
    if txt_path.is_file():
        try:
            danbooru, _ = split_sidecar(txt_path.read_text(encoding="utf-8"))
            haystack = danbooru.lower()
        except OSError:
            haystack = ""
    for is_neg, tok in tokens:
        present = tok in haystack
        if is_neg and present:
            return False
        if not is_neg and not present:
            return False
    return True


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
    """Replace the sidecar with the body's text.

    The request can be a single line (danbooru only) or two lines
    (danbooru + description). Either shape goes through ``split_sidecar``
    + ``join_sidecar`` so the danbooru line gets the standard dedupe and
    whitespace normalization — manual edits that introduce a duplicate tag
    don't make it to disk.

    When the body has no description line, the existing on-disk description
    is preserved. The frontend's tag-pill add/remove path can fire this
    PUT before its lazy hover-fetch of the sidecar resolves; in that race
    the body legitimately has no description to send, and we must not
    interpret silence as "delete the description". Sending an explicit
    two-line body still overwrites whatever's on disk — that's the
    description-modal path.
    """
    project = _load(request, slug)
    png, txt = _frame_paths(project, filename)
    if not png.exists():
        raise HTTPException(status_code=404, detail="frame not found")
    danbooru, description = split_sidecar(body.text)
    if not description and txt.exists():
        try:
            _, description = split_sidecar(txt.read_text(encoding="utf-8"))
        except OSError:
            description = ""
    final = join_sidecar(danbooru, description)
    txt.write_text(final, encoding="utf-8")
    # Return what was actually saved (post-dedupe), without the trailing
    # newline join_sidecar adds — the frontend uses this to refresh its
    # local tag cache, so any normalization the route applied stays
    # consistent on the client side too.
    return {"text": final.rstrip("\n")}


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


def _cleanup_crop_artifacts(project: Project, filename: str) -> None:
    """Remove crop-related siblings for a frame being deleted.

    Two cases:
      * Original deleted → drop its derivative png and the .crop.json
        sidecar so the next time a same-named frame is added it starts fresh.
        Also removes a legacy ``<name>_crop.txt`` if one is on disk from
        before the image-only derivative layout.
      * Derivative deleted → drop the parent's .crop.json sidecar so the
        next "open original" doesn't show a phantom rectangle for a crop
        whose result has been thrown away.
    """
    if filename.endswith(CROP_SUFFIX):
        parent = filename[: -len(CROP_SUFFIX)]
        spec = project.kept_dir / f"{parent}.crop.json"
        if spec.is_file():
            spec.unlink()
        return
    deriv_png, deriv_txt, spec = _crop_paths(project, filename)
    for p in (deriv_png, deriv_txt, spec):
        if p.is_file():
            p.unlink()


@router.delete("/{slug}/frames/{filename}", status_code=204)
async def delete_frame(request: Request, slug: str, filename: str) -> Response:
    project = _load(request, slug)
    png, txt = _frame_paths(project, filename)
    if png.exists():
        png.unlink()
    if txt.exists():
        txt.unlink()
    _cleanup_crop_artifacts(project, filename)
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
        _cleanup_crop_artifacts(project, filename)
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

    def _tag_one(filename: str) -> tuple[bool, str | None]:
        # Prefer the cropped derivative when one exists for an original.
        # Tags must describe the image we actually train against; tagging
        # a wide shot with a person who isn't in the crop is exactly the
        # bug this avoids.
        png, txt, eff = _resolve_tag_target(project, filename)
        if not png.is_file():
            return False, None
        with Image.open(png) as im:
            arr = np.array(im.convert("RGB"))
        new_danbooru = tagger.tag(arr).text
        old_text = txt.read_text(encoding="utf-8") if txt.exists() else ""
        _, description = split_sidecar(old_text)
        txt.write_text(join_sidecar(new_danbooru, description), encoding="utf-8")
        return True, eff

    retagged = 0
    effective_filenames: list[str | None] = []
    for filename in body.filenames:
        try:
            ok, eff = await asyncio.to_thread(_tag_one, filename)
        except Exception:
            ok, eff = False, None
        if ok:
            retagged += 1
        effective_filenames.append(eff)
    return {
        "retagged": retagged,
        "total": len(body.filenames),
        # Parallel to body.filenames; an entry differs from the input when
        # a `_crop` derivative was tagged in place of the original. The
        # frontend uses this to flip badges on the row that actually got
        # written rather than the row the user clicked.
        "effective_filenames": effective_filenames,
    }


@router.post("/{slug}/frames/bulk-retag-llm")
async def bulk_retag_llm(
    request: Request, slug: str, body: BulkRetagBody,
) -> dict:
    """Re-run the LLM description on selected frames; preserves the danbooru
    line. Returns 422 if the project has no LLM model configured — the
    frontend won't show the button in that state, but a stale tab might still
    fire it.
    """
    from neme_anima.llm import DEFAULT_PROMPT, LLMUnavailable, describe_image

    project = _load(request, slug)
    if not project.llm.model:
        raise HTTPException(
            status_code=422,
            detail="LLM tagging not configured: pick a model in Settings first",
        )
    endpoint = project.llm.endpoint
    model = project.llm.model
    prompt = project.llm.prompt or DEFAULT_PROMPT
    api_key = project.llm.api_key or None

    def _describe_one(filename: str) -> tuple[bool, str | None, str | None]:
        # Same retarget rule as the WD14 path: when a crop exists for an
        # original, describe the crop instead. Otherwise the LLM caption
        # would describe pixels the trainer never sees.
        png, txt, eff = _resolve_tag_target(project, filename)
        if not png.is_file():
            return False, None, None
        old_text = txt.read_text(encoding="utf-8") if txt.exists() else ""
        danbooru, _ = split_sidecar(old_text)
        try:
            description = describe_image(
                endpoint=endpoint, model=model, image_path=png,
                prompt=prompt, danbooru_tags=danbooru or None,
                api_key=api_key,
            )
        except LLMUnavailable as exc:
            return False, str(exc), eff
        txt.write_text(join_sidecar(danbooru, description), encoding="utf-8")
        return True, None, eff

    described = 0
    last_error: str | None = None
    effective_filenames: list[str | None] = []
    for filename in body.filenames:
        ok, err, eff = await asyncio.to_thread(_describe_one, filename)
        if ok:
            described += 1
        elif err:
            last_error = err
        effective_filenames.append(eff if ok else None)
    return {
        "described": described,
        "total": len(body.filenames),
        "error": last_error,
        # Parallel to body.filenames; entry is None when that filename
        # failed. When the crop took priority over the original, the entry
        # is the crop's filename so the frontend pops the right row.
        "effective_filenames": effective_filenames,
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


# Each original gets at most one crop derivative; re-cropping overwrites it.
# The fixed suffix (no numeric counter) is what makes the round-trip work —
# without it we'd accumulate _crop1/_crop2/... and have no way to find
# "the" derivative when re-opening the original. The suffix itself lives in
# storage/project.py because both the API and the trainer key off it.


def _crop_paths(project: Project, original_filename: str) -> tuple[Path, Path, Path]:
    """Return ``(derivative_png, derivative_txt, crop_spec_json)`` paths.

    The spec sidecar lives next to the *original* (not the derivative) so the
    "load the existing rect when reopening the original" lookup is a single
    file existence check. Naming it with a dot prefix to ``crop`` keeps it
    clearly separate from any tag .txt sidecar.
    """
    crop_base = f"{original_filename}{CROP_SUFFIX}"
    return (
        project.kept_dir / f"{crop_base}.png",
        project.kept_dir / f"{crop_base}.txt",
        project.kept_dir / f"{original_filename}.crop.json",
    )


@router.get("/{slug}/frames/{filename}/crop")
async def get_crop_rect(request: Request, slug: str, filename: str) -> dict:
    """Return the saved crop rectangle for ``filename`` if one exists.

    404 when no crop has been confirmed for this frame yet — the modal uses
    that as the "no overlay, start full-image" signal.
    """
    project = _load(request, slug)
    _, _, spec = _crop_paths(project, filename)
    if not spec.is_file():
        raise HTTPException(status_code=404, detail="no saved crop")
    try:
        data = json.loads(spec.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"corrupt crop spec: {exc}")
    # Only echo the four fields the client cares about; ignores anything
    # extra we may add later for forwards compat.
    return {
        "x": int(data.get("x", 0)),
        "y": int(data.get("y", 0)),
        "width": int(data.get("width", 0)),
        "height": int(data.get("height", 0)),
    }


@router.post("/{slug}/frames/{filename}/crop")
async def crop_frame_endpoint(
    request: Request, slug: str, filename: str, body: CropBody,
) -> dict:
    """Save (or overwrite) the cropped derivative for an original frame.

    Behavior: the original is never touched. The derivative lives at a fixed
    ``<filename>_crop`` filename — re-cropping overwrites it. A
    ``<filename>.crop.json`` sidecar records the rectangle so the modal can
    repaint the overlay when the original is reopened.
    """
    from PIL import Image

    project = _load(request, slug)
    src_png, _ = _frame_paths(project, filename)
    if not src_png.exists():
        raise HTTPException(status_code=404, detail="frame not found")

    original = _find_record(project, filename)
    if original is None:
        raise HTTPException(status_code=404, detail="frame metadata not found")

    out_png, out_txt, spec_path = _crop_paths(project, filename)
    crop_base = out_png.stem

    with Image.open(src_png) as im:
        im_w, im_h = im.size
        x = max(0, min(int(body.x), im_w))
        y = max(0, min(int(body.y), im_h))
        w = max(1, min(int(body.width), im_w - x))
        h = max(1, min(int(body.height), im_h - y))
        cropped = im.crop((x, y, x + w, y + h))
        cropped.save(out_png)  # overwrites prior derivative if present

    # The crop derivative is image-only; tags stay on the original sidecar.
    # Remove any legacy `<name>_crop.txt` left over from earlier versions
    # so the trainer-staging step doesn't see a phantom paired sidecar.
    if out_txt.exists():
        out_txt.unlink()

    spec_path.write_text(
        json.dumps({"x": x, "y": y, "width": w, "height": h}),
        encoding="utf-8",
    )

    new_rec = FrameRecord(
        filename=crop_base,
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
    # The metadata log is append-only; list_frames dedupes by filename and
    # keeps the latest record, so a re-crop "replaces" the visible bbox
    # without us having to rewrite the log.
    MetadataLog(project.metadata_path).append(new_rec)
    return _record_to_dict(new_rec)


def _get_or_make_tagger(request: Request):
    """Cache a single Tagger instance on app.state — WD14 model load is slow."""
    cached = getattr(request.app.state, "_tagger", None)
    if cached is not None:
        return cached
    from neme_anima.tag import Tagger
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
    # Last LLM-describe error (if any) is bubbled to the response so the
    # UI can surface a one-line warning instead of silently writing the
    # sidecar without line 2. We only keep the most recent message — the
    # first failure usually identifies the underlying issue (timeout,
    # auth, wrong model name) and the rest are echoes.
    llm_error: str | None = None

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
                from neme_anima.llm import (
                    DEFAULT_PROMPT, LLMUnavailable, describe_image,
                )

                def _do_describe() -> str:
                    return describe_image(
                        endpoint=project.llm.endpoint,
                        model=project.llm.model,
                        image_path=png_path,
                        prompt=project.llm.prompt or DEFAULT_PROMPT,
                        danbooru_tags=tag_text or None,
                        api_key=project.llm.api_key or None,
                    )

                try:
                    description = await asyncio.to_thread(_do_describe)
                except LLMUnavailable as exc:
                    # Endpoint reachable but unhappy (timeout, bad model id,
                    # auth refused). Log and remember so the user knows
                    # *why* their dropped image came back without line 2.
                    logger.warning(
                        "upload.describe_failed file=%s: %s",
                        f.filename or "<unknown>", exc,
                    )
                    llm_error = str(exc)
                    description = ""
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "upload.describe_crashed file=%s",
                        f.filename or "<unknown>",
                    )
                    llm_error = f"{type(exc).__name__}: {exc}"
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

    return {"added": added, "skipped": skipped, "llm_error": llm_error}
