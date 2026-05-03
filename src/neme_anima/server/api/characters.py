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

from neme_anima.balancing import compute_character_balancing
from neme_anima.core_tags import DEFAULT_BLACKLIST, compute_core_tags
from neme_anima.storage.project import Project

router = APIRouter(prefix="/api/projects", tags=["characters"])


class CreateCharacterBody(BaseModel):
    name: str
    slug: str | None = None


class PatchCharacterBody(BaseModel):
    name: str | None = None
    trigger_token: str | None = None
    core_tags: list[str] | None = None
    core_tags_freq_threshold: float | None = None
    core_tags_enabled: bool | None = None
    multiply: float | None = None


class CoreTagsComputeBody(BaseModel):
    """Optional override knobs for the compute-preview pass.

    Both fields are optional so the UI can call this with an empty body to
    run with the character's persisted threshold + the default blacklist.
    """
    threshold: float | None = None
    blacklist: list[str] | None = None


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
        # Surface the core-tag + balancing config so the UI can render the
        # Settings tab forms without a separate fetch — same trade-off as
        # the top-level project view exposing characters.
        "core_tags": list(c.core_tags),
        "core_tags_freq_threshold": c.core_tags_freq_threshold,
        "core_tags_enabled": c.core_tags_enabled,
        "multiply": c.multiply,
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
    if body.core_tags is not None:
        # The UI sends the user-confirmed list — strip whitespace + drop
        # blanks defensively so an editor inserting trailing commas doesn't
        # produce empty-string "tags" that would silently survive prune_tags.
        c.core_tags = [t.strip() for t in body.core_tags if t.strip()]
    if body.core_tags_freq_threshold is not None:
        # Clamp to (0, 1] so a slider misfire can't store a sentinel that
        # would zero-divide later or ban every tag.
        threshold = max(0.01, min(1.0, float(body.core_tags_freq_threshold)))
        c.core_tags_freq_threshold = threshold
    if body.core_tags_enabled is not None:
        c.core_tags_enabled = bool(body.core_tags_enabled)
    if body.multiply is not None:
        # 0.0 is the "auto" sentinel; negatives don't make sense and would
        # invert the trainer's exposure math.
        c.multiply = max(0.0, float(body.multiply))
    project.save()
    return _character_view(project, c.slug)


@router.post("/{slug}/characters/{character_slug}/core-tags/compute")
async def compute_character_core_tags(
    request: Request, slug: str, character_slug: str, body: CoreTagsComputeBody,
) -> dict:
    """Run the core-tag analysis for a character and return the suggested list.

    This is a *preview*: nothing is persisted. The UI surfaces the table,
    the user reviews, and a follow-up PATCH with ``core_tags`` saves the
    confirmed set. Threshold defaults to the character's persisted value
    so flipping the toggle alone is enough to refresh.
    """
    project = _load(request, slug)
    c = project.character_by_slug(character_slug)
    if c is None:
        raise HTTPException(status_code=404, detail=f"unknown character: {character_slug}")
    threshold = (
        float(body.threshold) if body.threshold is not None
        else c.core_tags_freq_threshold
    )
    blacklist = (
        tuple(body.blacklist) if body.blacklist is not None
        else DEFAULT_BLACKLIST
    )
    report = compute_core_tags(
        project=project, character_slug=character_slug,
        threshold=threshold, blacklist=blacklist,
    )
    return {
        "character_slug": report.character_slug,
        "corpus_size": report.corpus_size,
        "threshold": report.threshold,
        # Tag table is sorted desc by frequency; UI renders it as-is.
        "tags": [{"tag": t, "freq": f} for t, f in report.tags],
        "blacklisted": list(report.blacklisted),
    }


@router.get("/{slug}/balancing/preview")
async def balancing_preview(request: Request, slug: str) -> dict:
    """Return the per-character balancing table for the UI's preview view.

    The Training tab renders this so the user can sanity-check the
    multipliers before kicking off a multi-character run. The format
    mirrors :class:`~neme_anima.balancing.CharacterBalanceRow` plus the
    project totals so the UI can render percentages.
    """
    project = _load(request, slug)
    rows = compute_character_balancing(project=project)
    total_frames = sum(r.frame_count for r in rows)
    return {
        "total_frames": total_frames,
        "rows": [
            {
                "character_slug": r.character_slug,
                "name": r.name,
                "frame_count": r.frame_count,
                "auto_multiply": r.auto_multiply,
                "manual_multiply": r.manual_multiply,
                "effective_multiply": r.effective_multiply,
            }
            for r in rows
        ],
    }


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


class CopyToBody(BaseModel):
    destination_slug: str
    dry_run: bool = False


@router.post("/{slug}/characters/{character_slug}/copy-to")
async def copy_to_project(
    request: Request, slug: str, character_slug: str, body: CopyToBody,
) -> dict:
    """Copy this character (and all artifacts related to it) into another
    registered project. See ``character_copy.copy_character_to_project``."""
    from neme_anima.storage.character_copy import copy_character_to_project

    src = _load(request, slug)
    dst_entry = request.app.state.registry.get(body.destination_slug)
    if dst_entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown destination project: {body.destination_slug}",
        )
    try:
        dst = Project.load(Path(dst_entry.folder))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=(
                f"destination project files missing for "
                f"{body.destination_slug!r} at {dst_entry.folder}"
            ),
        ) from e
    try:
        report = copy_character_to_project(
            src=src,
            src_character_slug=character_slug,
            dst=dst,
            dry_run=body.dry_run,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return report.to_dict()
