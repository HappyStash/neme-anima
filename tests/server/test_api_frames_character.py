"""Tests for the character_slug surface on the frames API.

Covers:
  - GET /frames exposes ``character_slug`` on each row
  - GET /frames?character_slug= filters correctly, plus the ``__unsorted__``
    sentinel for orphan rows
  - POST /frames/{filename}/character moves a frame to a target character
  - POST /frames/{filename}/duplicate produces a true copy with its own row
  - POST /frames/upload?character_slug= routes the dropped frame
  - bulk-move + bulk-duplicate
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from neme_anima.server.api.frames import UNSORTED_FILTER_SENTINEL
from neme_anima.server.app import create_app
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG, Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    p = Project.create(tmp_path / "p", name="K-On!")
    p.add_character(name="Mio")
    return p


@pytest.fixture
def app(tmp_path: Path, project: Project):
    a = create_app(state_dir=tmp_path / "state")
    a.state.registry.register(project)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _seed_frame(project: Project, *, filename: str, character_slug: str) -> None:
    """Drop a real PNG + .txt onto disk and append a kept FrameRecord. The
    list_frames endpoint requires both the on-disk image and the metadata
    row to exist, so tests have to seed both."""
    png = project.kept_dir / f"{filename}.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(png)
    png.with_suffix(".txt").write_text("a, b\n", encoding="utf-8")
    MetadataLog(project.metadata_path).append(FrameRecord(
        filename=filename, kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 16, 16),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01", character_slug=character_slug,
    ))


async def test_list_includes_character_slug(client, project: Project):
    """Each frame row exposes ``character_slug`` so the UI can render the
    badge in 'All' view without a separate fetch."""
    _seed_frame(project, filename="ep01__a", character_slug=DEFAULT_CHARACTER_SLUG)
    _seed_frame(project, filename="ep01__b", character_slug="mio")
    resp = await client.get(f"/api/projects/{project.slug}/frames")
    assert resp.status_code == 200
    items = resp.json()["items"]
    by_name = {i["filename"]: i["character_slug"] for i in items}
    assert by_name == {"ep01__a": DEFAULT_CHARACTER_SLUG, "ep01__b": "mio"}


async def test_filter_by_character_slug(client, project: Project):
    """``?character_slug=mio`` returns Mio's frames only — total in view
    matches the filtered count, not the global count, so the UI's per-
    character total badge is honest."""
    _seed_frame(project, filename="ep01__a", character_slug=DEFAULT_CHARACTER_SLUG)
    _seed_frame(project, filename="ep01__b", character_slug="mio")
    _seed_frame(project, filename="ep01__c", character_slug="mio")
    resp = await client.get(
        f"/api/projects/{project.slug}/frames?character_slug=mio",
    )
    body = resp.json()
    assert body["total"] == 2
    assert {i["filename"] for i in body["items"]} == {"ep01__b", "ep01__c"}


async def test_filter_unsorted_sentinel_catches_orphan_slugs(
    client, project: Project,
):
    """A frame stamped with a slug that no longer exists in the project
    surfaces under the ``__unsorted__`` filter — that's the recovery path
    for renames/deletions that strand frames in metadata limbo."""
    _seed_frame(project, filename="ep01__orphan", character_slug="ghost")
    _seed_frame(project, filename="ep01__alive", character_slug="mio")
    resp = await client.get(
        f"/api/projects/{project.slug}/frames"
        f"?character_slug={UNSORTED_FILTER_SENTINEL}",
    )
    items = resp.json()["items"]
    assert {i["filename"] for i in items} == {"ep01__orphan"}


async def test_move_frame_appends_kept_record(client, project: Project):
    """POST /frames/{filename}/character flips the frame's slug via an
    append-only metadata row — last-write-wins is what makes this O(1)."""
    _seed_frame(project, filename="ep01__a", character_slug=DEFAULT_CHARACTER_SLUG)
    resp = await client.post(
        f"/api/projects/{project.slug}/frames/ep01__a/character",
        json={"character_slug": "mio"},
    )
    assert resp.status_code == 200
    assert resp.json()["character_slug"] == "mio"

    listing = (await client.get(
        f"/api/projects/{project.slug}/frames?character_slug=mio",
    )).json()
    assert {i["filename"] for i in listing["items"]} == {"ep01__a"}


async def test_move_frame_to_unknown_character_returns_404(
    client, project: Project,
):
    _seed_frame(project, filename="ep01__a", character_slug=DEFAULT_CHARACTER_SLUG)
    resp = await client.post(
        f"/api/projects/{project.slug}/frames/ep01__a/character",
        json={"character_slug": "nope"},
    )
    assert resp.status_code == 404


async def test_bulk_move(client, project: Project):
    """The bulk endpoint moves every named frame in one round-trip and
    reports any names whose metadata couldn't be found — the UI surfaces
    those as a partial-success warning."""
    _seed_frame(project, filename="ep01__a", character_slug=DEFAULT_CHARACTER_SLUG)
    _seed_frame(project, filename="ep01__b", character_slug=DEFAULT_CHARACTER_SLUG)
    resp = await client.post(
        f"/api/projects/{project.slug}/frames/bulk-move",
        json={"filenames": ["ep01__a", "ep01__b", "ep01__missing"],
              "character_slug": "mio"},
    )
    body = resp.json()
    assert body["moved"] == 2
    assert body["missing"] == ["ep01__missing"]


async def test_duplicate_frame_creates_independent_copy(
    client, project: Project,
):
    """``duplicate`` writes a new PNG + sidecar with a fresh filename so the
    target character's training set is independent. The original keeps its
    slug — duplicate is not move."""
    _seed_frame(project, filename="ep01__a", character_slug=DEFAULT_CHARACTER_SLUG)
    resp = await client.post(
        f"/api/projects/{project.slug}/frames/ep01__a/duplicate",
        json={"character_slug": "mio"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["character_slug"] == "mio"
    assert body["filename"].startswith("ep01__a_dup_")

    # Both characters now see the frame. Original keeps its slug.
    default_listing = (await client.get(
        f"/api/projects/{project.slug}/frames?character_slug={DEFAULT_CHARACTER_SLUG}",
    )).json()
    mio_listing = (await client.get(
        f"/api/projects/{project.slug}/frames?character_slug=mio",
    )).json()
    assert {i["filename"] for i in default_listing["items"]} == {"ep01__a"}
    mio_files = {i["filename"] for i in mio_listing["items"]}
    assert any(f.startswith("ep01__a_dup_") for f in mio_files)


async def test_bulk_duplicate_returns_new_filenames(client, project: Project):
    """The bulk endpoint returns the new filenames so the UI can refresh the
    grid and the user sees the copies appear immediately."""
    _seed_frame(project, filename="ep01__a", character_slug=DEFAULT_CHARACTER_SLUG)
    _seed_frame(project, filename="ep01__b", character_slug=DEFAULT_CHARACTER_SLUG)
    resp = await client.post(
        f"/api/projects/{project.slug}/frames/bulk-duplicate",
        json={"filenames": ["ep01__a", "ep01__b"], "character_slug": "mio"},
    )
    body = resp.json()
    assert len(body["duplicated"]) == 2
    assert body["missing"] == []


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


async def test_upload_routes_to_character_slug(client, project: Project):
    """The drag-drop endpoint accepts ``?character_slug=`` so dropping a
    file while filtered to Mio routes the new frame to Mio. Without the
    query param, the frame lands on the project's first character — the
    legacy mono-character behaviour."""
    resp = await client.post(
        f"/api/projects/{project.slug}/frames/upload?character_slug=mio",
        files=[("files", ("drop.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["added"]) == 1
    assert body["added"][0]["character_slug"] == "mio"


async def test_upload_unknown_character_returns_404(client, project: Project):
    resp = await client.post(
        f"/api/projects/{project.slug}/frames/upload?character_slug=nope",
        files=[("files", ("drop.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 404
