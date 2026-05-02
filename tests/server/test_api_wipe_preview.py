"""Tests for GET /sources/{idx}/wipe-preview.

The endpoint is what the Sources tab calls before firing Extract /
Re-process so it can show a confirmation modal explaining exactly
which frames the run would replace and which would survive. Critical
that the breakdown matches the actual wipe behaviour — otherwise the
modal lies to the user.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from neme_anima.server.app import create_app
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG, Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    p = Project.create(tmp_path / "p", name="show")
    # Two characters; only the default has a ref so it's the active one.
    # Mio is the inactive character whose frames should be preserved.
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


def _seed(project: Project, *, filename: str, character_slug: str, kept: bool = True) -> None:
    """Write a real PNG and a metadata row matching the filename so the
    preview endpoint's owner-attribution path exercises end-to-end."""
    png = (project.kept_dir if kept else project.rejected_dir) / f"{filename}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (5, 5, 5)).save(png)
    if kept:
        png.with_suffix(".txt").write_text("a, b\n", encoding="utf-8")
    MetadataLog(project.metadata_path).append(FrameRecord(
        filename=filename, kept=kept,
        scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01", character_slug=character_slug,
    ))


async def test_no_prior_outputs_returns_zero_total(
    client, project: Project, tmp_path: Path,
):
    """A first-time extract — nothing on disk for the stem — must report
    total=0 so the UI skips the modal entirely. Empty maps, empty lists
    everywhere; the only confusing thing would be if active_slugs were
    empty, which means the user has no refs at all."""
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    project.add_source(vid)
    img = tmp_path / "ref.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    project.add_ref(img)

    body = (await client.get(f"/api/projects/{project.slug}/sources/0/wipe-preview")).json()
    assert body["to_wipe"]["total"] == 0
    assert body["to_preserve"]["total"] == 0
    assert body["active_slugs"] == [DEFAULT_CHARACTER_SLUG]


async def test_preview_separates_active_and_inactive_characters(
    client, project: Project, tmp_path: Path,
):
    """The headline test: prior frames for the active character are
    counted as 'will wipe'; frames for the inactive character are
    counted as 'will preserve'. This is what the UX promise rests on.
    """
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    project.add_source(vid)
    # Only the default character gets a ref → only that character is active.
    img = tmp_path / "ref.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    project.add_ref(img)

    # 3 frames belong to default (active → wipe), 2 to mio (inactive → preserve).
    for i in range(3):
        _seed(project, filename=f"ep01__yui_{i}",
              character_slug=DEFAULT_CHARACTER_SLUG)
    for i in range(2):
        _seed(project, filename=f"ep01__mio_{i}", character_slug="mio")

    body = (await client.get(f"/api/projects/{project.slug}/sources/0/wipe-preview")).json()
    assert body["to_wipe"]["by_character"] == {DEFAULT_CHARACTER_SLUG: 3}
    assert body["to_wipe"]["total"] == 3
    assert body["to_preserve"]["by_character"] == {"mio": 2}
    assert body["to_preserve"]["total"] == 2
    assert body["preserve_slugs"] == ["mio"]


async def test_preview_counts_rejected_samples_as_wiped(
    client, project: Project, tmp_path: Path,
):
    """Rejected files always wipe regardless of attribution — they're
    diagnostic, not curation. The preview surfaces them under
    rejected_samples so the modal can render them as a separate row,
    visually distinct from per-character data the user might care about."""
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    project.add_source(vid)
    img = tmp_path / "ref.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    project.add_ref(img)
    _seed(project, filename="ep01__r1",
          character_slug=DEFAULT_CHARACTER_SLUG, kept=False)
    _seed(project, filename="ep01__r2",
          character_slug=DEFAULT_CHARACTER_SLUG, kept=False)

    body = (await client.get(f"/api/projects/{project.slug}/sources/0/wipe-preview")).json()
    assert body["to_wipe"]["rejected_samples"] == 2
    assert body["to_wipe"]["total"] == 2
    # Rejected don't get attributed to a character in the by-character map.
    assert body["to_wipe"]["by_character"] == {}


async def test_preview_counts_untracked_files_as_preserved(
    client, project: Project, tmp_path: Path,
):
    """A PNG matching the stem prefix with no metadata row (manual drop,
    legacy file) is preserved because we have no way to attribute it.
    The preview surfaces it under '__untracked__' so the modal can show
    the user how many unattributed files exist."""
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    project.add_source(vid)
    img = tmp_path / "ref.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    project.add_ref(img)
    Image.new("RGB", (8, 8), (5, 5, 5)).save(
        project.kept_dir / "ep01__manual_a.png",
    )
    Image.new("RGB", (8, 8), (5, 5, 5)).save(
        project.kept_dir / "ep01__manual_b.png",
    )

    body = (await client.get(f"/api/projects/{project.slug}/sources/0/wipe-preview")).json()
    assert body["to_preserve"]["by_character"]["__untracked__"] == 2
    assert body["to_preserve"]["total"] == 2


async def test_preview_only_counts_files_for_this_video_stem(
    client, project: Project, tmp_path: Path,
):
    """The wipe runs are per-video; another source's frames must NOT
    appear in this preview. Otherwise the modal would scare the user
    with counts they're not even affecting."""
    vid_a = tmp_path / "ep01.mkv"; vid_a.write_bytes(b"")
    vid_b = tmp_path / "ep02.mkv"; vid_b.write_bytes(b"")
    project.add_source(vid_a)
    project.add_source(vid_b)
    img = tmp_path / "ref.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    project.add_ref(img)
    # Seed frames for both stems; preview for ep01 should only see ep01.
    _seed(project, filename="ep01__a", character_slug=DEFAULT_CHARACTER_SLUG)
    # Manually fake a different video_stem for an "ep02" frame.
    png = project.kept_dir / "ep02__a.png"
    Image.new("RGB", (8, 8), (5, 5, 5)).save(png)
    MetadataLog(project.metadata_path).append(FrameRecord(
        filename="ep02__a", kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep02", character_slug=DEFAULT_CHARACTER_SLUG,
    ))

    body = (await client.get(f"/api/projects/{project.slug}/sources/0/wipe-preview")).json()
    assert body["video_stem"] == "ep01"
    assert body["to_wipe"]["total"] == 1  # just the ep01 frame


async def test_preview_only_counts_files_actually_on_disk(
    client, project: Project, tmp_path: Path,
):
    """Stale kept=True metadata rows whose physical files are no longer
    in kept_dir (moved to rejected/ by dedup, manually deleted, etc.)
    must NOT appear in the preview totals. The modal explains what the
    next Extract / Re-process will wipe — and the wipe only touches
    files actually on disk. Counting metadata-only rows scares the user
    with phantom 'will replace 202 frames' totals when only 20 of those
    files still exist."""
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    project.add_source(vid)
    img = tmp_path / "ref.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    project.add_ref(img)
    # 2 default frames on disk (will-wipe), 1 mio frame on disk (preserve).
    _seed(project, filename="ep01__yui_a", character_slug=DEFAULT_CHARACTER_SLUG)
    _seed(project, filename="ep01__yui_b", character_slug=DEFAULT_CHARACTER_SLUG)
    _seed(project, filename="ep01__mio_a", character_slug="mio")
    # 5 phantom kept=True rows for the ACTIVE character with no file on disk
    # — these must be ignored, not counted as 5 more frames to wipe.
    for i in range(5):
        MetadataLog(project.metadata_path).append(FrameRecord(
            filename=f"ep01__phantom_{i}", kept=True,
            scene_idx=0, tracklet_id=0, frame_idx=0,
            timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
            ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
            score=0.9, video_stem="ep01",
            character_slug=DEFAULT_CHARACTER_SLUG,
        ))
    # 3 phantom kept=True rows for the PRESERVED character with no file on
    # disk — these must also drop out of the preserve total.
    for i in range(3):
        MetadataLog(project.metadata_path).append(FrameRecord(
            filename=f"ep01__phantom_mio_{i}", kept=True,
            scene_idx=0, tracklet_id=0, frame_idx=0,
            timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
            ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
            score=0.9, video_stem="ep01", character_slug="mio",
        ))

    body = (await client.get(f"/api/projects/{project.slug}/sources/0/wipe-preview")).json()
    assert body["to_wipe"]["by_character"] == {DEFAULT_CHARACTER_SLUG: 2}
    assert body["to_wipe"]["total"] == 2
    assert body["to_preserve"]["by_character"] == {"mio": 1}
    assert body["to_preserve"]["total"] == 1


async def test_preview_404_for_unknown_source_index(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/sources/99/wipe-preview")
    assert resp.status_code == 404
