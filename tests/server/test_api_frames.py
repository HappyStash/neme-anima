"""Tests for /api/projects/{slug}/frames routes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from httpx import ASGITransport, AsyncClient

from neme_extractor.server.app import create_app
from neme_extractor.storage.metadata import FrameRecord, MetadataLog
from neme_extractor.storage.project import Project


@pytest.fixture
def project_with_frames(tmp_path: Path) -> Project:
    p = Project.create(tmp_path / "p", name="p")
    # Two synthetic kept frames so listing has something to return.
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for stem, fi in [("ep01", 10), ("ep02", 20)]:
        name = f"{stem}__s000_t001_f{fi:06}"
        Image.fromarray(img).save(p.kept_dir / f"{name}.png")
        (p.kept_dir / f"{name}.txt").write_text("1girl, smile\n")
        MetadataLog(p.metadata_path).append(FrameRecord(
            filename=name, kept=True,
            scene_idx=0, tracklet_id=1, frame_idx=fi,
            timestamp_seconds=fi / 24.0,
            bbox=(0, 0, 16, 16),
            ccip_distance=0.1, sharpness=10.0, visibility=1.0, aspect=0.95,
            score=0.9, video_stem=stem,
        ))
    return p


@pytest.fixture
def app(tmp_path: Path, project_with_frames: Project):
    a = create_app(state_dir=tmp_path / "state")
    a.state.registry.register(project_with_frames)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_list_all_frames(client, project_with_frames: Project):
    resp = await client.get(f"/api/projects/{project_with_frames.slug}/frames")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    filenames = sorted(f["filename"] for f in body["items"])
    assert filenames[0].startswith("ep01__")
    assert filenames[1].startswith("ep02__")


async def test_list_filtered_by_source(client, project_with_frames: Project):
    resp = await client.get(
        f"/api/projects/{project_with_frames.slug}/frames",
        params={"source": "ep02"},
    )
    body = resp.json()
    assert body["count"] == 1
    assert body["items"][0]["filename"].startswith("ep02__")


async def test_get_tags(client, project_with_frames: Project):
    name = "ep01__s000_t001_f000010"
    resp = await client.get(f"/api/projects/{project_with_frames.slug}/frames/{name}/tags")
    assert resp.status_code == 200
    assert resp.json()["text"] == "1girl, smile"


async def test_put_tags_overwrites(client, project_with_frames: Project):
    name = "ep01__s000_t001_f000010"
    resp = await client.put(
        f"/api/projects/{project_with_frames.slug}/frames/{name}/tags",
        json={"text": "1girl, blue_hair"},
    )
    assert resp.status_code == 200
    txt = (project_with_frames.kept_dir / f"{name}.txt").read_text(encoding="utf-8")
    assert txt == "1girl, blue_hair\n"


async def test_delete_frame_removes_png_and_txt(client, project_with_frames: Project):
    name = "ep01__s000_t001_f000010"
    resp = await client.delete(f"/api/projects/{project_with_frames.slug}/frames/{name}")
    assert resp.status_code == 204
    assert not (project_with_frames.kept_dir / f"{name}.png").exists()
    assert not (project_with_frames.kept_dir / f"{name}.txt").exists()


async def test_bulk_delete(client, project_with_frames: Project):
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-delete",
        json={"filenames": [
            "ep01__s000_t001_f000010", "ep02__s000_t001_f000020",
        ]},
    )
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    assert sorted(p.name for p in project_with_frames.kept_dir.iterdir()) == []


async def test_bulk_tags_replace_uses_regex(client, project_with_frames: Project):
    name = "ep01__s000_t001_f000010"
    # Write a known tag set first.
    (project_with_frames.kept_dir / f"{name}.txt").write_text("red_eyes, blue_hair\n")
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-tags-replace",
        json={
            "filenames": [name],
            "pattern": r"red_eyes",
            "replacement": "ruby_eyes",
            "case_insensitive": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["changed"] >= 1
    text = (project_with_frames.kept_dir / f"{name}.txt").read_text(encoding="utf-8")
    assert "ruby_eyes" in text
    assert "red_eyes" not in text


async def test_bulk_tags_replace_invalid_regex_returns_422(
    client, project_with_frames: Project
):
    resp = await client.post(
        f"/api/projects/{project_with_frames.slug}/frames/bulk-tags-replace",
        json={"filenames": [], "pattern": "[unclosed", "replacement": ""},
    )
    assert resp.status_code == 422


async def test_get_frame_image(client, project_with_frames: Project):
    name = "ep01__s000_t001_f000010"
    resp = await client.get(f"/api/projects/{project_with_frames.slug}/frames/{name}/image")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert len(resp.content) > 0
