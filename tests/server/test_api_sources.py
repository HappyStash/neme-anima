"""Tests for /api/projects/{slug}/sources routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from neme_extractor.server.app import create_app
from neme_extractor.storage.project import Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "p", name="p")


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


async def test_add_source(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    resp = await client.post(
        f"/api/projects/{project.slug}/sources",
        json={"paths": [str(vid)]},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert len(reloaded.sources) == 1


async def test_add_source_skips_duplicates(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    assert resp.status_code == 200
    body = resp.json()
    # Endpoint reports skipped, not error.
    assert "skipped" in body


async def test_remove_source(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.delete(f"/api/projects/{project.slug}/sources/0")
    assert resp.status_code == 204
    reloaded = Project.load(project.root)
    assert reloaded.sources == []


async def test_patch_excluded_refs(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    img = tmp_path / "ref.png"; img.write_bytes(b"")
    project.add_ref(img)
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.patch(
        f"/api/projects/{project.slug}/sources/0",
        json={"excluded_refs": [str(img.resolve())]},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert reloaded.sources[0].excluded_refs == [str(img.resolve())]


async def test_add_source_accepts_file_uri(client, project: Project, tmp_path: Path):
    vid = tmp_path / "Show E01.mkv"
    vid.write_bytes(b"")
    uri = f"file://{vid.as_posix().replace(' ', '%20')}"
    resp = await client.post(
        f"/api/projects/{project.slug}/sources",
        json={"paths": [uri]},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert len(reloaded.sources) == 1
    assert Path(reloaded.sources[0].path) == vid.resolve()


async def test_add_source_skips_browser_vfs_sentinel(client, project: Project):
    # When the browser hides the path, the frontend may still send the vfs:// fallback;
    # the server should reject it cleanly rather than try to open a junk path.
    resp = await client.post(
        f"/api/projects/{project.slug}/sources",
        json={"paths": ["vfs://Classroom of the Elite - S03E11 [1080p].mkv"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["added"] == []
    assert "vfs://" in body["skipped"][0]


async def test_extract_enqueues_job(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    img = tmp_path / "ref.png"; img.write_bytes(b"")
    project.add_ref(img)
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.post(f"/api/projects/{project.slug}/sources/0/extract")
    assert resp.status_code == 202
    assert "job_id" in resp.json()
