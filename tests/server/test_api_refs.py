"""Tests for /api/projects/{slug}/refs routes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
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


def _png(path: Path) -> Path:
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)
    return path


async def test_add_ref(client, project: Project, tmp_path: Path):
    img = _png(tmp_path / "r.png")
    resp = await client.post(
        f"/api/projects/{project.slug}/refs", json={"paths": [str(img)]}
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert len(reloaded.refs) == 1


async def test_remove_ref_strips_from_excluded(client, project: Project, tmp_path: Path):
    a = _png(tmp_path / "a.png")
    b = _png(tmp_path / "b.png")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/refs", json={"paths": [str(a), str(b)]})
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    await client.patch(
        f"/api/projects/{project.slug}/sources/0",
        json={"excluded_refs": [str(a.resolve()), str(b.resolve())]},
    )
    # Remove 'a'; the source should drop a from its excluded_refs too.
    resp = await client.request(
        "DELETE", f"/api/projects/{project.slug}/refs",
        json={"path": str(a.resolve())},
    )
    assert resp.status_code == 204
    reloaded = Project.load(project.root)
    assert len(reloaded.refs) == 1
    assert reloaded.sources[0].excluded_refs == [str(b.resolve())]
