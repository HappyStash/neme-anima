"""Tests for /api/projects/{slug}/refs routes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from httpx import ASGITransport, AsyncClient

from neme_anima.server.app import create_app
from neme_anima.storage.project import Project


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
    # Stored path is the project-internal copy, not the source.
    stored = Path(reloaded.refs[0].path)
    assert stored.parent == (project.root / "refs").resolve()
    assert stored.exists()


async def test_upload_refs_copies_bytes(client, project: Project):
    files = {
        "files": ("portrait.png", b"\x89PNG\r\n\x1a\n--PNGDATA--", "image/png"),
    }
    # httpx.AsyncClient supports `files` kwarg on POST for multipart.
    resp = await client.post(f"/api/projects/{project.slug}/refs/upload", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["added"]) == 1
    saved = Path(body["added"][0])
    assert saved.parent == (project.root / "refs").resolve()
    assert saved.read_bytes().startswith(b"\x89PNG")


async def test_upload_multiple_refs_collide_safely(client, project: Project):
    files = [
        ("files", ("ref.png", b"AAAA", "image/png")),
        ("files", ("ref.png", b"BBBB", "image/png")),
    ]
    resp = await client.post(f"/api/projects/{project.slug}/refs/upload", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["added"]) == 2
    names = sorted(Path(p).name for p in body["added"])
    assert names == ["ref-2.png", "ref.png"]


async def test_get_ref_image_serves_bytes(client, project: Project):
    files = {"files": ("portrait.png", b"\x89PNG\r\nDATA", "image/png")}
    resp = await client.post(f"/api/projects/{project.slug}/refs/upload", files=files)
    saved = Path(resp.json()["added"][0])
    img = await client.get(f"/api/projects/{project.slug}/refs/{saved.name}/image")
    assert img.status_code == 200
    assert img.content == saved.read_bytes()


async def test_get_ref_image_rejects_traversal(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/refs/..%2Fsecret/image")
    # FastAPI decodes the path param; we explicitly reject "/" and "..".
    assert resp.status_code in (400, 404)


async def test_get_ref_image_404_for_unknown(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/refs/missing.png/image")
    assert resp.status_code == 404


async def test_remove_ref_strips_from_excluded(client, project: Project, tmp_path: Path):
    a = _png(tmp_path / "a.png")
    b = _png(tmp_path / "b.png")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/refs", json={"paths": [str(a), str(b)]})
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    # Refs are now stored as copies inside the project — fetch their stored paths.
    after_add = Project.load(project.root)
    a_in_proj = next(r.path for r in after_add.refs if Path(r.path).name == "a.png")
    b_in_proj = next(r.path for r in after_add.refs if Path(r.path).name == "b.png")
    await client.patch(
        f"/api/projects/{project.slug}/sources/0",
        json={"excluded_refs": [a_in_proj, b_in_proj]},
    )
    # Remove 'a'; the source should drop a from its excluded_refs and the file from disk.
    resp = await client.request(
        "DELETE", f"/api/projects/{project.slug}/refs",
        json={"path": a_in_proj},
    )
    assert resp.status_code == 204
    reloaded = Project.load(project.root)
    assert len(reloaded.refs) == 1
    # excluded_refs is the new per-character map; the default character keeps
    # the lone surviving opt-out, the deleted ref is gone.
    assert reloaded.sources[0].excluded_refs == {"default": [b_in_proj]}
    assert not Path(a_in_proj).exists()
    assert Path(b_in_proj).exists()
