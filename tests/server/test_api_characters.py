"""Tests for /api/projects/{slug}/characters CRUD + character-aware refs/sources.

Covers the new character endpoints plus the ``?character_slug=`` query param
on the existing refs and sources endpoints — the bridge that lets future
character-aware UI components target a specific character while the legacy
mono-character UI still talks to the default character.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from neme_anima.server.app import create_app
from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG, Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "p", name="K-On!")


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


async def test_list_returns_seeded_default_character(client, project: Project):
    """A fresh project always lists exactly one default character — that's
    what the multi-character migration guarantees and what mono-character UI
    callers depend on (so ``characters[0]`` is always populated)."""
    resp = await client.get(f"/api/projects/{project.slug}/characters")
    assert resp.status_code == 200
    chars = resp.json()
    assert len(chars) == 1
    assert chars[0]["slug"] == DEFAULT_CHARACTER_SLUG
    assert chars[0]["name"] == "K-On!"
    assert chars[0]["refs"] == []


async def test_create_character_persists(client, project: Project):
    resp = await client.post(
        f"/api/projects/{project.slug}/characters",
        json={"name": "Mio Akiyama"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Mio Akiyama"
    assert body["slug"] == "mio-akiyama"
    # The state was saved to disk, not just held in memory — reload and check.
    reloaded = Project.load(project.root)
    assert [c.slug for c in reloaded.characters] == [DEFAULT_CHARACTER_SLUG, "mio-akiyama"]


async def test_create_character_explicit_slug(client, project: Project):
    """An explicit slug overrides the name-derived one — useful when the
    display name has unicode the user doesn't want in URLs/paths."""
    resp = await client.post(
        f"/api/projects/{project.slug}/characters",
        json={"name": "ユイ", "slug": "yui"},
    )
    assert resp.status_code == 201
    assert resp.json()["slug"] == "yui"


async def test_create_character_rejects_empty_name(client, project: Project):
    resp = await client.post(
        f"/api/projects/{project.slug}/characters",
        json={"name": "   "},
    )
    assert resp.status_code == 400


async def test_patch_character_updates_name_and_trigger(client, project: Project):
    resp = await client.patch(
        f"/api/projects/{project.slug}/characters/{DEFAULT_CHARACTER_SLUG}",
        json={"name": "Ho-kago Tea Time", "trigger_token": "htt_band"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Ho-kago Tea Time"
    assert body["trigger_token"] == "htt_band"
    reloaded = Project.load(project.root)
    assert reloaded.characters[0].trigger_token == "htt_band"


async def test_delete_character(client, project: Project):
    create = await client.post(
        f"/api/projects/{project.slug}/characters", json={"name": "Mio"},
    )
    assert create.status_code == 201
    resp = await client.delete(f"/api/projects/{project.slug}/characters/mio")
    assert resp.status_code == 204
    listing = (await client.get(f"/api/projects/{project.slug}/characters")).json()
    assert [c["slug"] for c in listing] == [DEFAULT_CHARACTER_SLUG]


async def test_delete_last_character_returns_409(client, project: Project):
    """Refusing to delete the last character maps to 409 (conflict — request
    is well-formed but the operation isn't allowed in the current state)."""
    resp = await client.delete(
        f"/api/projects/{project.slug}/characters/{DEFAULT_CHARACTER_SLUG}",
    )
    assert resp.status_code == 409


async def test_delete_unknown_character_returns_404(client, project: Project):
    resp = await client.delete(f"/api/projects/{project.slug}/characters/nope")
    assert resp.status_code == 404


async def test_add_ref_with_character_slug_targets_that_character(
    client, project: Project, tmp_path: Path,
):
    """``POST /refs?character_slug=mio`` adds the ref to Mio only — the
    default character's ref list stays empty so the legacy ``project.refs``
    alias still reads as expected for mono-character clients."""
    await client.post(
        f"/api/projects/{project.slug}/characters", json={"name": "Mio"},
    )
    img = _png(tmp_path / "mio.png")
    resp = await client.post(
        f"/api/projects/{project.slug}/refs?character_slug=mio",
        json={"paths": [str(img)]},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert len(reloaded.character_by_slug("mio").refs) == 1
    assert len(reloaded.character_by_slug(DEFAULT_CHARACTER_SLUG).refs) == 0


async def test_add_ref_with_unknown_character_returns_404(
    client, project: Project, tmp_path: Path,
):
    img = _png(tmp_path / "r.png")
    resp = await client.post(
        f"/api/projects/{project.slug}/refs?character_slug=ghost",
        json={"paths": [str(img)]},
    )
    assert resp.status_code == 404


async def test_patch_excluded_refs_with_character_slug(
    client, project: Project, tmp_path: Path,
):
    """PATCH /sources/{idx}?character_slug=mio writes Mio's opt-outs without
    disturbing the default character's. Verifies the per-character map is
    correctly keyed and survives a reload."""
    await client.post(
        f"/api/projects/{project.slug}/characters", json={"name": "Mio"},
    )
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    img = _png(tmp_path / "mio_ref.png")
    await client.post(
        f"/api/projects/{project.slug}/refs?character_slug=mio",
        json={"paths": [str(img)]},
    )
    await client.post(
        f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]},
    )
    reloaded = Project.load(project.root)
    mio_ref_path = reloaded.character_by_slug("mio").refs[0].path

    resp = await client.patch(
        f"/api/projects/{project.slug}/sources/0?character_slug=mio",
        json={"excluded_refs": [mio_ref_path]},
    )
    assert resp.status_code == 200
    final = Project.load(project.root)
    assert final.sources[0].excluded_refs == {"mio": [mio_ref_path]}


async def test_project_view_exposes_characters_array(client, project: Project):
    """The full project view includes the characters list so future UI can
    render it without a separate fetch — while keeping the top-level ``refs``
    alias intact for mono-character clients."""
    resp = await client.get(f"/api/projects/{project.slug}")
    assert resp.status_code == 200
    body = resp.json()
    assert "characters" in body
    assert len(body["characters"]) == 1
    assert body["characters"][0]["slug"] == DEFAULT_CHARACTER_SLUG
    # Backwards-compat: top-level refs still mirrors the default character's.
    assert "refs" in body
