"""Tests for /api/projects routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from neme_anima.server.app import create_app
from neme_anima.storage.project import Project


@pytest.fixture
def app(tmp_path: Path):
    return create_app(state_dir=tmp_path / "state")


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_list_empty(client):
    resp = await client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_create_project(client, tmp_path: Path):
    target = tmp_path / "newproj"
    resp = await client.post("/api/projects", json={
        "name": "newproj",
        "folder": str(target),
    })
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["slug"] == "newproj"
    assert (target / "project.json").exists()


async def test_create_rejects_existing_folder(client, tmp_path: Path):
    target = tmp_path / "exists"
    target.mkdir()
    resp = await client.post("/api/projects", json={
        "name": "x", "folder": str(target),
    })
    assert resp.status_code == 409


async def test_get_project_returns_full_state(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.get("/api/projects/p")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == "p"
    assert body["sources"] == []
    assert body["refs"] == []


async def test_get_missing_returns_404(client):
    resp = await client.get("/api/projects/nope")
    assert resp.status_code == 404


async def test_source_view_includes_extraction_cache_state(
    client, tmp_path: Path,
):
    """The Sources tab needs ``extraction_cache`` per source to drive the
    smart Extract / Re-process button states. A freshly-added video with
    no detection cache must report 'none' so the UI keeps Re-process
    disabled and Extract primary."""
    project = Project.create(tmp_path / "p", name="p")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    project.add_source(vid)
    await client.post(
        "/api/projects/register", json={"folder": str(tmp_path / "p")},
    )
    body = (await client.get("/api/projects/p")).json()
    assert body["sources"][0]["extraction_cache"] == "none"


async def test_source_view_reports_current_after_stamp(
    client, tmp_path: Path,
):
    """Stamping the cache snapshot externally (as run_extract does at
    the end of the track stage) must flip the state to 'current' on the
    next project view fetch — that's what unmutes Re-process and mutes
    Extract."""
    from neme_anima.config import Thresholds
    from neme_anima.extraction_cache import stamp_meta

    project = Project.create(tmp_path / "p", name="p")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    project.add_source(vid)
    # Simulate a completed extract by writing a stub parquet + the meta.
    cache_dir = project.cache_dir_for("ep01")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "tracklets.parquet").write_bytes(b"stub")
    stamp_meta(project, "ep01", Thresholds())

    await client.post(
        "/api/projects/register", json={"folder": str(tmp_path / "p")},
    )
    body = (await client.get("/api/projects/p")).json()
    assert body["sources"][0]["extraction_cache"] == "current"


async def test_source_view_reports_stale_after_threshold_drift(
    client, tmp_path: Path,
):
    """If the user changes a scan-affecting threshold (here: detect's
    frame stride) AFTER the cache was stamped, the project view must
    surface 'stale' so the UI can warn the user that Re-process would
    silently use the old detections."""
    from neme_anima.config import Thresholds
    from neme_anima.extraction_cache import stamp_meta

    project = Project.create(tmp_path / "p", name="p")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    project.add_source(vid)
    cache_dir = project.cache_dir_for("ep01")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "tracklets.parquet").write_bytes(b"stub")
    # Stamp with the dataclass defaults, then bump frame_stride in the
    # project's overrides — the next view fetch resolves thresholds with
    # the new stride and sees the snapshot diverge.
    stamp_meta(project, "ep01", Thresholds())
    project.thresholds_overrides = {"detect": {"frame_stride": 8}}
    project.save()

    await client.post(
        "/api/projects/register", json={"folder": str(tmp_path / "p")},
    )
    body = (await client.get("/api/projects/p")).json()
    assert body["sources"][0]["extraction_cache"] == "stale"


async def test_get_registered_but_files_deleted_returns_404(client, tmp_path: Path):
    """Registry entry survives but project folder/files are gone — must 404, not 500."""
    import shutil
    folder = tmp_path / "p"
    Project.create(folder, name="p")
    await client.post("/api/projects/register", json={"folder": str(folder)})
    shutil.rmtree(folder)
    resp = await client.get("/api/projects/p")
    assert resp.status_code == 404, resp.text
    assert "p" in resp.json()["detail"]


async def test_patch_thresholds_overrides(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.patch("/api/projects/p", json={
        "thresholds_overrides": {"identify": {"body_max_distance_loose": 0.22}}
    })
    assert resp.status_code == 200
    reloaded = Project.load(tmp_path / "p")
    assert reloaded.thresholds_overrides["identify"]["body_max_distance_loose"] == 0.22


async def test_patch_llm_config_persists_and_returns_in_view(
    client, tmp_path: Path,
):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})

    # Default view: disabled, default endpoint, no model selected.
    resp = await client.get("/api/projects/p")
    assert resp.json()["llm"] == {
        "enabled": False,
        "endpoint": "http://localhost:1234",
        "model": "",
        "prompt": "",
        "api_key": "",
    }

    # Patch only some fields — others stay untouched.
    resp = await client.patch("/api/projects/p", json={
        "llm": {"enabled": True, "model": "qwen2-vl-7b"}
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm"]["enabled"] is True
    assert body["llm"]["model"] == "qwen2-vl-7b"
    # Endpoint preserved at default since the patch didn't touch it.
    assert body["llm"]["endpoint"] == "http://localhost:1234"

    # Persisted on disk.
    reloaded = Project.load(tmp_path / "p")
    assert reloaded.llm.enabled is True
    assert reloaded.llm.model == "qwen2-vl-7b"


async def test_delete_project_unregisters_only(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.request("DELETE", "/api/projects/p", json={"delete_files": False})
    assert resp.status_code == 204
    # Files still on disk.
    assert (tmp_path / "p" / "project.json").exists()
    # But registry is empty.
    list_resp = await client.get("/api/projects")
    assert list_resp.json() == []


async def test_delete_project_with_files(client, tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    await client.post("/api/projects/register", json={"folder": str(tmp_path / "p")})
    resp = await client.request("DELETE", "/api/projects/p", json={"delete_files": True})
    assert resp.status_code == 204
    assert not (tmp_path / "p").exists()
